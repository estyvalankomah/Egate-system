import cv2
import os
from lib import detect_number_plate, extract_number_plate_text
from flask import Flask, Response, render_template
import argparse
import datetime
import threading
import urllib.request, json

outputFrame = None
lock = threading.Lock()
reg_number = ""
user_id = ""
roi = None

app = Flask(__name__)

vs = cv2.VideoCapture('Egate_vidoe_feed.mp4')
vs.set(3,1280)
vs.set(4,720)
vs.set(10,70)

@app.route("/")
def index():
	global reg_number, user_id

	searchPlate = "http://localhost:8080/api/subscriptions?number_plate={}".format(reg_number)
	searchUser = "http://localhost:8080/api/users/{}".format(user_id)
	response = urllib.request.urlopen(searchPlate)
	data = response.read()
	dict = json.loads(data)
	getValues = lambda key,dict: [subVal[key] for subVal in dict if key in subVal]
	user_id = getValues('user_id', dict)
	user_id = user_id[0]
	if user_id != "":
		response = urllib.request.urlopen(searchUser)
		data = response.read()
		user_details = json.loads(data)
	# return the rendered template
	return render_template("index.html",results=user_details, reg_number=reg_number)

def detect_plate():
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, reg_number, roi
	# initialize the motion detector and the total number of frames
	# read thus far

	while True:
			# read the next frame from the video stream, resize it,
			# convert the frame to grayscale, and blur it
			success,frame = vs.read()
			car_roi = frame[150:600, 100: ]

			detections, image_np_with_detections, boxes = detect_number_plate(car_roi)
			extracted_plate = extract_number_plate_text(image_np_with_detections,detections ,0.3, boxes)

			with lock:
				outputFrame = image_np_with_detections.copy()
				reg_number = extracted_plate
				# roi = region

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	# ap.add_argument("-f", "--frame-count", type=int, default=32,
	# 	help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_plate)
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=True)



 