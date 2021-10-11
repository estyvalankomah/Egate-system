import cv2
import os
from lib import detect_number_plate, extract_number_plate_text
thres = 0.6 # Threshold to detect object
detection_threshold = 0.3

img = cv2.imread('./frames/kang104.jpg')
car_roi = img[150:600, 100: ]
detections, image_np_with_detections, boxes = detect_number_plate(car_roi)

print(len(image_np_with_detections))

# cap = cv2.VideoCapture('Egate_vidoe_feed.mp4')
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,70)

# classNames= []
# classFile = 'coco.names'
# with open(classFile,'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'frozen_inference_graph.pb'


# net = cv2.dnn_DetectionModel(weightsPath,configPath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/ 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)

index = 0

# while True:
#     success,img = cap.read()
#     car_roi = img[150:600, 100: ]
#     my_roi = img[200:370 , 200:]
#     plateRoi = img[371:600, 100:]
#     classIds, confs, bbox = net.detect(car_roi,confThreshold=thres)

#     if len(classIds) != 0:
#         for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#             if (classId == 3) :
#                 detections, image_np_with_detections, boxes = detect_number_plate(car_roi)
#                 extracted_plate = extract_number_plate_text(image_np_with_detections,detections ,detection_threshold, boxes)
#                 print(detections['detection_boxes'])
                # cv2.rectangle(car_roi,box,color=(0,255,0),thickness=2)
                # cv2.putText(car_roi,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                # cv2.putText(car_roi,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # cv2.imshow("Plate",image_np_with_detections)
    # cv2.imshow("Output",region)
    # cv2.imshow("Output",plateRoi)
    # cv2.waitKey(1)

    
    