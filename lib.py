import os
import tensorflow as tf
import cv2
import easyocr
import numpy as np
from bbox import BBox2D
from matplotlib import pyplot as plt
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Convert this part intZAaawwwwwwwo a function/method
def detect_number_plate(img):
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detection_threshold = 0.3
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]

    width = image_np_with_detections.shape[1]
    height = image_np_with_detections.shape[0]

    try:
        for idx, box in  enumerate(boxes):
            roi = box * [height, width, height, width]
            region_height = int(roi[2]) - int(roi[0])
            region_width = int(roi[3]) - int(roi[1])
            if((region_height not in range(30,60)) and (region_width not in range(70,125))):
                detections.pop(idx)
               
            viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.3,
                    agnostic_mode=False)

    except Exception as e:
        pass

    # cv2.imshow("Plate",image_np_with_detections)
    # cv2.waitKey(1)
    return detections, image_np_with_detections, boxes
    # return detections['detection_boxes']

def formatStringFromNumberPlate(input):
    size = len(input)
    
    for i in range(size):
        input[i] = input[i].upper()

    if size == 1:
        _string = input[0]
        _string = _string.replace("O","", 1)
        _StringArr = _string.split("-")
        _StringArr[0] = _StringArr[0].rstrip()
        new_StringArr = []
        try:
            for i in _StringArr[1]:
                if i.isnumeric() and len(new_StringArr) < 2:
                    new_StringArr.append(i)
            year = new_StringArr[0] + new_StringArr[1]
            _string = _StringArr[0] + "-" + year
            _string = _string.replace(" ", "")
            return _string
        except Exception as e:
            print(e)
    if size == 2:
        return input[0].replace("O","")
    if size == 3:
        str = ["GH", "CH", "GF", "CF"]
        for x in str:
            if x in input:
                input.remove(x)
        _string = input[0] + input[1]
        _string = _string.replace("O","", 1)
        _string = _string.replace(" ", "")
        return _string
    if size == 4:
        for x in input:
            if len(x) > 10:
                input.remove(x)
        str = ["GH", "CH", "GF", "CF"]
        for i in str:
            if i in input:
                input.remove(i)
        if len(input) == 3:
            if input[2].isnumeric():
                _string = input[0] + input[1] + "-" + input[2]
                _string = _string.replace("O","", 1)
                _string = _string.replace(" ", "")
                return _string
            if input[2].isalpha():
                _string = input[0] + input[1] + input[2]
                _string = _string.replace("O","", 1)
                _string = _string.replace(" ", "")
                return _string
        if len(input) == 2:
            _string = input[0] + input[1]
            _string = _string.replace("O","", 1)
            _string = _string.replace(" ", "")
            return _string
        

def extract_number_plate_text(img, detections, detection_threshold, boxes):
    image = img
    # scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    # boxes = detections['detection_boxes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]

    for box in boxes:
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        region_height = int(roi[2]) - int(roi[0])
        region_width = int(roi[3]) - int(roi[1])
        if((region_height in range(30,60)) and (region_width in range(70,125))):
            reader = easyocr.Reader(['en'])
            ocr_result = reader.readtext(region, detail=0)
            extracted_plate = formatStringFromNumberPlate(ocr_result)

            return extracted_plate
