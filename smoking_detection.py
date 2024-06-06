'''Tensorflow 2.2 object detection bu Prof. KIm Byung-Gyu on 2020. 07. 25
   With local downloaded models from tensorflow :
   GitHub https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import warnings
#warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)               #-- Webcam based-case
#cap = cv2.VideoCapture("<video-path>")  #-- video file based-case

#-- Control the camera resolution with CAM
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

print("[INFO] TF verion = ",tf.__version__)

#--- Model Preparation ----#
def load_model(model_name):
  #--- 1) tf 1.x model base url
  #base_url = 'http://download.tensorflow.org/models/object_detection/'
  '''model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)
  #model_dir = pathlib.Path(model_dir)/"saved_model"
  '''

  #--- 2) tf 2.2 model base url
  #base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' #not working....!!!
  #model_file = model_name + '.tar.gz'
  #model_dir = tf.keras.utils.get_file(
  #  fname=model_name,
  #  origin=base_url + model_file,
  #  untar=True)
  #model_dir = pathlib.Path(model_dir)/"saved_model"
  model_dir = "local_models/trained_model_large_original_20000/saved_model"

  #--- 3) tf 2.2 local saved model after downloading
  '''model_dir = 'local_models/efficientdet_d1_coco17_tpu-32'  # efficientdet_d1
  model_dir = 'local_models/'+model_name  # efficientdet_d1

  model_dir = pathlib.Path(model_dir)/"saved_model"
  '''
  print('[INFO] Loading the modle from '+ str(model_dir))
  model = tf.saved_model.load(str(model_dir))

  return model

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
#PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
PATH_TO_LABELS = 'data/cigarette_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)#, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

#--- Detection --#
#model_name =  'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'#'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8' #'efficientdet_d0_coco17_tpu-32' #'ssd_mobilenet_v1_coco_2017_11_17'
#print('[INFO] Downloading model and loading to network : '+ model_name)
model_name = "hello"  # 이제 모델명은 별로 필요 없는 듯?
detection_model = load_model(model_name)
#print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))

def run_inference(model, cap):
    fn=0
    while cap.isOpened():
        ret, image_np = cap.read()
        # Actual detection.
        print("[INFO]"+str(fn)+"-th frame -- Running the inference and showing the result....!!!")
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', image_np)
        fn=fn+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

#-- Run it on each test image and show the results: --#
#for image_path in TEST_IMAGE_PATHS:
#  show_inference(detection_model, image_path)

#-- Web camp-based detection run and show the results...!!! ---#
run_inference(detection_model, cap)
