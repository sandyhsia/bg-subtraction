import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import copy as copy
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import myVizTool as vis_util

# This is needed since we need the sublibraries under research and object_detection
init_path = os.getcwd()
PATH_TO_TENSORFLOW_OBJDETECTION = '/home/dongqxia/softwares/tensorflow/models/research/object_detection'
os.chdir(PATH_TO_TENSORFLOW_OBJDETECTION)
print(os.path.isdir('utils'))
print(os.getcwd())
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

PATH_TO_TEST_IMAGES_DIR = '/home/dongqxia/projects/bgsubtraction/submission/debug/test_B'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '00{}.png'.format(i)) for i in range(2550, 2700) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                            'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main():
    segmentation(TEST_IMAGE_PATHS, PATH_TO_LABELS, PATH_TO_FROZEN_GRAPH)


def segmentation(test_image_paths, path_to_labels, path_to_frozen_graph):
    ##uncomment the code when you want to down load the model file
    # opener = urllib.request.URLopener()
    # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #     file_name = os.path.basename(file.name)
    #     if 'frozen_inference_graph.pb' in file_name:
    #         tar_file.extract(file, os.getcwd())

    ## 
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    for image_path in test_image_paths:
        image = Image.open(image_path)
        # image = image.resize((int(np.shape(image)[0]*2), int(np.shape(image)[1]*2)), Image.BILINEAR)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()

def segmentOne(one_image_np, minScore=0.6, idList=[1], path_to_labels=PATH_TO_LABELS, path_to_frozen_graph=PATH_TO_FROZEN_GRAPH):
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    # Actual detection.
    output_dict = run_inference_for_single_image(one_image_np, detection_graph)
    
    # check class and scores
    valid_output_dict = {'detection_masks':[]}
    for i in range(len(output_dict['detection_classes'])):
        if output_dict['detection_classes'][i] in idList and output_dict['detection_scores'][i] > minScore:
            valid_output_dict['detection_masks'].append(output_dict['detection_masks'][i])

    
    return valid_output_dict

def testSegmentOne():
    import cv2
    img = cv2.imread('../debug/init/000900.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = segmentOne(img)
    cv2.imshow(mask)



if __name__ == '__main__':
    main()
    # testSegmentOne()