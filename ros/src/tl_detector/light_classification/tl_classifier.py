from styx_msgs.msg import TrafficLight

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
## from matplotlib import pyplot as plt
## from PIL import Image
from os import path
## import time
import cv2


MINIMUM_CONFIDENCE = 0.6
CATEGORY_INDEX = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'None'}}


## def load_image_into_numpy_array(image):
##    (im_width, im_height) = image.size
##    return np.array(image.getdata()).reshape(
##        (im_height, im_width, 3)).astype(np.uint8)


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        #--------Load a (frozen) Tensorflow model into memory
        # Path to frozen detection graph. This is the actual model that is used for the object detection.

        #MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
        MODEL_NAME = 'ssd_inception_v2_coco'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph = detection_graph)

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.tensors = (sess, image_tensor, detection_boxes, detection_scores, detection_classes)
        print('\n\n * * * * * SUCCESSFULLY LOADED THE GRAPH * * * * * \n\n')

        ####
        #Test/Dummy call to get_classification() so that the tensorflow will initialize and classify faster
        # when the real image comes from the simulator.
        ####
        test_img = 'img_6.jpg'
        img = cv2.imread(test_img)
        self.get_classification(img)
        print("\n\n Get Classification is called once  to initialize Tensorflow \n\n")




    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        class_name = ''

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.

        ## since image is already numpy array, we remove this
        ## image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ## image_np_expanded = np.expand_dims(image_np, axis=0)
        image_np_expanded = np.expand_dims(image, axis=0)

        (sess, image_tensor, detection_boxes, detection_scores, detection_classes) = self.tensors

        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes,
                                             detection_scores,
                                             detection_classes],
                                            feed_dict={image_tensor: image_np_expanded})

        for i in range(len(boxes[0])):
            if scores is None or scores[0][i] > MINIMUM_CONFIDENCE:
                class_name = CATEGORY_INDEX[classes[0][i]]['name']
        
        #Debug
        #print("Traffic light is:{}".format(class_name))

        if (class_name == 'Red'):
            return TrafficLight.RED
        elif (class_name == 'Green'):
            return TrafficLight.GREEN
        elif (class_name == 'Yellow'):
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN