import numpy as np
import os
import sys
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import datetime
import cv2
import imutils


cap = cv2.VideoCapture("/home/pcroot/Documents/documents/test.mp4")

sys.path.append("..")
MODEL_NAME = '/home/pcroot/Documents/models/research/object_detection/left_luggage/final2_training'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/pcroot/Documents/models/research/object_detection/training', 'object-detection.pbtxt')

NUM_CLASSES = 2



class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
    
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
    
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        config = tf.ConfigProto(device_count = {'GPU': 0})
        

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph,config=config) as sess:
        fps = FPS().start()
        while True:
            ret, image_np = cap.read()
            image_np = imutils.resize(image_np, width=450)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = np.dstack([image_np, image_np, image_np])
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            k = cv2.waitKey(25)
            fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.stop()
        cv2.destroyAllWindows()
