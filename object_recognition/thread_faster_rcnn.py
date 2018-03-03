import numpy as np
import os
import sys
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import datetime
import cv2
import time
from threading import Thread
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

sys.path.append("..")

MODEL_NAME = '/home/pcroot/Desktop/models/research/object_detection/left_luggage/final2_training'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/pcroot/Desktop/models/research/object_detection/training', 'object-detection.pbtxt')

NUM_CLASSES = 2
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue



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


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
    
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                
                # add the frame to the queue
                self.Q.put(frame)
    
    def read(self):
        # return next frame in the queue
        return self.Q.get()
    
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


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
    with tf.Session(graph=detection_graph,config=tf.ConfigProto(log_device_placement=True)) as sess:
        cap = FileVideoStream("/home/pcroot/Documents/documents/demo1.mp4").start()
        time.sleep(0.2)
        fps = FPS().start()
        while cap.more():
            image_np = cap.read()
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
            
            cv2.imshow('object detection', image_np)
            k = cv2.waitKey(25)
            fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.stop()
        cv2.destroyAllWindows()
