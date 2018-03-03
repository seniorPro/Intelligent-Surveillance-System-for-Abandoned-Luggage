import copy
from imutils.video import FPS
import imutils
from static_object import *
from intensity_processing import *
from pathlib import Path

import numpy as np
import os
import sys
import cv2
import tensorflow as tf
from queue import Queue

from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from app_utils import draw_boxes_and_labels

sys.path.append("..")

MODEL_NAME = '/home/pcroot/Documents/models/research/object_detection/left_luggage/final2_training'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/pcroot/Documents/models/research/object_detection/training', 'object-detection.pbtxt')

NUM_CLASSES = 2
gamma = 1.7

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(image_np, sess, detection_graph):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals, old_frame, current_frame):
    bbox_to_add = []
    if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
        for old in bbox_last_frame_proposals:
            old_drawn = False
            for curr in bbox_current_frame_proposals:
                if rect_similarity2(old, curr):
                    old_drawn = True
                    break
            if not old_drawn:
                # Check if the area defined by the bounding box in the old frame and in the new one is still the same
                old_section = old_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                new_section = current_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                if norm_correlate(old_section, new_section)[0] > 0.9:
                    bbox_to_add.append(old)
    return bbox_to_add





if __name__ == '__main__':
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
        sess = tf.Session(graph=detection_graph, config=config)
        my_file = Path("/home/pcroot/Documents/documents/videos/out1.mp4")
        if not my_file.is_file():
            print("Video does not exist")
            exit()
            
        stream = cv2.VideoCapture("/home/pcroot/Documents/documents/videos/use8.mp4")
#        stream = cv2.VideoCapture(0)
        fps = FPS().start()
        first_run = True
        (ret, frame) = stream.read()
        while not ret:
            (ret, frame) = stream.read()
    
        frame = imutils.resize(frame, width=450)
        adjusted = adjust_gamma(frame, gamma=gamma) # gamma correction
        frame = adjusted
        (height, width, channel) = frame.shape
        image_shape = (height, width)
        rgb = IntensityProcessing(image_shape)
    
        bbox_last_frame_proposals = []
        static_objects = []
        count=0
        n_frame=0
        while 1:
            (ret, frame) = stream.read()
            if not ret:
                break
            else:
                frame = imutils.resize(frame, width=450)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.dstack([frame, frame, frame])
    
                rgb.current_frame = frame  # .getNumpy()
                if first_run:
                    old_rgb_frame = copy.copy(rgb.current_frame) # old frame is the new frame
                    first_run = False
    
                rgb.compute_foreground_masks(rgb.current_frame)  # compute foreground masks
                rgb.update_detection_aggregator()   # detect if new object proposed
    
                rgb_proposal_bbox = rgb.extract_proposal_bbox()     # bounding boxes of the areas proposed
                foreground_rgb_proposal = rgb.proposal_foreground   # rgb proposals
    
                bbox_current_frame_proposals = rgb_proposal_bbox
                final_result_image = rgb.current_frame.copy()
    
                old_bbox_still_present = check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals,
                                                              old_rgb_frame, rgb.current_frame.copy())
    
                # add the old bbox still present in the current frame to the bbox detected
                bbox_last_frame_proposals = bbox_current_frame_proposals + old_bbox_still_present
    
                '''
                
                    BBOX_LAST_FRAME_PROPOSALS  BU FRAME DE OLAN TUM BBOX DEGERLERI LISTE OLARAK VAR.
                    
                '''
                old_rgb_frame = rgb.current_frame.copy()
    
                # static object ######################
                if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
                    for old in bbox_last_frame_proposals:
                        old_drawn = False
                        for curr in static_objects:
                            if rect_similarity2(curr.bbox_info, old):
                                old_drawn = True
                                break
                        if not old_drawn:
                            owner_frame = rgb.current_frame.copy()
                            # draw_bounding_box2(owner_frame, old)
                            count+=1
                            
                            frame_rgb = cv2.cvtColor(dim_image(owner_frame, old), cv2.COLOR_BGR2RGB)

#                            image_np = load_image_into_numpy_array(owner_frame)
                            data = detect_objects(frame_rgb, sess, detection_graph)
                            rec_points = data['rect_points']
                            class_names = data['class_names']
                            class_colors = data['class_colors']
                            for point, name, color in zip(rec_points, class_names, class_colors):
                                cv2.rectangle(rgb.current_frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                                              (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
                                cv2.rectangle(rgb.current_frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                                              (int(point['xmin'] * width) + len(name[0]) * 6,
                                               int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
                                cv2.putText(rgb.current_frame, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.3, (0, 0, 0), 1)
                            cv2.imshow('Final Result', rgb.current_frame)
                                                    
                            
                            
                            static_objects.append(StaticObject(old, owner_frame, 0))
                
                count=0
    

    
                cv2.imshow('Original Frame', final_result_image)
                cv2.imshow('Background Modelling Result', foreground_rgb_proposal)
#                cv2.imshow('frame', frame)
                n_frame+=1
            k = cv2.waitKey(25)
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        print("[INFO] number of frame: {}".format(n_frame))


        stream.release()
    cv2.destroyAllWindows()



