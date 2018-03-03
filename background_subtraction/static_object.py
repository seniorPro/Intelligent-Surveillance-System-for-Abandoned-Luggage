from utils import *


class StaticObject:
    def __init__(self, bbox, owner_frame, obj_type):
        self.bbox_info = bbox
        self.owner = owner_frame.copy()
        self.object_type = obj_type

    def print_object(self):
        # print(self.bbox_info, self.object_type)

        cv2.imshow("Owner Frame", self.owner)
        # cv2.waitKey(1000)
        cv2.destroyWindow("Owner Frame")
