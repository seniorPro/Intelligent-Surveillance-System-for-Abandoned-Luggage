import cv2
import numpy as np
import utils
from const import *


# get rgb background
def compute_foreground_mask_from_func(f_bg, current_frame, alpha):
    foreground = np.zeros(shape=current_frame.shape, dtype=np.uint8)
    foreground = f_bg.apply(current_frame, foreground, alpha)
    foreground = np.where((foreground == 0), 0, 1)
    return foreground


def cut_foreground(image, mask):
    # Cut the foreground from the image using the mask supplied
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image * mask
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image * utils.to_rgb(mask)
    else:
        raise IndexError("image has the wrong number of channels (must have 1 or 3 channels")


def apply_dilation(image, kernel_size, kernel_type):
    # Apply dilation to image with the specified kernel type and image
    u_image = image.astype(np.uint8)
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
    return u_image


def get_bounding_boxes(image):
    # Return Bounding Boxes in the format x,y,w,h where (x,y) is the top left corner
    bbox = []
    im, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > BBOX_MIN_AREA:
            rect = cv2.boundingRect(cnt)
            if rect not in bbox:
                bbox.append(rect)

    return bbox
