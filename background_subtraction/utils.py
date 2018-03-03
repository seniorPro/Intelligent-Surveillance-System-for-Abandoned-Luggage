import numpy as np
import cv2


def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def boxes_intersect(bbox1, bbox2):  # Return if two rect overlap
    return ((np.abs(bbox1[0]-bbox2[0])*2) < (bbox1[2]+bbox2[2])) and \
           ((np.abs(bbox1[1]-bbox2[1])*2) < (bbox1[3]+bbox2[3]))


def rect_similarity2(r1, r2):
    if boxes_intersect(r1, r2):     # Return if r1 and r2 satisfy overlapping criterion
        if similarity_measure_rect(r1, r2) > 0.5:   # return similarity
            return True
        return False
    return False


def similarity_measure_rect(bbox_test, bbox_target):
    # Return similarity measure between two bounding box
    def gen_box(bbox):
        from shapely.geometry import box
        box = box(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        return box
    bbtest = gen_box(bbox_test)
    bbtarget = gen_box(bbox_target)
    return bbtarget.intersection(bbtest).area/bbtarget.union(bbtest).area


def norm_correlate(a, v):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)
    return np.correlate(a, v)


def draw_bounding_box(image, bbox):
    # Draw all bounding box inside image as red rectangle
    for s in bbox:
        cv2.rectangle(image, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), 255, 1)
    return image


def draw_bounding_box2(image, bbox):
    # Draw all bounding box inside image as red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, 1)
    return image


def dim_image(image, bbox):
    # First make bbox bigger
    divisor = 4
    bbox_area_to_add = [x / divisor for x in bbox]
    bbox_area_to_add = [int(x) for x in bbox_area_to_add]
    (x1, y1, w1, h1) = bbox_area_to_add
    (x, y, w, h) = bbox
    x -= x1
    y -= y1
    w += 2*x1
    h += 2*y1
    (height, width, channel) = image.shape

    for a in range (0, height):
        for b in range(0, width):
            if a < y:
                image[a, b] = (0, 0, 0)
            elif a > (y + h):
                image[a, b] = (0, 0, 0)
            elif b < x:
                image[a, b] = (0, 0, 0)
            elif b > (x + w):
                image[a, b] = (0, 0, 0)
            else:
                continue
    return image

def dim_image2(image, bbox):
    # First make bbox bigger
    divisor = 4
    bbox_area_to_add = [x / divisor for x in bbox]
    bbox_area_to_add = [int(x) for x in bbox_area_to_add]
    (x1, y1, w1, h1) = bbox_area_to_add
    (x, y, w, h) = bbox
    x -= x1
    y -= y1
    w += 2*x1
    h += 2*y1
    (height, width, channel) = image.shape

    for a in range (0, height):
        for b in range(0, width):
            if a > y and a < (y + h) and b > x and b < (x + w):
                image[a, b] = (0, 0, 0)
            else:
                continue
    return image

def reverse_image(curr, image):
    (height, width, channel) = curr.shape
    for a in range (0, height):
        for b in range(0, width):
            if np.any(image[a, b] == 0):
                image[a, b] = curr[a, b]
            else:
                image[a, b] = (0, 0, 0)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)