import bg_models
from const import *
from utils import *
import cv2


class IntensityProcessing:
    def __init__(self, image_shape):
        shape_rgb = (image_shape + (3,))     # shape have 3 channels

        self.current_frame = np.zeros(shape=shape_rgb, dtype=np.uint8)
        self.proposal_foreground = np.zeros(shape=shape_rgb, dtype=np.uint8)
        self.foreground_mask_long_term = np.zeros(shape=image_shape)
        self.foreground_mask_short_term = np.zeros(shape=image_shape)
        self.background_aggregator = np.zeros(shape=image_shape, dtype=np.int8)
        self.proposal_mask = np.zeros(shape=image_shape, dtype=np.uint8)  # mask from aggregator

        # define Zivkovich background subtraction function LONG and SHORT
        self.f_bg_long = cv2.createBackgroundSubtractorMOG2(BG_ZIV_HIST, BG_ZIV_LONG_THRESH, False)
        self.f_bg_short = cv2.createBackgroundSubtractorMOG2(BG_ZIV_HIST, BG_ZIV_SHORT_THRESH, False)

    def compute_foreground_masks(self, frame):
        # Compute foreground masks for long term background and short term background following Porikli's method
        self.foreground_mask_long_term = bg_models.compute_foreground_mask_from_func(self.f_bg_long, frame,
                                                                                     BG_ZIV_LONG_LRATE)
        self.foreground_mask_short_term = bg_models.compute_foreground_mask_from_func(self.f_bg_short, frame,
                                                                                      BG_ZIV_SHORT_LRATE)
        self.foreground_mask_long_term = bg_models.apply_dilation(self.foreground_mask_long_term, 1, cv2.MORPH_ELLIPSE)
        self.foreground_mask_short_term = bg_models.apply_dilation(self.foreground_mask_short_term, 1,
                                                                   cv2.MORPH_ELLIPSE)

        return self.foreground_mask_long_term, self.foreground_mask_short_term

    def update_detection_aggregator(self):
        # Update aggregator with the provided foregrounds.
        # If a pixel is in foreground_long but not in foreground_short increment its accumulator otherwise decrement it.
        # If a particular area has already been detected as proposal don't decrement if the above condition is not
        # verified.
        proposal_candidate = self.foreground_mask_long_term * np.int8(np.logical_not(self.foreground_mask_short_term))
        other_cases = np.int8(np.logical_not(proposal_candidate))
        result = self.background_aggregator + proposal_candidate
        result -= other_cases * AGG_RGB_PENALTY  # - mask_penalty * (AGG_RGB_MAX_E-1)
        self.background_aggregator = np.clip(result, 0, AGG_RGB_MAX_E)
        return self.background_aggregator

    def extract_proposal_bbox(self):
        # Extract RGB proposal as the bounding boxes of the areas of the accumulator that have reached a
        # value of AGG_RGB_MAX_E
        self.proposal_mask = np.where(self.background_aggregator == AGG_RGB_MAX_E, 1, 0)
        bbox = bg_models.get_bounding_boxes(self.proposal_mask.astype(np.uint8))
        self.proposal_foreground = bg_models.cut_foreground(self.current_frame, self.proposal_mask)
        return bbox
