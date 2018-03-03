# Dual Background Porikli method using Zivkovich model background for (RGB) Parameters
BG_ZIV_LONG_LRATE = 0.0001      #: Background learning rate in Zivkovich method for long background model
BG_ZIV_SHORT_LRATE = 0.002      #: Background learning rate in Zivkovich method for short background model
BG_ZIV_HIST = 1                 #: History for Zivkovich background method
BG_ZIV_LONG_THRESH = 1000       #: Threshold for Zivkovich method for long background model
BG_ZIV_SHORT_THRESH = 120       #: Threshold for Zivkovich method for short background model

# Aggregator parameters
AGG_RGB_MAX_E = 15              #: number of frames after which a pixel is considered an left item in rgb domain
AGG_RGB_PENALTY = 7             #: penalty in the accumulator for a pixel not in current foreground in rgb domain

# Bounding Boxes
BBOX_MIN_AREA = 100              #: minimum area in pixel to create a bounding box

