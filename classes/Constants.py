import glob
import numpy as np
import os

class Constants:

    BASE_PATH = os.getcwd() + '/'

    ## CAMERA CAL CONSTS
    CAMCAL = {
        'IMAGES': glob.glob(BASE_PATH + 'media/camera_cal/calibration*.jpg'),
        'NX': 9,
        'NY': 6
    }

    ## PERSPECTIVE CONSTS
    PERSPECTIVE = {
        'SRC_POINTS': np.float32([
            [595,450],
            [685,450],
            [250,690],
            [1030,690]
        ]),
        'DST_POINTS': np.float32([
            [300,0],
            [980,0],
            [300,720],
            [980,720]
        ])
    }

    ## PIPE CONSTANTS
    PIPE = {
        'THRESHOLD': {
            'ABS_SOBEL_X': (30, 90),
            'ABS_SOBEL_Y': (60, 170),
            'MAG': (60, 180),
            'DIR': (.5, 1.3),
            'CH': (75, 240),
            'CL': (75, 240),
            'CS': (75, 240)
        }
    }
