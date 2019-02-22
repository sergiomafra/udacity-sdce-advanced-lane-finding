import glob
import os

class Constants:

    BASE_PATH = os.getcwd() + '/'
    IMAGES = glob.glob(BASE_PATH + 'media/camera_cal/calibration*.jpg')
    ## CAMERA CAL CONSTS
    CAMCAL = {
        'IMAGES': IMAGES,
        'NX': 9,
        'NY': 6
    }
