## Udacity - Self-driving Car Engineer Nanodegree
## Period: Term 1
## Class: December 2018
## Project: Advanced Lane Finding
## Author: Sérgio Mafra
## Email: sergio@mafra.io
## GitHub: github.com/sergiomafra/udacity-sdce-advanced-lane-finding

#IMPORTS
import json
import numpy as np
import os
from classes.CalibrateCamera import CalibrateCamera
from classes.Constants import Constants
from classes.Pipe import Pipe


consts = Constants()

## CALIBRATE CAMERA
cc_file_path = 'state/camcal.json'

if os.path.isfile(cc_file_path):
    with open(cc_file_path, 'r') as cc_file:
        cc_data = json.load(cc_file, )
else:
    camcal = CalibrateCamera(consts.CAMCAL)
    mtx, dist = camcal.calcCalibrationMatrix()

    ## Saving mtx and dist to a .json
    cc_data = {}
    cc_data['mtx'] = mtx.tolist()
    cc_data['dist'] = dist.tolist()

    with open(cc_file_path, 'w') as cc_file:
        cc_file.write(json.dumps(
            cc_data,
            ensure_ascii=False,
            sort_keys=True,
            indent=4)
        )

## PIPE
pipe = Pipe(
    consts.PIPE,
    consts.PERSPECTIVE,
    np.array(cc_data['mtx']),
    np.array(cc_data['dist'])
)
pipe.run()
