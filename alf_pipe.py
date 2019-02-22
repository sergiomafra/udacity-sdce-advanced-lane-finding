from classes.CalibrateCamera import CalibrateCamera
from classes.Constants import Constants
from classes.Pipe import Pipe

consts = Constants()

## CALIBRATE CAMERA
camcal = CalibrateCamera(consts.CAMCAL)
mtx, dist = camcal.calcCalibrationMatrix()

## PIPE
pipe = Pipe(consts.PIPE, consts.PERSPECTIVE, mtx, dist)
pipe.run()
