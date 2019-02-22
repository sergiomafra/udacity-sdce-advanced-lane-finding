from classes.CalibrateCamera import CalibrateCamera
from classes.Constants import Constants

consts = Constants()

## CALIBRATE CAMERA
camcal = CalibrateCamera(consts.CAMCAL)
mtx, dist = camcal.calcCalibrationMatrix()

## PIPE
pipe = Pipe(mtx, dist)
pipe.run()
