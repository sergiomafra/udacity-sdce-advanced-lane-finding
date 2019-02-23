## Udacity - Self-driving Car Engineer Nanodegree
## Period: Term 1
## Class: December 2018
## Project: Advanced Lane Finding
## Author: Sérgio Mafra
## Email: sergio@mafra.io
## GitHub: github.com/sergiomafra/udacity-sdce-advanced-lane-finding

## IMPORTS
import cv2
import matplotlib.image as mpimg
import numpy as np


# CalibrateCamera Class
class CalibrateCamera:

    def __init__(self, camcal_consts):
        self.camcal_consts = camcal_consts

    def calcCalibrationMatrix(self):

        nx = self.camcal_consts['NX']
        ny = self.camcal_consts['NY']
        images = self.camcal_consts['IMAGES']

        img = mpimg.imread(images[0])

        objpoints = []
        imgpoints = []

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for image in images:
            img = mpimg.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[1::-1], None, None
        )

        return (mtx, dist)
