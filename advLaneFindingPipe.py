## Udacity - Self-driving Car Engineer Nanodegree
## Period: Term 1
## Class: December 2018
## Project: Advanced Lane Finding
## Autor: Sérgio Mafra
## Email: sergio@mafra.io
## GitHub: github.com/sergiomafra/udacity-sdce-advanced-lane-finding

## IMPORTS
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

# %matplotlib inline

## CONSTANTS
IMAGES_DIR = 'camera_cal'
IMAGES = glob.glob('{}/calibration*.jpg'.format(IMAGES_DIR))
NX = 9
NY = 6

## DEFINITIONS
def calibrateCamera(images):
    objpoints = []
    imgpoints = []

    objp = np.zeros((NX*NY, 3), np.float32)
    objp[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)

    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            img = cv2.drawChessboardCorners(img, (NX,NY), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1::-1], None, None)

    return (mtx, dist)

## CAMERA CALIBRATION
mtx, dist = calibrateCamera(IMAGES)

# # Calibration Test
# img = mpimg.imread('camera_cal/calibration1.jpg')
# undistorted = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(undistorted)
