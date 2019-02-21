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

SRC_POINTS = np.float32([
    [540,460],
    [740,460],
    [-100,720],
    [1380,720]
])
DST_POINTS = np.float32([
    [0,0],
    [1280,0],
    [0,720],
    [1280,720]
])

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

def undistortImage(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

def abs_sobel_thresh(image, orient='x', thresh=(0, 255)):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary

def mag_thresh(image, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobelxy = (sobelx**2 + sobely**2)**(1/2)

    scaled_sobel = np.uint8(abs_sobelxy*255/np.max(abs_sobelxy))

    mbinary = np.zeros_like(scaled_sobel)
    mbinary[(scaled_sobel >= mag_thresh[0]) & \
            (scaled_sobel <= mag_thresh[1])]  \
            = 1

    return mbinary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    dbinary = np.zeros_like(grad_dir)
    dbinary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return dbinary

def hlsChannelToBinary(image, channel, threshold=(75,240)):

    channel = channel.lower()

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    if channel == 'h':
        channel = hls[:,:,0]
    elif channel == 'l':
        channel = hls[:,:,1]
    else:
        channel = hls[:,:,2]

    hlsbinary = np.zeros_like(sat)
    hlsbinary[(sat > threshold[0]) & (sat <= threshold[1])] = 1

    return hlsbinary

def genPerspectiveMatrix(src_points, dst_points):

    return cv2.getPerspectiveTransform(src_points, dst_points)

def applyThreshold(image):
    pass

def pipe():
    pass

## CAMERA CALIBRATION
mtx, dist = calibrateCamera(IMAGES)

# # Calibration Test
# img = mpimg.imread('camera_cal/calibration1.jpg')
# undistorted = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(undistorted)


