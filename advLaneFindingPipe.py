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

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

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


