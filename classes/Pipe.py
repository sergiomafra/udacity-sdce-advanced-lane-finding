## Udacity - Self-driving Car Engineer Nanodegree
## Period: Term 1
## Class: December 2018
## Project: Advanced Lane Finding
## Autor: Sérgio Mafra
## Email: sergio@mafra.io
## GitHub: github.com/sergiomafra/udacity-sdce-advanced-lane-finding

## IMPORTS
import cv2
import matplotlib.image as mpimg
import numpy as np
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

class Pipe:

    def __init__(self, pipe_consts, pers_consts, mtx, dist):
        self.pers_consts = pers_consts
        self.pipe_consts = pipe_consts
        self.mtx = mtx
        self.dist = dist

    def _absSobelThresh(self, image, orientation, thresh):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if orientation == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        elif orientation == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobel = np.absolute(sobel)

        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return sbinary

    def _magThresh(self, image, mag_thresh):

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

    def _dirThresh(self, image, sobel_kernel, thresh):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        grad_dir = np.arctan2(abs_sobely, abs_sobelx)

        dbinary = np.zeros_like(grad_dir)
        dbinary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

        return dbinary

    def _hlsChannel(self, image, channel, threshold=(75,240)):

        channel = channel.lower()

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        if channel == 'h':
            channel = hls[:,:,0]
        elif channel == 'l':
            channel = hls[:,:,1]
        else:
            channel = hls[:,:,2]

        hlsbinary = np.zeros_like(channel)
        hlsbinary[(channel > threshold[0]) & (channel <= threshold[1])] = 1

        return hlsbinary

    def applyThreshold(self, image, filter_type, threshold, orientation='x',
        channel='s', ksize=3):

        if filter_type == 'abs_sobel':
            return self._absSobelThresh(image, orientation, threshold)
        elif filter_type == 'mag':
            return self._magThresh(image, threshold)
        elif filter_type == 'dir':
            return self._dirThresh(image, kernel, threshold)
        elif filter_type == 'hls':
            return self._hlsChannel(image, channel, threshold)

        return 'Something went wrong with filter type: {}'.format(filter_type)

    def run(self):
        ## SINGLE IMAGE PIPE

        ## STEP 1: Undistort Image Applying MTX and DIST
        img = mpimg.imread('media/test_images/test6.jpg')
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        ## STEP 2: Apply Threshold to an Image to Identify the Lane Lines
        ## The best combination that I found was between
        ## MAG, GRADX, GRADY, Channel S from HLS. All summed.
        gradx = self.applyThreshold(
                undistorted,
                'abs_sobel',
                self.pipe_consts['THRESHOLD']['ABS_SOBEL_X'],
                orientation = 'x')
        grady = self.applyThreshold(
                undistorted,
                'abs_sobel',
                self.pipe_consts['THRESHOLD']['ABS_SOBEL_Y'],
                orientation = 'y')
        mag_bin = self.applyThreshold(
                undistorted,
                'mag',
                self.pipe_consts['THRESHOLD']['MAG'])
        # dir_bin = self.applyThreshold(
        #         undistorted,
        #         'dir',
        #         self.pipe_consts['THRESHOLD']['DIR'],
        #         kernel = 9)
        # hls_ch = self.applyThreshold(
        #         undistorted,
        #         'hls',
        #         self.pipe_consts['THRESHOLD']['CH'],
        #         channel = 'h')
        # hls_cl = self.applyThreshold(
        #         undistorted,
        #         'hls',
        #         self.pipe_consts['THRESHOLD']['CL'],
        #         channel = 'l')
        hls_cs = self.applyThreshold(
                undistorted,
                'hls',
                self.pipe_consts['THRESHOLD']['CS'],
                channel = 's')

        ## Uncomment to save images locally and see progress
        # mpimg.imsave('gradx.jpg', gradx, cmap='gray')
        # mpimg.imsave('grady.jpg', grady, cmap='gray')
        # mpimg.imsave('mag_bin.jpg', mag_bin, cmap='gray')
        # mpimg.imsave('hls_cs.jpg', hls_cs, cmap='gray')

        ## Combination formula
        combined = np.zeros_like(undistorted[:,:,0])
        combined[(gradx == 1)|(grady == 1)|(mag_bin == 1)|(hls_cs == 1)] = 1

        ## Uncomment to save combined locally
        # mpimg.imsave('combined.jpg', combined, cmap='gray')

        ## STEP 3: Perspective Transform

