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

    def _find_lane_pixels(self, binary_warped):
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
        nwindows = 13
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
            cv2.rectangle(
                out_img,
                (win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),
                (0,255,0),
                2
            )
            cv2.rectangle(
                out_img,
                (win_xright_low,win_y_low),
                (win_xright_high,win_y_high),
                (0,255,0),
                2
            )

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &
                (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &
                (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on
            # their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of
        # lists of pixels)
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

    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = \
            self._find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + \
                right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still
            # none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')

        return out_img

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

        ## STEP 3: Perspective Transform to a Bird View
        M = cv2.getPerspectiveTransform( # Perspective Matrix
            self.pers_consts['SRC_POINTS'],
            self.pers_consts['DST_POINTS']
        )
        birdview = cv2.warpPerspective(
            combined,
            M,
            (combined.shape[1], combined.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        ## STEP 4: Apply Sliding Windows to Find Lane Lines and
        ## Fit a Polynomial
        out_img = self.fit_polynomial(birdview)

        ## Uncomment to save the out_img
        # mpimg.imsave('out_img.jpg', out_img)
