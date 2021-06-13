'''
@Time          : 
@Author        : yongjae lee, jisoo lee
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import numpy as np
import cv2
import math
import time


class ParamFindHelper:

    def __init__(self, frame):
        
        self.width = 0
        self.height = 0
        self.roi_upper_y = 0
        self.roi_lower_y = 0
        self.guide_rect_y = 0 
        self.candidate_lines = []
        self.lmr = {}
        self.target_point = (0, 0)
        self.bgr_frame = frame


    # added function
    # def edgeDetect(self, img_dir):
    def edgeDetect(self):
        # self.bgr_frame = cv2.imread(img_dir) #read image
        img = self.bgr_frame

        self.width = img.shape[1]
        self.height = img.shape[0]
        self.target_point = (int(self.width/2), 0)

        self.roi_upper_y = int(self.height*(2/3))
        self.roi_lower_y = self.height
        self.guide_rect_y = self.height 


        self.processGrayscale()
        self.processCanny()
        # set ROI
        self.canny_frame[:self.roi_upper_y, :] = 0
        self.canny_frame[self.roi_lower_y:, :] = 0

        self.processHough(20, 10)

        lines = self.candidate_lines
        # classify which area the line belongs to: left or middle or right
        # select representative line for each area
        lmr = self.findLmr(lines)

        self.lmr = lmr
        self.drawAuxiliaries()

        # print(self.bgr_frame.shape)
        # cv2.imshow("main", self.bgr_frame)
        # cv2.imwrite('sample_res.png', self.bgr_frame)
        return self.bgr_frame


    def processGrayscale(self):
        self.grayscale_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('gray.png', self.grayscale_frame)


    def processCanny(self):
        # blurred = cv2.GaussianBlur(self.grayscale_frame, (3, 3), 0)
        self.canny_frame = self.autoCanny(self.grayscale_frame, sigma=0.01)
        cv2.imwrite('canny7.png', self.canny_frame) 
 

    def processHough(self, minLineLength=100, maxLineGap=10):
        lines = cv2.HoughLinesP(self.canny_frame, 1, np.pi / 180, 50, 0, minLineLength, maxLineGap)

        if lines is not None:
            # transform as numpy array
            lines = np.array(lines)
            # reshape array: (len, 1, 4) -> (len, 4)
            lines = lines[:,0]
            # calculate slope
            lines = map(self.determineSlope, lines)
            # filter out None elements that were x1==x2
            self.candidate_lines = [x for x in lines if x is not None]
        else:
            self.candidate_lines = []



    def drawAuxiliaries(self):
        # draw line candidates
        # for line in self.candidate_lines:
        #     x1, y1, x2, y2, a = line
        #     cv2.line(self.bgr_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw lmr lines and rectangles
        # for line in self.lmr.values():
        #     x1, y1, x2, y2, a = line
        #     scale = 1000
        #     guide_rect_x = int((self.guide_rect_y - y1 + a * x1) / a)
        #     cv2.line(self.bgr_frame, (int(x1-scale), int(y1-scale*a)), (int(x2+scale), int(y2+scale*a)), (255, 0, 0), 2)
        for line in self.lmr.values():
            x1, y1, x2, y2, a = line
            scale = 1000
            # limitwidth = self.width / 2
            ylimit = self.height * (2/3)
            slope = (y2-y1) / (x2-x1)
            xlimit = (ylimit-y1)/slope + x1

            guide_rect_x = int((self.guide_rect_y - y1 + a * x1) / a)
            print(x1, y1, x2, y2, slope)
            if slope<0:
                cv2.line(self.bgr_frame, (int(x1-scale), int(y1-scale*a)), (int(xlimit), int(ylimit)), (0, 255, 0), 2)
            else:
                cv2.line(self.bgr_frame, (int(xlimit), int(ylimit)), (int(x2+scale), int(y2+scale*a)), (0, 255, 0), 2)

        
        # print fps
        # now_time = time.time()
        # fps = 1 / (now_time - self.last_time)
        # while fps > self.target_fps:
        #     now_time = time.time()
        #     fps = 1 / (now_time - self.last_time)
        # self.last_time = now_time
        # cv2.putText(self.bgr_frame, "fps: {:.5f}".format(fps), (0, self.height - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))



    def autoCanny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged


    def determineSlope(self, line):
        x1, y1, x2, y2 = line

        # ignore vertical line
        if x1 == x2:
            return None

        a = (y2 - y1) / float(x2 - x1)

        # ignore line of which slope is over or under threshold
        th = float(3)
        if abs(a) > th or abs(a) < 1/th:
            return None

        return (x1, y1, x2, y2, a)

    def findLmr(self, lines):
        lmr = {}

        for line in lines:
            x1, y1, x2, y2, a = line

            # find left line
            if a < 0 and max(x1, x2) < self.width * 5 / 10:
                left = lmr.get('left', (0, 0, 0, 0, 0))
                left_y = max(left[1], left[3])
                line_y = max(y1, y2)
                lmr['left'] = line if line_y > left_y else left

            # find right line
            elif a > 0 and min(x1, x2) > self.width * 5/10:
                right = lmr.get('right', (0, 0, 0, 0, 0))
                right_y = max(right[1], right[3])
                line_y = max(y1, y2)
                lmr['right'] = line if line_y > right_y else right


        return lmr

    def getCrossPoint(self, l, r):
        x = ((l[1] - r[1]) + (r[4]*r[0] - l[4]*l[0])) / (r[4] - l[4])
        y = l[4] * x + l[1] - l[4] * l[0]
        return int(x), int(y)




if __name__ == '__main__':

    img_dir = 'bdd_sample.jpg'

    obj = ParamFindHelper()

    obj.edgeDetect(img_dir)





    