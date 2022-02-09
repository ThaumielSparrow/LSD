'''
Module to locate contours to find multiple digits in a single image. Saves images to CWD as "roi[iter].png".

Author: Luzhou Zhang
'''
from types import ModuleType
import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt

class DigitsByContour():
    '''
    Contains structure to make digit detection scalable for the future

    Functions: digits_by_contour
    '''
    def __init__(self):
        pass

    def digits_by_countour(self, img, save=True, passOrig=False):
        '''
        Finds digits in an image by contours.

        Arguments: image to be analyzed, whether to save images physically, whether to return original image with bounds around digits

        Returns: NumPy array containing all digits
        '''
        contours = []
        img_main = cv.imread(img)
        img_blur = cv.GaussianBlur(img_main, (7, 7), 1)
        img_blur_RGB = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)

        ret, thresh1 = cv.threshold(img_blur_RGB, 127, 256, cv.THRESH_BINARY_INV)
        dilate = cv.dilate(thresh1, None, iterations=2)

        cnts = cv.findContours(dilate.copy(), cv.RETR_TREE, cv.cv2.CHAIN_APPROX_SIMPLE)
        # Case for first element when OpenCV is v4+
        cnts = cnts[0] # if imutils.is_cv2() else cnts[1]

        orig = img_main.copy()
        i = 0

        for cnt in cnts:
            # Check the area of contour, may need to refine this
            if(cv.contourArea(cnt) < 100):
                continue
            # Detect filtered contours
            x,y,w,h = cv.boundingRect(cnt)
            # Get region of interest
            roi = img_main[y:y+h, x:x+w]
            # Mark these on the original image
            cv.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
            # Save contours
            
            if save:
                cv.imwrite('roi' + str(i) + '.png', roi)
            i += 1
            contours.append(roi)
        contours_arr = np.asarray(contours)
        if passOrig:
            return contours_arr, orig
        else:
            return contours_arr

# Testing Block
def main():
    get_contours = DigitsByContour()
    contours, orig_marked = get_contours.digits_by_countour('clocknum.png', save=False, passOrig=True)
    # cv.imshow('Image', orig)
    # cv.waitKey(0)
    plt.imshow(orig_marked)
    plt.show()
    # for i in range(1, contours.size):
    #     plt.imshow(contours[i-1])
    #     plt.show()

if __name__ == '__main__':
    main()