import os
import shutil
import numpy as np
import cv2

class focalStack:

    def __init__(self, inputDir = './Input/', detectorMethod = 'ORB'):
        self.inputDir = inputDir
        self.detectorMethod = detectorMethod
        self.images = [cv2.imread(self.inputDir + imName) for imName in sorted(os.listdir(self.inputDir)) 
            if imName.split('.')[-1] in ["jpg", "jpeg", "png"] ]
        self.alignedImages = []
        self.mergedImage = None
        if not os.path.exists('./Output/'):
            os.mkdir('./Output/')
        else:
            shutil.rmtree('./Output/')
            os.mkdir('./Output/')


    def stack(self):
        self.align()


    def align(self):

        # correspondence(match) + homography
        # ref:
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # https://www.learnopencv.com/homography-examples-using-opencv-python-c/

        if self.detectorMethod == 'SIFT':
            detector = cv2.xfeatures2d.SIFT_create()
        elif self.detectorMethod == 'ORB':
            detector = cv2.ORB_create(1000)

        self.alignedImages.append(self.images[0])
        baseIm = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY)
        baseImKp, baseImDes = detector.detectAndCompute(baseIm, None)
        for i in range(1, len(self.images)):
            imKp, imDes = detector.detectAndCompute(self.images[i], None)


            if self.detectorMethod == 'SIFT':
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(imDes, baseImDes, k = 2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
            elif self.detectorMethod == 'ORB':
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                good = bf.match(imDes, baseImDes)

            # homography
            # using the most condident matches for homography
            good = sorted(good, key = lambda x: x.distance)[:128]
            
            baseImPnt = np.array([list(baseImKp[match.trainIdx].pt) for match in good])
            imPnt = np.array([list(imKp[match.queryIdx].pt) for match in good])

            homography, mask = cv2.findHomography(imPnt, baseImPnt, cv2.RANSAC, ransacReprojThreshold = 2.0)
            alignedIm = cv2.warpPerspective(self.images[i], homography, (self.images[i].shape[1], self.images[i].shape[0]), flags=cv2.INTER_LINEAR)
            self.alignedImages.append(alignedIm)
        
        if not os.path.exists('./Output/Aligned'):
            os.mkdir('./Output/Aligned/')
        for i in range(len(self.alignedImages)):
            cv2.imwrite('./Output/Aligned/aligned{}.png'.format(i), self.alignedImages[i])

        self.alignedImages = np.asarray(self.alignedImages)

        return
        
