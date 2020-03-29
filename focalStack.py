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
            shutil.rmtree('./Output/*')


    def stack(self):
        self.align()


