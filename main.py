from focalStack import focalStack
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--inputDir', default='./Input/', type=str)
parser.add_argument('--detectorMethod', default='ORB', type=str)

opt = parser.parse_args()
if not opt.inputDir.endswith('/'):
    opt.inputDir += '/'

if __name__ == "__main__":
    focalStack = focalStack(inputDir=opt.inputDir, detectorMethod=opt.detectorMethod)
    focalStack.stack()