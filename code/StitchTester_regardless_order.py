import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from cyvlfeat.sift import sift
from SIFTSimpleMatcher import SIFTSimpleMatcher
from RANSACFit import RANSACFit
from statistics import median
from PairStitch import PairStitch
import random

'''
# This script takes from a specific folder a bunch of images (files) which are used to create a "pano" scene
# stitch them each other. The sequence of each image is no relevant for this implementation 
# Most of the method implemented here are wraps or derived form the original script StitchTester.py
'''
def swapcolumn(arr):
    # sweep sift index of [Y,X] -> [X,Y] for easy computation of affine transform
    swap = []
    swap.append(arr[:,1])
    swap.append(arr[:,0])
    swap = np.asarray(swap).T
    return swap

def sift_test(img):
    img = np.asarray(img.convert('L')).astype('single')
    f, descriptor = sift(img, compute_descriptor=True, float_descriptors=True)
    return swapcolumn(f[:,0:2]), descriptor

#%% Parameters
Thre = 0.5
RESIZE = 0.5

#%% Load a list of images (Change file name if you want to use other images)
imgList = glob('../data/Rainier*.png')
saveFileName = '../results/Rainier_regardless_order.png'

imgList = glob('../data/yosemite*.jpg')
saveFileName = '../results/yosemite_regardless_order.jpg'
#%% Add path
Images = {}
for idx, imgPath in enumerate(sorted(imgList)):
    print(idx, imgPath)
    fileName = os.path.basename(imgPath)
    img = Image.open(imgPath, 'r')
    if (max(img.size)>1000 or len(imgList)>10):
        img.thumbnail((np.asarray(img.size)*RESIZE).astype('int'), Image.ANTIALIAS)
    Images.update({idx: img})
print('Images loaded. Beginning feature detection...')
print(type(Images))

ref_idx, idx, max_img = 1, 0, len(Images)
print("The image {} chosen as reference".format(ref_idx))
pano = np.asarray(Images[ref_idx])
pt_ref, descri_ref = sift_test(Images[ref_idx])

while True:
    if len(Images) == idx: break
    if idx == ref_idx: 
        idx+=1
        continue
    img = Images[idx]    
    pts, descr = sift_test(img)
    M = SIFTSimpleMatcher(descr, descri_ref, Thre)
    if len(M)<3:
        Images.update({max_img: img})
        continue
    H = RANSACFit(pts, pt_ref, M)
    pano = PairStitch(Image.fromarray(np.asarray(img)), Image.fromarray(pano), 
                      H, save=False, get_array=True)
    pt_ref, descri_ref = sift_test(pano) 
    pano = np.asarray(pano)
    idx+=1

result = Image.fromarray(pano)
result.save(saveFileName)
print('The completed file has been saved as '+saveFileName)