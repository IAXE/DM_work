#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:11:46 2018

@author: eddy
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('fig_1160.jpg',0)          # queryImage
img2 = cv2.imread('fig_1170.jpg',0) # trainImage
 
# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
#print matches

 
# Apply ratio test
good = []
count = 0
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        count +=1
print count 
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img2.copy(),flags=2)
 
plt.imshow(img3),plt.show()