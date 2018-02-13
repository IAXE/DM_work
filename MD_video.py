#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 03:36:11 2018

@author: eddy
"""
import numpy as np
import cv2

cap = cv2.VideoCapture('Hachiko.mkv')
print(cap.isOpened())

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('Pelicula_Hachiko',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()