#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 01:36:01 2018

@author: eddy
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from time import time

def compara_img(hsv1,hsv2):
    hue1, sat1, val1 = hsv1[:,:,0], hsv1[:,:,1], hsv1[:,:,2]
    hue1_1=np.histogram(hue1.flatten(), bins=180)[0]
    sat1_1=np.histogram(sat1.flatten(), bins=128)[0]
    val1_1=np.histogram(val1.flatten(), bins=128)[0]

    hue2, sat2, val2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]
    hue2_1=np.histogram(hue2.flatten(), bins=180)[0]
    sat2_1=np.histogram(sat2.flatten(), bins=128)[0]
    val2_1=np.histogram(val2.flatten(), bins=128)[0]

    huediff=np.sqrt((hue2_1 - hue1_1)**2)
    valor=math.log10(np.sum(huediff))
    
    satdiff=np.sqrt((sat2_1 - sat1_1)**2)
    valor_sat=math.log10(np.sum(satdiff))
    
    lumidiff=np.sqrt((val2_1 - val1_1)**2)
    valor_lumi=math.log10(np.sum(lumidiff))
    
#    print(valor)
    return (valor, valor_sat, valor_lumi)
#plt.plot(huediff)
    
def diff_hsv(last_hsv,curr_hsv):
    delta_hsv_avg, delta_h, delta_s, delta_l = 0.0, 0.0, 0.0, 0.0
    delta_hsv = [-1, -1, -1]
    for i in range(3):
        num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
        curr_hsv[i] = curr_hsv[i].astype(np.int32)
        last_hsv[i] = last_hsv[i].astype(np.int32)
        delta_hsv[i] = np.sum(np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)

    delta_hsv.append(sum(delta_hsv) / 3.0)
#    print(delta_hsv)
    delta_h, delta_s, delta_l, delta_hsv_avg = delta_hsv
    return (delta_h, delta_s, delta_l, delta_hsv_avg)

captura = cv2.VideoCapture('Hachiko.mkv')

Total_frames=int(captura.get(7)) #Número total de frames que contiene la pelicula
print (Total_frames)
inicio=50000
i=inicio #Para que empiece del frame 877
fin=Total_frames
captura.set(1,i)
ctoma=500
ret1,img1 = captura.read()
hsv1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))
i +=1
ult_frame=-100
captura.set(1,i)
ini_time=time()
Tomas=[]
Imagenes=[]
Valores_diffhue=[]
Valores_diffsat=[]
Valores_difflumi=[]
ACC_deltas=[]

while(True):
    
    ret2,img2 = captura.read()
    hsv2=cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))
    
    i +=1
#    print ('--imagen:--',i)
    delta_V, delta_S, delta_L, delta_AVG = diff_hsv(hsv1,hsv2)

    ACC_deltas.append(delta_AVG)
    diff_frame=i-ult_frame
    if  (delta_AVG >= 35)and((diff_frame>=58)or(diff_frame<=30 and delta_AVG>58)or(diff_frame>30 and diff_frame<58 and delta_AVG>41.18)):
        print ('++++Nueva toma++++')
        print (i-1)
        print ('diferencia_v:', delta_V)
        print ('diferencia_sat:', delta_S)
        print ('diferencia_lumi:', delta_L)
        print ('diferencia_prom:', delta_AVG)
        ctoma +=1
        print('N° de toma:',ctoma)
        cv2.imwrite('Tomas55/toma_'+str(ctoma)+'.jpg',img2)
        Tomas.append(ctoma)
        Imagenes.append(i-1)
        Valores_diffhue.append(delta_V)
        Valores_diffsat.append(delta_S)
        Valores_difflumi.append(delta_L)
        ult_frame=i
        
    hsv1 = hsv2
#    img1 = img2
    
    if i >= fin :
        break
    
fin_time=time()
total_time=fin_time - ini_time
print('Tiempo de ejecución:',total_time)

