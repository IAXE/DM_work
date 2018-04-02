#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 01:36:01 2018

@author: Eddy Palo
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from time import time
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

""" FUNCION diff_hsv """    

def diff_hsv_v3(ult_hsv,act_hsv):
    A=np.ones((256,1),dtype=np.int8)
    B=np.ones((1,138),dtype=np.int8)
    delta_h, delta_s, delta_l, delta_hsv_prom = 0.0, 0.0, 0.0, 0.0
    delta_hsv = [-1, -1, -1, -1]
    num_pixels = act_hsv[0].shape[0] * act_hsv[0].shape[1]
    for i in range(3):
        act_hsv[i] = act_hsv[i].astype(np.int32)
        ult_hsv[i] = ult_hsv[i].astype(np.int32)
        A_0=np.dot(act_hsv[i],A)
        B_0=np.dot(B,A_0)
        A_1=np.dot(ult_hsv[i],A)
        B_1=np.dot(B,A_1)
        delta_hsv[i]=float(((B_0-B_1)/(num_pixels))**2)
        
    delta_hsv[3]=(math.sqrt(sum(delta_hsv[0:3])))
    delta_hsv[0]=math.sqrt(delta_hsv[0])
    delta_hsv[1]=math.sqrt(delta_hsv[1])
    delta_hsv[2]=math.sqrt(delta_hsv[2])
    delta_h, delta_s, delta_l, delta_hsv_prom = delta_hsv
    return (delta_h, delta_s, delta_l, delta_hsv_prom)

def diff_hsv_v4(frame_hsv,punto):
    A=np.ones((256,1),dtype=np.int8)
    B=np.ones((1,138),dtype=np.int8)
    delta_h, delta_s, delta_l, delta_hsv_prom = 0.0, 0.0, 0.0, 0.0
    delta_hsv = [-1, -1, -1, -1]
    num_pixels = frame_hsv[0].shape[0] * frame_hsv[0].shape[1]
    for i in range(3):
        if i==1:
            j=2
        else:
            j=1
        frame_hsv[i] = frame_hsv[i].astype(np.int32)
        A_0=np.dot(frame_hsv[i],A)
        B_0=np.dot(B,A_0)
        delta_hsv[i]=float(((B_0-punto[i])*j/(num_pixels))**2)
        
    delta_hsv[3]=math.sqrt((sum(delta_hsv[0:3])))
    delta_hsv[0]=math.sqrt(delta_hsv[0])
    delta_hsv[1]=math.sqrt(delta_hsv[1])
    delta_hsv[2]=math.sqrt(delta_hsv[2])
    delta_h, delta_s, delta_l, delta_hsv_prom = delta_hsv
    return (delta_h, delta_s, delta_l, delta_hsv_prom)
    
""" FUNCION KMEANS """
def fun_KM(X,Y,nc):
    Mdata=np.transpose([X, Y])
    Mdata.T[0]=(Mdata.T[0]-Mdata.T[0].mean())/Mdata.T[0].mean()
    Mdata.T[1]=(Mdata.T[1]-Mdata.T[1].mean())/Mdata.T[1].mean()
    Mdata = StandardScaler().fit_transform(Mdata)
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(Mdata)
    centroids = kmeans.cluster_centers_
    t_cent = np.array(centroids)
    t_cent.T[0] = Mdata.T[0].mean() * (1+t_cent.T[0])
    t_cent.T[1] = Mdata.T[1].mean() * (1+t_cent.T[1])
#    labels = kmeans.predict(Mdata)
    plt.scatter(Mdata.T[0], Mdata.T[1], c = kmeans.labels_, cmap='rainbow')
    plt.scatter(centroids.T[0], centroids.T[1], marker='*',c='orange', s=200)
    plt.title('KMEANS - Número de clusters: %d' % nc)
    plt.show()
    return t_cent

""" FUNCION DBSCAN para Tomas """
def fun_DBSCAN_tomas(X,Y):
    Mdata = np.transpose([X, Y])
    Mdata_std = StandardScaler().fit_transform(Mdata)
    
    db = DBSCAN(eps=0.23, min_samples=15).fit(Mdata_std)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = Mdata[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                  markersize=6)

        xy = Mdata[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=2)

    plt.title('DBSCAN - Número de tomas estimados: %d' % n_clusters_)
    plt.xlabel('N° de Frame')
    plt.ylabel('Distancia')
    plt.grid(True)
    plt.show()
    """
    return labels, n_clusters_

""" FUNCION DBSCAN para Escenas """
def fun_DBSCAN_escenas(X,Y):
    Mdata = np.transpose([X, Y])
    Mdata_std = StandardScaler().fit_transform(Mdata)
    
    db = DBSCAN(eps=0.27, min_samples=1).fit(Mdata_std)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = Mdata[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                  markersize=6)

        xy = Mdata[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=2)

    plt.title('Número de Escenas estimados: %d' % n_clusters_)
    plt.xlabel('N° de Frame')
    plt.ylabel('Distancia')
    plt.grid(True)
    plt.show()
    """
    return labels, n_clusters_


""" FUNCION fun_CARACT """
def fun_CARACT(C,Valores):
    val = np.array(Valores)
    datos = np.column_stack((clasi,val))
    df_datos=pd.DataFrame(datos)
    df_datos=pd.DataFrame(datos, columns=('clase','avg_H', 'avg_S', 'avg_L', 'avg_HSV'))
    resu = df_datos.groupby('clase').agg([np.mean, np.std])
    return resu

""" FUNCION fun_extraer_toma_ant """
def fun_extraer_toma_ant(frame_actual, delta_frame):
    frame_ext=frame_actual + 1 - int(delta_frame/2)
    captura.set(1,frame_ext)
    ret3,img3 = captura.read()
    cv2.imwrite('Tomas200/toma_'+str(c_toma)+'.png',img3)
    captura.set(1,frame_actual+1)
    return
    
""" INICIO DE PROGRAMA """
captura = cv2.VideoCapture('Hachiko.mkv')

Total_frames=int(captura.get(7)) #Número total de frames que contiene la pelicula

inicio=873  #indicar el N° de frame de inicio de análisis
i=inicio 
fin=100000    #indicar el N° de frame de fin de análisis
c_toma=0
c_toma_clus=0
c_escena=0
c_toma_esce=0
ult_frame=inicio
ult_toma=0


captura.set(1,i) #nos ubicamos en el frame de inicio de análisis
ret1,img1 = captura.read()
r = 256 / img1.shape[1]
dim = (256, int(img1.shape[0] * r)) #redimensionar los frames a 138x256 pixeles
img10 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
hsv1 = cv2.split(cv2.cvtColor(img10, cv2.COLOR_BGR2HSV))
i +=1

ini_time=time()
Tomas=[]
Imagenes=[]     # registro de frames de inicio de tomas
Tiempos_ini=[]  # registro de tiempos de inio de tomas
Deltas_prom=[]
Valores_diffhue=[]
Valores_diffsat=[]
Valores_difflumi=[]
ACC_deltas=[]
dist_PROM=[]
Distancias=[]
FRAMES=[]
Distancias_esce=[]
dist_PROM_esce=[]
FRAMES_esce=[]
CENTROIDES=[]
CARACTERISTICAS=np.array([-1,-1,-1,-1,-1,-1,-1,-1])

while(True):
    
    ret2,img2 = captura.read()
    
    img20 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    hsv2=cv2.split(cv2.cvtColor(img20, cv2.COLOR_BGR2HSV))
     
    delta_V, delta_S, delta_L, delta_PROM = diff_hsv_v3(hsv1,hsv2)

    ACC_deltas.append(delta_PROM)
    diff_frame = i - ult_frame
    
    if  ((delta_PROM >= 7.99 and diff_frame>=58)or(diff_frame<=30 and delta_PROM>100)or(diff_frame>30 and diff_frame<58 and delta_PROM>15.24)):
       
        c_toma +=1
        #fun_extraer_toma_ant(i, diff_frame)
    
        
        if c_toma==1:
            Distancias=[]
            FRAMES=[]
            dist_PROM=[]
            Distancias_esce=[]
            dist_PROM_esce=[]
            FRAMES_esce=[]
            
        t_ini_toma = captura.get(0)/1000
        print ('++++Nueva toma++++')
        print ('N° de toma:',c_toma)
        print ('Frame de inicio:',i)
        print ('Tiempo de inicio:',t_ini_toma,'segundos' )
        print ('Diferencia_v:', delta_V)
        print ('Diferencia_sat:', delta_S)
        print ('Diferencia_lumi:', delta_L)
        print ('Diferencia_prom:', delta_PROM)
        
#        cv2.imwrite('Tomas100/toma_'+str(c_toma)+'.jpg',img20)
        Tomas.append(c_toma)
        Imagenes.append(i)
        Tiempos_ini.append(t_ini_toma)
        Deltas_prom.append(delta_PROM)
        Valores_diffhue.append(delta_V)
        Valores_diffsat.append(delta_S)
        Valores_difflumi.append(delta_L)
        ult_frame=i
        
        if ((c_toma - ult_toma) == 5) or i == fin:
            ult_toma = c_toma
            c_toma_esce +=1
            clasi, clusters = fun_DBSCAN_tomas(FRAMES,dist_PROM)
            #fun_KM(FRAMES,dist_PROM,clusters)    #solo para comparar con método KMEANS
            carac = fun_CARACT(clasi,Distancias)
            c_toma_clus +=clusters
            CARACTERISTICAS=np.vstack((CARACTERISTICAS,carac))
            dist_PROM=[]
            Distancias=[]
            FRAMES=[]
        if c_toma_esce == 6:
            c_toma_esce=0
            clasi_esce, clu_esce = fun_DBSCAN_escenas(FRAMES_esce,dist_PROM_esce)
            #fun_KM(FRAMES_esce,dist_PROM_esce,clu_esce)
            c_escena +=clu_esce
            Distancias_esce=[]
            dist_PROM_esce=[]
            FRAMES_esce=[]
            

    dist_temp = diff_hsv_v4(hsv2,[0,0,0])
    Distancias.append(dist_temp)
    dist_PROM.append(dist_temp[3])
    FRAMES.append(i)
    Distancias_esce.append(dist_temp)
    dist_PROM_esce.append(dist_temp[3])
    FRAMES_esce.append(i)
    i +=1
    hsv1 = hsv2
    if i >= fin :
        clasi, clusters = fun_DBSCAN_tomas(FRAMES,dist_PROM)
        carac = fun_CARACT(clasi,Distancias)
        c_toma_clus +=clusters
        CARACTERISTICAS=np.vstack((CARACTERISTICAS,carac))
        clasi_esce, clu_esce = fun_DBSCAN_escenas(FRAMES_esce,dist_PROM_esce)
        c_escena +=clu_esce
        break
    
fin_time=time()
total_time=fin_time - ini_time
""" Generar archivo csv """
data1=np.transpose([Tomas, Imagenes, Tiempos_ini, Valores_diffhue, Valores_diffsat, Valores_difflumi,Deltas_prom])
df_tomas = pd.DataFrame(data1, columns=('N° Toma', 'Frame de inicio', 'Tiempo de inicio(seg)', 'Delta_hue', 'Delta_sat', 'Delta_lumi', 'Delta_prom'))
df_tomas.to_csv("Resumen_tomas.csv")
df_base_escenas=pd.DataFrame(CARACTERISTICAS, columns=('H_mean','H_std','S_mean','S_std','V_mean','V_std','HSV_mean','HSV_std'))
df_base_escenas.to_csv("Data _para_escenas.csv")
""" Impresión de resultados """
print('')
print('+++++++ RESUMEN DE RESULTADOS +++++++')
print ('Total de Frames de la pelicula:',Total_frames)
print ('Analizado desde el frame N°:', inicio,'al frame N°:',fin)
print ('Número de tomas encontradas:', c_toma)
print ('Número de tomas encontradas utilizando clustering:', c_toma_clus)
print ('Número de escenas encontradas:', c_escena)
print('Tiempo de ejecución:',total_time,'segundos')


