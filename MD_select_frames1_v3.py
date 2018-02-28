
"""
Created on Sun Feb 11 03:36:11 2018

@author:EPV
"""
import numpy as np
import cv2

captura = cv2.VideoCapture('Hachiko.mkv')
#print(cap.isOpened())
Total_frames = int(captura.get(7)) #Numero total de frames que contiene la pelicula
print (Total_frames)
print (captura.set(3,640.0))
print (captura.get(3))
captura.set(4,345.0)
print (captura.get(4))
for i in range(1000,Total_frames+10,10):
    print(captura.set(1,i))    #selecciono el frame con valor i
    print (captura.get(1))
    print (captura.get(3))
    print (captura.get(4))
    ret, frame = captura.read()
    cv2.imwrite('Figuras_v3/fig_'+str(i)+'.jpg',frame)
captura.release()
cv2.destroyAllWindows()


