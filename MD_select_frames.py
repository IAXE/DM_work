
import numpy as np
import cv2

captura = cv2.VideoCapture('Hachiko.mkv')
#print(cap.isOpened())
Total_frames=int(captura.get(7)) #Número total de frames que contiene la pelicula
print Total_frames

for i in range(0,Total_frames+1,10):
    captura.set(1,i)    #selecciono el frame con valor i
    print captura.get(1)
    ret, frame = captura.read()
    cv2.imwrite('Figuras/fig_'+str(i)+'.jpg',frame)
captura.release()
cv2.destroyAllWindows()


