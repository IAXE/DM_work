
import numpy as np
import cv2

cap = cv2.VideoCapture('Pelicula.mkv')
print(cap.isOpened())

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('Pelicula',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
