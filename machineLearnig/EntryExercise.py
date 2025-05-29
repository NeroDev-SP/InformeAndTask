import cv2
import numpy as np

imagen = cv2.imread('neymar4.jpg')
#comprueba si se cargo correctamente la imagen
if imagen is None:
    print("Error: No se pudo cargar la imagen")
    exit()

gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Detección de rostros
rostros = rostro_cascade.detectMultiScale(gray, 1.1, 4)
#dibujo de rectangulos o cuadrados
for (x, y, w, h) in rostros:
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow('Rostro detectado', imagen)
cv2.waitKey(0)#espera a que se presione una tecla
cv2.destroyAllWindows()#cierra todas las ventanas