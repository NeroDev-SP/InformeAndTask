import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('tridente.jpg')

# Comprobar si se cargó correctamente la imagen
if imagen is None:
    print("Error: No se pudo cargar la imagen")
    exit()

# Convertir a escala de grises
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Cargar los clasificadores en cascada
rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ojo_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detección de rostros
rostros = rostro_cascade.detectMultiScale(gray, 1.1, 4)

# Para cada rostro detectado
for i, (x, y, w, h) in enumerate(rostros):
    print(f"rostro{i + 1}; x={x}, y={y}, w={w}, h={h}")
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = imagen[y:y+h, x:x+w]
    
    ojos = ojo_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in ojos:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
#recortar la imagen
recortar_imagen = gray[y:y+h, x:x+w]
recortar_color = imagen[y:y+h, x:x+w]
#detectar los ojos dentro del rostro
ojos = ojo_cascade.detectMultiScale(gray, 1.1, 4)
#dibujar rectangulos alrededor de los ojos
for (ox, oy, ow, oh) in ojos:
    cv2.rectangle(recortar_color, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 2)
    print(f"ojos{i + 1}; x={ox}, y={oy}, w={ow}, h={oh}")

# Mostrar la imagen resultante
cv2.imshow('Detección de rostros y ojos', imagen)
cv2.waitKey(0)  # Espera a que se presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas
#Derechos de autor NeroDev