import cv2
import os

# Cargar la imagen desde la carpeta 'public/image'
imagen = cv2.imread('public/imageTest/neymar4.jpg')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen...")
    exit()

# Reducción de ruido y detalle
imagen_color = cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)

# Convertir a escala de grises y aplicar mediana
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris = cv2.medianBlur(imagen_gris, 7)

# Detectar bordes con umbral adaptativo
bordes = cv2.adaptiveThreshold(imagen_gris, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 9, 10)

# Combinar bordes y color
imagen_comic = cv2.bitwise_and(imagen_color, imagen_color, mask=bordes)

# Agregar texto a la imagen
cv2.putText(imagen_comic, 'Filtro Comic', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Mostrar imágenes
cv2.imshow('Original', imagen)
cv2.imshow('Comic', imagen_comic)

# Verificar y crear carpeta si no existe
os.makedirs('public/results', exist_ok=True)

# Guardar imagen con efecto cómic
cv2.imwrite('public/results/ComicFilter2.jpg', imagen_comic)
print("Imagen guardada como 'public/results/ComicFilter2.jpg'")

cv2.waitKey(0)
cv2.destroyAllWindows()
