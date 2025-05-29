import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('public/imageTest/neymar4.jpg')

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen...")
    exit()

# Filtro sepia
filter_sepia = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

# Aplicar filtro sepia
image_sepia = cv2.transform(image, filter_sepia)
image_sepia = np.clip(image_sepia, 0, 255).astype(np.uint8)

# Agregar texto a la imagen
cv2.putText(image_sepia, 'Filtro Sepia ND', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Mostrar imágenes
cv2.imshow('Original', image)
cv2.imshow('Sepia', image_sepia)

# Guardar imagen
cv2.imwrite('public/results/SepiaFilter2.jpg', image_sepia)
print("Imagen guardada como 'SepiaFilter2.jpg'")

cv2.waitKey(0)
cv2.destroyAllWindows()
#Derechos de autor NeroDev