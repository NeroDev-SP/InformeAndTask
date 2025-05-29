import cv2

# Cargar la imagen
image = cv2.imread('public/imageTest/neymar4.jpg')

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen...")
    exit()

# Aplicar efecto acuarela
image_acuarela = cv2.stylization(image, sigma_s=60, sigma_r=0.6)

# Agregar texto a la imagen
cv2.putText(image_acuarela, 'Filtro Acuarela', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Mostrar imágenes
cv2.imshow('Original', image)
cv2.imshow('Acuarela', image_acuarela)

# Guardar imagen
cv2.imwrite('public/results/AcuarelaFilter2.jpg', image_acuarela)
print("Imagen guardada como 'AcuarelaFilter2.jpg'")
cv2.waitKey(0)
cv2.destroyAllWindows()
#Derechos de autor NeroDev