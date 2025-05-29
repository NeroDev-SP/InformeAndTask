# 1. Importar la librería de visión por computadora
import cv2

# 2. Cargar el clasificador Haar para detección de rostros
clasificador_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Cargar el clasificador Haar para detección de sonrisas
clasificador_sonrisa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 4. Iniciar la cámara (0 indica la cámara por defecto)
camara = cv2.VideoCapture(0)

# 5. Verificar si la cámara se abrió correctamente
if not camara.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# 6. Iniciar el bucle para procesar video en tiempo real
while True:
    # 7. Leer el fotograma actual desde la cámara
    ret, frame = camara.read()

    # 8. Convertir la imagen a escala de grises para mejor procesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 9. Detectar rostros en la imagen en escala de grises
    rostros = clasificador_rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 10. Recorrer todos los rostros detectados
    for (x, y, w, h) in rostros:
        # 11. Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 12. Definir la región de interés (ROI) del rostro detectado en gris y color
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 13. Detectar sonrisas dentro de la región del rostro
        sonrisas = clasificador_sonrisa.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,      # Aumenta el tamaño de la imagen para detectar mejor detalles como la sonrisa
            minNeighbors=22,      # Requiere varias coincidencias para validar la sonrisa (evita falsos positivos)
            minSize=(25, 25)      # Tamaño mínimo de sonrisa a detectar
        )

        # 14. Recorrer las sonrisas detectadas y dibujar rectángulos sobre ellas
        for (sx, sy, sw, sh) in sonrisas:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            # 15. Mostrar texto indicando que se detectó una sonrisa
            cv2.putText(frame, 'Sonrisa :)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar la imagen procesada con detecciones
    cv2.imshow('Detección de Rostro y Sonrisa', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
camara.release()
cv2.destroyAllWindows()
#Derechos de autor NeroDev