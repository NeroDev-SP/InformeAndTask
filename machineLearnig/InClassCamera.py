import cv2

imagen = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#captura de la imagen
camara = cv2.VideoCapture(0)
if camara.isOpened() == False:
    print("Error: No se pudo abrir la camara")
    exit()
while True:
    ret, frame = camara.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = imagen.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('camara', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camara.release()
cv2.destroyAllWindows()
#Derechos de autor NeroDev