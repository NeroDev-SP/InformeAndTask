import cv2

# Cargar el clasificador para cuerpo completo
clasificador_cuerpo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = camara.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cuerpos = clasificador_cuerpo.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    for (x, y, w, h) in cuerpos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow('Detección de Cuerpo Entero ND', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
#Derechos de autor NeroDev