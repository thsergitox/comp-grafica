# --- ANÁLISIS DEL SCRIPT ---
#
# PROPÓSITO GENERAL:
# Este script utiliza la cámara web para detectar una mano en tiempo real, dibujar los
# puntos de referencia (landmarks) sobre ella y contar cuántos dedos están levantados.
#
# LIBRERÍAS UTILIZADAS:
# - cv2 (OpenCV): Para la captura de video, procesamiento de imágenes (conversión
#   de color) y para dibujar formas y texto en la imagen.
# - mediapipe (mp): Es la librería clave para la detección de manos. Proporciona
#   modelos pre-entrenados para localizar los 21 puntos de referencia de la mano.
#
# SECUENCIA DE EJECUCIÓN:
# 1.  Inicializa la captura de video y el detector de manos de MediaPipe.
# 2.  Entra en un bucle infinito para leer cuadros de la cámara.
# 3.  Convierte cada cuadro al formato de color RGB, requerido por MediaPipe.
# 4.  Procesa el cuadro para detectar los landmarks de la mano.
# 5.  Si se detecta una mano:
#     a. Dibuja los landmarks y las conexiones en la imagen.
#     b. Extrae las coordenadas en píxeles de cada landmark.
#     c. Implementa una lógica para contar los dedos levantados basándose en la posición
#        relativa de las puntas de los dedos.
#     d. Muestra el contador de dedos en pantalla.
# 6.  Muestra la imagen procesada en una ventana.
# 7.  El bucle termina si no se puede leer un cuadro de la cámara.

# Importación de las librerías necesarias.
import cv2
import mediapipe as mp

# Inicia la captura de video desde la cámara web (el índice 0 es usualmente la cámara por defecto).
video = cv2.VideoCapture(0)

# Accede al módulo de soluciones de MediaPipe para la detección de manos.
mano = mp.solutions.hands
# Crea una instancia del detector de manos. 'max_num_hands=1' le dice que solo busque una mano para optimizar el rendimiento.
Mano = mano.Hands(max_num_hands=1)
# Inicializa la utilidad de dibujo de MediaPipe para visualizar los landmarks y las conexiones de la mano.
mpDraw = mp.solutions.drawing_utils

# Inicia un bucle infinito para procesar el video cuadro a cuadro.
while True:
    # Lee un cuadro (frame) del video. 'check' es un booleano (True si la lectura fue exitosa) e 'img' es el cuadro en sí.
    check, img = video.read()
    # Si no se pudo leer el cuadro, se termina el bucle.
    if not check:
        break
    # Convierte la imagen de formato BGR (usado por OpenCV) a RGB, que es el formato que MediaPipe espera.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Procesa la imagen RGB con el detector de manos para encontrar los landmarks.
    results = Mano.process(imgRGB)
    # Extrae las coordenadas de los puntos de la mano detectados.
    handPoints = results.multi_hand_landmarks
    # Obtiene las dimensiones (alto, ancho) de la imagen. El guion bajo `_` ignora el canal de color.
    h,w, _ = img.shape
    # Inicializa una lista vacía para almacenar las coordenadas (en píxeles) de los puntos de la mano.
    puntos = []
    if handPoints:
        # Itera sobre los puntos de la mano encontrada.
        for points in handPoints:
            # Dibuja los puntos y las conexiones entre ellos sobre la imagen original.
            mpDraw.draw_landmarks(img, points, mano.HAND_CONNECTIONS)
            # Itera sobre cada punto (landmark), obteniendo su ID (0-20) y sus coordenadas normalizadas (de 0 a 1).
            for id, cord in enumerate(points.landmark):
                # Convierte las coordenadas normalizadas a coordenadas en píxeles.
                cx, cy = int(cord.x*w), int(cord.y*h)
                # Dibuja el ID de cada punto para facilitar la identificación.
                cv2.putText(img, str(id), (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # Añade las coordenadas en píxeles a la lista 'puntos'.
                puntos.append((cx, cy))

        # Define una lista con los IDs de las puntas de los dedos índice, medio, anular y meñique.
        dedos = [8, 12, 16, 20]
        # Inicializa el contador de dedos.
        contador = 0
        # Comprueba si la lista de puntos no está vacía.
        if points:
            # Lógica para el pulgar: Comprueba si el dedo pulgar está levantado.
            # Lo hace comparando la coordenada 'x' de la punta del pulgar (punto 4) con la de un punto más a la base (punto 2).
            # Esta lógica asume una orientación específica de la mano.
            if puntos[4][0] < puntos[2][0]:
                contador += 1
            # Lógica para los otros 4 dedos: Itera sobre los otros cuatro dedos.
            for x in dedos:
                # Comprueba si el dedo está levantado comparando la coordenada 'y' de la punta (ej. punto 8)
                # con la de un punto en la base del mismo dedo (punto 6).
                # Si la punta está más arriba (menor valor en 'y'), el dedo está levantado.
                if puntos[x][1] < puntos[x-2][1]:
                    contador +=1

        # Dibuja un rectángulo azul en la esquina superior izquierda de la imagen para mostrar el resultado.
        cv2.rectangle(img, (80, 10), (200, 100), (255,0,0), -1)
        # Muestra el número de dedos contados dentro del rectángulo, con una fuente grande y en color blanco.
        cv2.putText(img, str(contador), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 5)



    # Muestra la imagen procesada en una ventana llamada 'Imagen'.
    cv2.imshow('Imagen', img)
    # Espera 1 milisegundo por una tecla. Si se presiona una, el bucle se romperá (y permite que la ventana se refresque).
    cv2.waitKey(1)