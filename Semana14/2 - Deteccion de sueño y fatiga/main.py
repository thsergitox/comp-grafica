# --- ANÁLISIS DEL SCRIPT ---
#
# PROPÓSITO GENERAL:
# Este programa detecta si una persona frente a la cámara tiene los ojos cerrados.
# Si los mantiene cerrados por más de un umbral de tiempo (2 segundos), muestra
# una alerta de "DORMIDO" en la pantalla, simulando un detector de fatiga.
#
# LIBRERÍAS UTILIZADAS:
# - cv2 (OpenCV): Para la captura y manipulación de video e imágenes.
# - mediapipe (mp): Utiliza el modelo `FaceMesh` para detectar una malla facial con
#   468 puntos, permitiendo un seguimiento preciso de los landmarks de los ojos.
# - math: Se usa para calcular la distancia euclidiana (`hypot`) entre los párpados
#   superior e inferior, lo que determina si un ojo está abierto o cerrado.
# - time: Para medir el tiempo que los ojos permanecen cerrados.
#
# SECUENCIA DE EJECUCIÓN:
# 1.  Inicializa la captura de video y el detector de malla facial de MediaPipe.
# 2.  Define variables para rastrear el estado (despierto/dormido) y el tiempo.
# 3.  Entra en un bucle infinito para leer cuadros de la cámara.
# 4.  Procesa cada cuadro para detectar la malla facial.
# 5.  Si se detecta una cara:
#     a. Extrae las coordenadas de los landmarks de los párpados de ambos ojos.
#     b. Calcula la distancia vertical entre los párpados para cada ojo.
#     c. Si ambas distancias son menores a un umbral, se considera que los ojos
#        están cerrados. Se inicia un temporizador.
#     d. Si los ojos están abiertos, se resetea el estado y el temporizador.
#     e. Si los ojos han estado cerrados por más de 2 segundos, se muestra una
#        alerta de "DORMIDO".
# 6.  Muestra la imagen procesada con las alertas correspondientes.

import cv2
import mediapipe as mp
import math
import time

# Inicia la captura de video desde la cámara web.
video = cv2.VideoCapture(0)
# Accede a la solución de malla facial de MediaPipe.
mpFaceMesh = mp.solutions.face_mesh
# Crea una instancia del detector de malla facial.
faceMesh = mpFaceMesh.FaceMesh()
# Inicializa la utilidad de dibujo de MediaPipe (aunque no se usa en este código).
mpDraw = mp.solutions.drawing_utils
# Inicializa variables para controlar el estado (dormido/despierto) y el tiempo.
estado = 'X'
inicio = 0
estado_actual = ''

# Inicia el bucle principal.
while True:
    # Lee un cuadro del video.
    check, img = video.read()
    # Redimensiona la imagen para unificar el tamaño de procesamiento y visualización.
    img = cv2.resize(img, (1000, 720))
    # Si no se pudo leer el cuadro, termina el bucle.
    if not check:
        break
    # Procesa la imagen para detectar la malla facial. No se convierte a RGB
    # porque FaceMesh puede manejar BGR, aunque RGB es recomendado.
    results = faceMesh.process(img)
    # Obtiene las dimensiones de la imagen.
    h, w, _ = img.shape

    # Comprueba si se detectó alguna cara.
    if results:
        # Si la lista de landmarks está vacía (no se detectaron caras), continúa a la siguiente iteración.
        if not results.multi_face_landmarks:
            continue
        # Itera sobre cada cara detectada (normalmente solo una).
        for face in results.multi_face_landmarks:
            # Obtiene las coordenadas en píxeles de 4 puntos clave de los párpados:
            # Puntos 159 (arriba) y 145 (abajo) para el ojo derecho.
            # Puntos 386 (arriba) y 374 (abajo) para el ojo izquierdo.
            d1x, d1y = int((face.landmark[159].x)*w), int((face.landmark[159].y)*h)
            d2x, d2y = int((face.landmark[145].x) * w), int((face.landmark[145].y) * h)
            i1x, i1y = int((face.landmark[386].x) * w), int((face.landmark[386].y) * h)
            i2x, i2y = int((face.landmark[374].x) * w), int((face.landmark[374].y) * h)

            # Calcula la distancia euclidiana (vertical) entre los párpados del ojo derecho.
            distD = math.hypot(d1x - d2x, d1y - d2y)
            # Calcula la distancia euclidiana (vertical) entre los párpados del ojo izquierdo.
            distI = math.hypot(i1x - i2x, i1y - i2y)


            # Si la distancia de ambos ojos es muy pequeña (menor o igual a 15 píxeles), se considera que están cerrados.
            if distI <= 15 and distD <= 15:
                # Dibuja un rectángulo rojo y el texto 'OJOS CERRADOS'.
                cv2.rectangle(img, (100, 30), (390, 80), (0,0,255), -1)
                cv2.putText(img, 'OJOS CERRADOS', (105,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                # Actualiza el estado a 'Dormido'.
                estado = 'Dormido'
                # Si el estado acaba de cambiar a 'Dormido' (previamente era 'Despierto'), guarda el tiempo de inicio.
                if estado != estado_actual:
                    inicio = time.time()
            else:
                # Si los ojos están abiertos, dibuja un rectángulo azul y el texto 'OJOS ABIERTOS'.
                cv2.rectangle(img, (100, 30), (390, 80), (255, 0, 0), -1)
                cv2.putText(img, 'OJOS ABIERTOS', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                # Actualiza el estado a 'Despierto' y resetea el temporizador.
                estado = 'Despierto'
                inicio = time.time()
                tiempo = int(time.time() - inicio)

            # Si el estado actual es 'Dormido', calcula cuánto tiempo ha pasado.
            if estado == 'Dormido':
                tiempo = int(time.time() - inicio)

                # Si el tiempo con los ojos cerrados es de 2 segundos o más...
                if tiempo >= 2:
                    # ...dibuja un rectángulo grande de alerta y muestra el tiempo que lleva dormido.
                    cv2.rectangle(img, (300, 150), (850, 220), (0,0,255), -1)
                    cv2.putText(img, f'DORMIDO: {tiempo} SEG', (310, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 5)
            # Actualiza el 'estado_actual' para la siguiente iteración. Esto es clave para detectar el *cambio* de estado.
            estado_actual = estado
    # Muestra la imagen en una ventana.
    cv2.imshow('Detector', img)
    # Espera 10 milisegundos por una tecla.
    cv2.waitKey(10)