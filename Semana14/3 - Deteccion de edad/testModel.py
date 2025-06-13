# --- ANÁLISIS DEL SCRIPT ---
#
# PROPÓSITO GENERAL:
# Este proyecto detecta rostros en un video y utiliza dos modelos de Deep Learning
# (guardados como archivos .h5) para predecir el GÉNERO (Hombre/Mujer) y un
# RANGO DE EDAD de la persona detectada. Las predicciones se muestran en tiempo real.
#
# LIBRERÍAS UTILIZADAS:
# - numpy (np): Para el manejo de arrays, fundamental para preparar la imagen del
#   rostro en el formato que el modelo de Keras requiere (un "batch" o lote).
# - cv2 (OpenCV): Para captura de video, manipulación de imágenes (recorte,
#   redimensión, conversión de color) y para dibujar los resultados.
# - keras.models.load_model: Para cargar los modelos de predicción de género y
#   edad previamente entrenados.
# - mediapipe (mp): Se usa su detector de rostros (`FaceDetection`) para localizar
#   caras en la imagen de forma robusta.
# - cvzone: Una librería de conveniencia que simplifica dibujar texto con un fondo
#   rectangular sobre imágenes de OpenCV.
#
# SECUENCIA DE EJECUCIÓN:
# 1.  Carga las librerías necesarias.
# 2.  Inicializa el detector de rostros de MediaPipe.
# 3.  Inicia la captura de video desde la cámara.
# 4.  Carga los dos modelos .h5 (género y edad). `compile=False` se usa para
#     acelerar la carga, ya que los modelos solo se usarán para inferencia.
# 5.  Define las etiquetas de clase para las predicciones (e.g., 'Hombre', '6-20').
# 6.  Entra en un bucle infinito para procesar el video:
#     a. Lee un cuadro y lo convierte a RGB.
#     b. Detecta rostros en el cuadro con MediaPipe.
#     c. Por cada rostro detectado:
#        i.  Recorta la región del rostro de la imagen.
#        ii. Redimensiona el rostro a 224x224 píxeles (el tamaño de entrada del modelo).
#        iii. Pre-procesa la imagen: la convierte a un array de numpy, ajusta su
#            forma (batch_size, height, width, channels) y normaliza sus
#            píxeles al rango [-1, 1].
#        iv. Pasa el rostro pre-procesado a ambos modelos para obtener predicciones.
#        v.  Extrae la clase con mayor probabilidad y su nivel de confianza.
#        vi. Si la confianza supera un umbral, dibuja un recuadro y muestra las
#            predicciones (género, edad y confianza) sobre el rostro usando cvzone.
# 7.  Muestra el video resultante con las anotaciones.

import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp
import cvzone

# Comentarios en el código original sobre versiones sugeridas
#tf 2.9.1
#keras 2.6.0

# Accede a la solución de detección de rostros de MediaPipe.
face = mp.solutions.face_detection
# Crea una instancia del detector.
Face = face.FaceDetection()
# Inicializa la utilidad de dibujo de MediaPipe (aunque no se usa en el código final).
mpDwaw = mp.solutions.drawing_utils

# Inicia la captura de video. Puede ser desde la cámara (0) o un archivo de video.
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Videos/vd06.mp4')

# Carga el modelo entrenado para predecir el género. 'compile=False' acelera la carga.
modelGender = load_model("model_gender.h5", compile=False)

# Carga el modelo entrenado para predecir la edad.
modelAge = load_model("model_age.h5", compile=False)

# Define las etiquetas de clase para el GÉNERO. El orden debe coincidir con el entrenamiento.
classesGender=['Hombre','Mujer']
# Define las etiquetas para los RANGOS DE EDAD.
classesAge=['6-20','25-30','42-48','60-98']

# Inicia el bucle principal.
while True:
    # Lee un cuadro del video.
    success, imgOrignal = cap.read()
    # Convierte la imagen a RGB, el formato esperado por MediaPipe.
    imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
    # Procesa la imagen para detectar rostros.
    results = Face.process(imgRGB)
    # Extrae la información de los rostros detectados.
    facesPoints = results.detections
    # Obtiene las dimensiones de la imagen original.
    hO, wO, _ = imgRGB.shape
    if facesPoints:
        # ...itera sobre cada rostro encontrado.
        for id, detection in enumerate(facesPoints):
            # Obtiene el cuadro delimitador (bounding box) relativo del rostro.
            bbox = detection.location_data.relative_bounding_box
            # Calcula las coordenadas en píxeles del cuadro delimitador.
            x,y,w,h = int(bbox.xmin*wO),int(bbox.ymin*hO),int(bbox.width*wO),int(bbox.height*hO)
            # Recorta la imagen original para obtener solo el rostro.
            imgFace = imgOrignal[y:y + h, x:x + w]
            # Redimensiona la imagen del rostro a 224x224, el tamaño de entrada que los modelos esperan.
            imgFace = cv2.resize(imgFace, (224, 224))
            # Pre-procesamiento para el modelo:
            # 1. Convierte el rostro a un array de numpy.
            # 2. Cambia la forma a (1, 224, 224, 3) -> un lote (batch) de 1 imagen.
            face = np.asarray(imgFace, dtype=np.float32).reshape(1, 224, 224, 3)
            # 3. Normaliza los valores de los píxeles al rango [-1, 1], un paso común para muchos modelos de redes neuronales.
            face = (face / 127.5) - 1

            # Realiza la predicción de GÉNERO usando el modelo correspondiente.
            predictions = modelGender.predict(face)
            # Encuentra el índice de la clase con la probabilidad más alta (0 para Hombre, 1 para Mujer).
            indexGender = np.argmax(predictions)
            # Obtiene el valor de confianza (la probabilidad más alta) de esa predicción.
            confGender = np.amax(predictions)

            # Realiza la predicción de EDAD.
            predictions = modelAge.predict(face)
            # Encuentra el índice del rango de edad con la probabilidad más alta.
            indexAge = np.argmax(predictions)
            # Obtiene la confianza de la predicción de edad.
            confAge = np.amax(predictions)

            # Si la confianza en la predicción de género es mayor a 30%...
            if confGender >0.30:
                # ...dibuja un rectángulo verde alrededor del rostro.
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0,255,0), 3)
                # Muestra la predicción de género (ej. 'Hombre') sobre el rectángulo.
                cvzone.putTextRect(imgOrignal, str(classesGender[indexGender]), (x, y-15), 2, 3)
                # Muestra el porcentaje de confianza de la predicción.
                cvzone.putTextRect(imgOrignal, str(round(confGender * 100, 2)) + "%", (x+130, y-15), 1.5, 2)

            # Si la confianza en la predicción de edad es mayor al 40%...
            if confAge > 0.40:
                # ...muestra la predicción del rango de edad.
                cvzone.putTextRect(imgOrignal,f'Edad: {classesAge[indexAge]}',(x, y-50),2,3)


    # Muestra la imagen final con las predicciones en una ventana.
    cv2.imshow("Result", imgOrignal)
    # Espera 15 milisegundos.
    cv2.waitKey(15)

