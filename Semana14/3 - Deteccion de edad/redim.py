import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp
import cvzone
import os

# --- ANÁLISIS DEL SCRIPT ---
#
# PROPÓSITO GENERAL:
# Este script procesa archivos de video de una carpeta específica ('Videos/').
# Para cada video, detecta rostros en tiempo real y utiliza dos modelos de Deep
# Learning (.h5) para predecir el GÉNERO y un RANGO DE EDAD de las personas.
#
# LIBRERÍAS UTILIZADAS:
# - numpy, cv2, keras, mediapipe, cvzone: Mismas que en 'testModel.py'.
# - os: Para interactuar con el sistema de archivos, listar los videos en el directorio.
#
# SECUENCIA DE EJECUCIÓN:
# 1.  Carga los modelos de Keras y el detector de rostros de MediaPipe.
# 2.  Define la ruta a la carpeta que contiene los videos.
# 3.  Itera sobre cada archivo en la carpeta de videos.
# 4.  Para cada archivo de video:
#     a. Inicia la captura de video desde ese archivo.
#     b. Entra en un bucle para leer el video cuadro a cuadro.
#     c. Aplica la misma lógica de 'testModel.py' para detectar rostros,
#        pre-procesarlos y predecir género y edad.
#     d. Dibuja las predicciones en el cuadro.
#     e. Muestra el cuadro procesado.
# 5.  El bucle de un video termina cuando el video acaba o se presiona una tecla.
#     Luego, continúa con el siguiente video en la carpeta.

# --- INICIO DEL CÓDIGO ---

# Accede a la solución de detección de rostros de MediaPipe.
face = mp.solutions.face_detection
# Crea una instancia del detector.
Face = face.FaceDetection()

# Carga el modelo entrenado para predecir el género.
modelGender = load_model("model_gender.h5", compile=False)
# Carga el modelo entrenado para predecir la edad.
modelAge = load_model("model_age.h5", compile=False)

# Define las etiquetas de clase para el GÉNERO y EDAD.
classesGender = ['Hombre', 'Mujer']
classesAge = ['6-20', '25-30', '42-48', '60-98']

# Ruta a la carpeta que contiene los videos
videos_path = 'Videos/'
video_files = os.listdir(videos_path)

# Itera sobre cada archivo de video en la carpeta
for video_file in video_files:
    video_path = os.path.join(videos_path, video_file)
    print(f"Procesando video: {video_file}")

    # Inicia la captura de video desde el archivo actual.
    cap = cv2.VideoCapture(video_path)

    # Bucle para procesar el video actual cuadro a cuadro
    while True:
        # Lee un cuadro del video.
        success, imgOrignal = cap.read()
        # Si 'success' es False, significa que el video terminó. Rompemos el bucle.
        if not success:
            break

        # Convierte la imagen a RGB, el formato esperado por MediaPipe.
        imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
        # Procesa la imagen para detectar rostros.
        results = Face.process(imgRGB)
        # Extrae la información de los rostros detectados.
        facesPoints = results.detections
        # Obtiene las dimensiones de la imagen original.
        hO, wO, _ = imgRGB.shape

        if facesPoints:
            for id, detection in enumerate(facesPoints):
                # Obtiene y calcula las coordenadas del cuadro delimitador.
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * wO), int(bbox.ymin * hO), int(bbox.width * wO), int(bbox.height * hO)

                # Asegurarse de que el recorte no esté fuera de los límites de la imagen
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue

                # Recorta la imagen del rostro.
                imgFace = imgOrignal[y:y + h, x:x + w]

                # Pre-procesamiento para el modelo.
                try:
                    imgFace = cv2.resize(imgFace, (224, 224))
                except cv2.error:
                    continue # Si hay un error al redimensionar (ej. cara en el borde), saltar este rostro.

                face = np.asarray(imgFace, dtype=np.float32).reshape(1, 224, 224, 3)
                face = (face / 127.5) - 1

                # Predicción de GÉNERO
                predictions_gender = modelGender.predict(face, verbose=0)
                indexGender = np.argmax(predictions_gender)
                confGender = np.amax(predictions_gender)

                # Predicción de EDAD
                predictions_age = modelAge.predict(face, verbose=0)
                indexAge = np.argmax(predictions_age)
                confAge = np.amax(predictions_age)

                # Mostrar resultados si la confianza es suficiente
                if confGender > 0.30:
                    cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cvzone.putTextRect(imgOrignal, str(classesGender[indexGender]), (x, y - 15), 2, 3)
                    cvzone.putTextRect(imgOrignal, str(round(confGender * 100, 2)) + "%", (x + 130, y - 15), 1.5, 2)

                if confAge > 0.40:
                    cvzone.putTextRect(imgOrignal, f'Edad: {classesAge[indexAge]}', (x, y - 50), 2, 3)

        # Muestra el cuadro procesado.
        cv2.imshow("Result", imgOrignal)
        # Espera 1ms. Si se presiona una tecla, rompe el bucle para pasar al siguiente video.
        if cv2.waitKey(1) != -1:
            break
    
    # Libera el objeto de captura de video antes de pasar al siguiente.
    cap.release()
    # Si se presionó una tecla para salir del video, también salimos del bucle principal de videos.
    if cv2.waitKey(1) != -1:
        break

# Cierra todas las ventanas de OpenCV al finalizar.
cv2.destroyAllWindows()



