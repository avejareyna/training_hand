import os
import random
import numpy as np
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model

IMG_FOLDER = INPUT_DIR = Path(__file__).parent.parent / "data" / "test"
MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "best_model.keras"

class_names = ['paper','rock' ]  

# Tamaño de entrada del modelo
IMG_SIZE = (128, 128)  

def predict():

    # Cargar modelo
    model = load_model(MODEL_PATH)

    # Listar imágenes válidas
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(valid_exts)]

    if not image_files:
        raise ValueError("No se encontraron imágenes en la carpeta.")

    # Seleccionar imagen aleatoria
    chosen_img_name = random.choice(image_files)
    chosen_img_path = os.path.join(IMG_FOLDER, chosen_img_name)
    print(f"[INFO] Imagen seleccionada: {chosen_img_name}")

    # Cargar y preprocesar imagen
    img = cv2.imread(chosen_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Asegurar formato RGB
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # (1, 128, 128, 3)

    # Predicción
    prediction = model.predict(img_input, verbose=0)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    # Mostrar resultado
    label = f"{class_names[class_idx]} ({confidence*100:.1f}%)"
    img_display = cv2.resize(img, (300, 300))
    cv2.putText(img_display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Predicción", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
