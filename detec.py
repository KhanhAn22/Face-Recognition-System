import cv2
import numpy as np
from keras.models import load_model
import os

# Load Keras
keras_model = load_model("my-model.keras")

def preprocess_face(image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.resize(processed_image, (100, 40))
    processed_image = np.reshape(processed_image, (1, 40 * 100)).astype('float32')
    return processed_image

#hàm tách tên từ tên file ảnh
def extract_name(file_name):
    parts = file_name.split("_")
    if len(parts) >= 1:
        return parts[0]
    return "Unknown"

data_directory = "data/Face"

known_names = set()

for file in os.listdir(data_directory):
    if file.endswith(".jpg"):
        name = extract_name(file)
        known_names.add(name)

# Thực hiện xử lý trên ảnh
image_path = "data/Face/An_586.jpg"
frame = cv2.imread(image_path)

preprocessed_face = preprocess_face(frame)

# Dự đoán bằng Keras
prediction = keras_model.predict(preprocessed_face)

threshold = 0.1

if 0 <= prediction <= 1 and prediction >= threshold:
    name = extract_name(os.path.basename(image_path))
    if name in known_names:
        print(f"Name: {name}")
    else:
        print("Không có trong dữ liệu")
else:
    print("Không nhận diện được")