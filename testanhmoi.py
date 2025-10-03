# test_image.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

IMG_SIZE = 224
MODEL_PATH = "age_gender_model_vgg16_final.h5"
# Load model
model = load_model(MODEL_PATH)
# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Xử lý ảnh
def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img
# Dự đoán từ ảnh
def predict_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        print("❌ Không tìm thấy khuôn mặt nào trong ảnh.")
        return
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        img_input = preprocess_image(face_img)
        # Model trả về [age, gender]
        age_pred, gender_pred = model.predict(img_input)
        predicted_age = int(age_pred[0])
        predicted_gender = "Nam" if gender_pred[0] < 0.5 else "Nu"

        # Vẽ khung + text lên ảnh
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"Tuoi: {predicted_age}, GT: {predicted_gender}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị ảnh
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.axis('off')
    plt.show()

# Mở hộp thoại chọn ảnh
def open_image_dialog():
    Tk().withdraw()
    path = askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.jfif")])
    if path:
        predict_image(path)
    else:
        print("❌ Không có ảnh nào được chọn.")

# Chạy
open_image_dialog()
