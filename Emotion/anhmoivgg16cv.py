# predict_vgg16.py
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

# ==== Load model ====
model = tf.keras.models.load_model("vgg16_model.h5")

# ==== Nhãn (giống như train_gen.class_indices) ====
label_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}

img_size = (48, 48)

# ==== Haar cascade để phát hiện khuôn mặt ====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không đọc được ảnh:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        print("❌ Không tìm thấy khuôn mặt trong ảnh:", image_path)
    else:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            # Chuẩn hóa ảnh khuôn mặt
            face_resized = cv2.resize(face, img_size)
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # Dự đoán
            pred = model.predict(face_resized)
            label = label_map[np.argmax(pred)]

            print("👉 Ảnh:", image_path, "→", label)

            # Vẽ khung và label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Kết quả dự đoán - VGG16", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Mở hộp thoại chọn ảnh
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để dự đoán",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *jfif")]
    )
    if file_path:
        predict_image(file_path)
    else:
        print("❌ Bạn chưa chọn ảnh nào.")
