# predict_webcam_vgg16.py
import cv2
import numpy as np
import tensorflow as tf

# ==== Load model ====
model = tf.keras.models.load_model("vgg16_model.h5")

# ==== Nhãn ====
label_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}
img_size = (48, 48)

# Haar cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_face(face_img):
    face_resized = cv2.resize(face_img, img_size)
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)
    pred = model.predict(face_resized)
    return label_map[np.argmax(pred)]

def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]  # crop khuôn mặt màu (RGB)
            label = predict_face(face_img)

            # Vẽ khung + nhãn
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Webcam Emotion Recognition - VGG16", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_webcam()
