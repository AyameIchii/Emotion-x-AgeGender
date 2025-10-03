import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------- CẤU HÌNH -----------
IMG_SIZE = 224
MODEL_PATH = "age_gender_model_vgg16_final.h5"

# ----------- LOAD MÔ HÌNH -----------
model = load_model(MODEL_PATH)

# ----------- HÀM DỰ ĐOÁN -----------
def predict_age_gender(face_img):
    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    age_pred, gender_pred = model.predict(img, verbose=0)

    # Tuổi: lấy giá trị dự đoán và ép int
    age = int(age_pred[0][0])

    # Giới tính: sigmoid → 0 = Nam, 1 = Nữ
    prob = gender_pred[0][0]
    gender = "Nam" if prob < 0.5 else "Nu"

    # Debug nếu cần
    # print(f"Raw gender_pred: {prob:.3f}")

    return age, gender

# ----------- MỞ WEBCAM -----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Khong Mo Duoc Webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        age, gender = predict_age_gender(face_img)
        label = f"{gender}, {age} tuoi"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("🎥 Du doan tuoi & gioi tinh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
