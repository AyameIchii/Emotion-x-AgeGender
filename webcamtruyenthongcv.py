import cv2
import joblib
from skimage.feature import hog
from skimage.transform import resize

# ==== Cấu hình ====
img_size = (48, 48)
label_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}

# ==== Load model, PCA, Scaler ====
clf = joblib.load("sgd_svm_model.pkl")
pca = joblib.load("pca_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==== Load Haar Cascade để phát hiện khuôn mặt ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Hàm trích xuất HOG ====
def extract_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)

# ==== Hàm dự đoán trên 1 khuôn mặt ====
def predict_face(face_img):
    face_resized = resize(face_img, img_size, anti_aliasing=True)
    hog_feat = extract_hog(face_resized).reshape(1, -1)
    feat_pca = pca.transform(hog_feat)
    feat_scaled = scaler.transform(feat_pca)
    pred = clf.predict(feat_scaled)[0]
    return label_map[pred]

# ==== Dự đoán realtime webcam ====
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
            face_img = gray[y:y+h, x:x+w]
            label = predict_face(face_img)

            # Vẽ khung + nhãn
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Webcam Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_webcam()
