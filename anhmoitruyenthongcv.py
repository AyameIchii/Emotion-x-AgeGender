import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from tkinter import Tk, filedialog   # ✅ để mở hộp thoại chọn file

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

# ==== Load Haar Cascade để phát hiện mặt ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Hàm trích xuất HOG ====
def extract_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)

# ==== Hàm dự đoán trên khuôn mặt ====
def predict_face(face_img):
    gray = resize(face_img, img_size, anti_aliasing=True)
    hog_feat = extract_hog(gray).reshape(1, -1)
    feat_pca = pca.transform(hog_feat)
    feat_scaled = scaler.transform(feat_pca)
    pred = clf.predict(feat_scaled)[0]
    return label_map[pred]

# ==== Hàm dự đoán 1 ảnh (tìm mặt trước) ====
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không đọc được ảnh:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("❌ Không tìm thấy khuôn mặt trong ảnh:", image_path)
    else:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label = predict_face(face_img)

            # Vẽ khung + label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print("👉 Ảnh:", image_path, "→", label)

    # Hiển thị ảnh kết quả
    cv2.imshow("Kết quả dự đoán", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ✅ Mở hộp thoại chọn ảnh
    root = Tk()
    root.withdraw()  # ẩn cửa sổ gốc Tkinter
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để dự đoán",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *jfif")]
    )
    if file_path:
        predict_image(file_path)
    else:
        print("❌ Bạn chưa chọn ảnh nào.")
