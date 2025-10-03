from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import joblib
from skimage.feature import hog

app = Flask(__name__)

# ========== AGE & GENDER ==========
IMG_SIZE = 224
AGE_GENDER_MODEL_PATH = "age_gender_model_vgg16_final.h5"
age_gender_model = load_model(AGE_GENDER_MODEL_PATH)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_age_gender(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ========== EMOTION MODELS ==========
# Deep Learning model
emotion_deep = load_model("vgg16_model.h5")
# Truyền thống (SVM/RF/…)
trad_clf   = joblib.load("sgd_svm_model.pkl")
trad_pca   = joblib.load("pca_model.pkl")
trad_scaler= joblib.load("scaler.pkl")
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def preprocess_emotion(img):
    img = cv2.resize(img, (48, 48))   # hoặc (224,224) tùy model
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def extract_hog(image):
    # chuyển ảnh sang grayscale như khi train
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))  # dùng đúng size lúc train
    return hog(gray,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               visualize=False)

# ========== ROUTES ==========
@app.route("/")
def home():
    return render_template("home.html")

# ---- Trang chủ: Tuổi & Giới tính ----
@app.route("/age")
def index():
    return render_template("age/index.html")   # giao diện age/gender

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "❌ Không có file ảnh."})

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "❌ Không đọc được ảnh."})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return jsonify({"error": "❌ Không tìm thấy khuôn mặt nào."})

    results = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        x_in = preprocess_age_gender(face_img)
        age_pred, gender_pred = age_gender_model.predict(x_in, verbose=0)
        age = int(age_pred[0][0])

        prob_female = float(gender_pred[0][0])
        if prob_female < 0.5:
            gender = "Nam"
            confidence = round((1 - prob_female) * 100, 2)
        else:
            gender = "Nu"
            confidence = round(prob_female * 100, 2)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f"{gender}, {age}t", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, crop_buf = cv2.imencode('.jpg', face_img)
        cropped_b64 = base64.b64encode(crop_buf).decode('utf-8')

        results.append({
            "age": age,
            "gender": gender,
            "confidence": confidence,
            "cropped_face": cropped_b64
        })

    _, full_buf = cv2.imencode('.jpg', img)
    full_b64 = base64.b64encode(full_buf).decode('utf-8')

    return jsonify({
        "image_with_box": full_b64,
        "results": results
    })

# Webcam cho tuổi & giới tính
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            x_in = preprocess_age_gender(face_img)
            age_pred, gender_pred = age_gender_model.predict(x_in, verbose=0)
            age = int(age_pred[0][0])
            gender = "Nam" if gender_pred[0][0] < 0.5 else "Nu"

            label = f"{gender}, {age}t"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Trang & API Phân tích Cảm xúc ----
@app.route("/emoo")
def emotion_page():
    return render_template("emoo/index.html")

@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "❌ Không có file ảnh."})

    file_bytes = np.frombuffer(request.files["file"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "❌ Không đọc được ảnh."})

    # Resize ảnh nếu quá lớn (giúp Haar Cascade hoạt động tốt hơn)
    if img.shape[1] > 600:
        scale = 600.0 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return jsonify({"error": "❌ Không tìm thấy khuôn mặt nào."})

    # Copy ảnh để vẽ kết quả riêng cho deep/trad
    deep_img = img.copy()
    trad_img = img.copy()
    results = []

    for (x, y, w, h) in faces:
        # Giới hạn bounding box trong biên ảnh
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)
        if x + w > img.shape[1] or y + h > img.shape[0]:
            continue

        face_img = img[y:y+h, x:x+w]
        if face_img is None or face_img.size == 0:
            continue

        # Deep Learning model
        x_in = preprocess_emotion(face_img)
        deep_pred = emotion_deep.predict(x_in, verbose=0)
        deep_idx = int(np.argmax(deep_pred))
        deep_conf = round(float(np.max(deep_pred)) * 100, 2)
        deep_emotion = emotion_labels[deep_idx]

        cv2.rectangle(deep_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(deep_img, deep_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Truyền thống model
        feat = extract_hog(face_img).reshape(1, -1)
        feat_pca = trad_pca.transform(feat)
        feat_scaled = trad_scaler.transform(feat_pca)
        trad_idx = int(trad_clf.predict(feat_scaled)[0])
        trad_emotion = emotion_labels[trad_idx]

        cv2.rectangle(trad_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(trad_img, trad_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        results.append({
            "deep": {"emotion": deep_emotion, "confidence": deep_conf},
            "trad": {"emotion": trad_emotion}
        })

    # Encode ảnh sau khi vẽ tất cả khuôn mặt
    _, buf_orig = cv2.imencode('.jpg', img)
    _, buf_deep = cv2.imencode('.jpg', deep_img)
    _, buf_trad = cv2.imencode('.jpg', trad_img)

    return jsonify({
        "results": results,
        "original_image": base64.b64encode(buf_orig).decode("utf-8"),
        "deep_image": base64.b64encode(buf_deep).decode("utf-8"),
        "trad_image": base64.b64encode(buf_trad).decode("utf-8")
    })




def gen_frames_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam Emotion")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Dùng config giống file test local
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            face_img_gray = gray[y:y+h, x:x+w]
            face_img_color = frame[y:y+h, x:x+w]

            # Deep model (VGG16) giống file webcamvgg16cv.py
            if face_img_color.size != 0:
                x_in = cv2.resize(face_img_color, (48, 48))
                x_in = x_in.astype("float32") / 255.0
                x_in = np.expand_dims(x_in, axis=0)
                deep_pred = emotion_deep.predict(x_in, verbose=0)
                deep_label = emotion_labels[int(np.argmax(deep_pred))]

            # Truyền thống (HOG + SVM) giống file webcamtruyenthongcv.py
            if face_img_gray.size != 0:
                face_resized = cv2.resize(face_img_gray, (48, 48))
                hog_feat = hog(face_resized, orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               visualize=False).reshape(1, -1)
                feat_pca = trad_pca.transform(hog_feat)
                feat_scaled = trad_scaler.transform(feat_pca)
                trad_idx = trad_clf.predict(feat_scaled)[0]
                trad_label = emotion_labels[int(trad_idx)]

            # Vẽ khung + nhãn y như local test
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Deep: {deep_label}", (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Trad: {trad_label}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed_emotion")
def video_feed_emotion():
    return Response(gen_frames_emotion(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
