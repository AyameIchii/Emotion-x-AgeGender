import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# --------- CẤU HÌNH ----------
IMG_SIZE = 224
DATASET_PATH = "UTKFace/"
BATCH_SIZE = 32
EPOCHS = 100   # để nhanh hơn, bạn có thể để 100 nếu GPU khỏe

# --------- CUSTOM DATA GENERATOR ----------
class UTKFaceGenerator(Sequence):
    def __init__(self, file_list, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.file_list = file_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, ages, genders = [], [], []
        for img_name in batch_files:
            try:
                parts = img_name.split("_")
                age = int(parts[0])
                gender = int(parts[1])
                img_path = os.path.join(DATASET_PATH, img_name)

                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype("float32") / 255.0

                images.append(img)
                ages.append(age)
                genders.append(gender)
            except:
                continue
        return np.array(images), {"age": np.array(ages), "gender": np.array(genders)}

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_list)

# --------- CHUẨN BỊ DATASET ----------
all_files = [f for f in os.listdir(DATASET_PATH) if "_" in f]

# Train: 60%, Val: 20%, Test: 20%
train_files, test_files = train_test_split(all_files, test_size=0.4, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

print("📊 Số mẫu:")
print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))

train_gen = UTKFaceGenerator(train_files, batch_size=BATCH_SIZE)
val_gen = UTKFaceGenerator(val_files, batch_size=BATCH_SIZE, shuffle=False)
test_gen = UTKFaceGenerator(test_files, batch_size=BATCH_SIZE, shuffle=False)

# --------- MÔ HÌNH ----------
base_model = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")

# Fine-tune các lớp cuối
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
age_output = Dense(1, activation="linear", name="age")(x)
gender_output = Dense(1, activation="sigmoid", name="gender")(x)

model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss={"age": "mae", "gender": "binary_crossentropy"},
    metrics={"age": "mae", "gender": "accuracy"}
)

model.summary()

# --------- TRAIN ----------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --------- ĐÁNH GIÁ TRÊN TEST ----------
test_results = model.evaluate(test_gen, verbose=1)
metrics = dict(zip(model.metrics_names, test_results))
print("\n📌 Kết quả trên tập Test:", metrics)

# --------- CONFUSION MATRIX - GIỚI TÍNH ----------
y_true_gender, y_pred_gender = [], []
for batch_x, batch_y in test_gen:
    preds = model.predict(batch_x, verbose=0)
    pred_gender = (preds[1].ravel() > 0.5).astype(int)  # sigmoid -> nhị phân
    y_pred_gender.extend(pred_gender)
    y_true_gender.extend(batch_y["gender"])

y_true_gender = np.array(y_true_gender)
y_pred_gender = np.array(y_pred_gender)

cm = confusion_matrix(y_true_gender, y_pred_gender)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nam", "Nữ"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Ma trận nhầm lẫn - Giới tính")
plt.show()

print("\n📊 Classification Report (Giới tính):")
print(classification_report(y_true_gender, y_pred_gender, target_names=["Nam", "Nữ"]))

# --------- SUMMARY TABLE ----------
summary_data = {"Metric": model.metrics_names, "Value": test_results}
df_summary = pd.DataFrame(summary_data)
print("\n📌 Summary kết quả trên tập Test:")
print(df_summary)

# --------- VẼ BIỂU ĐỒ HUẤN LUYỆN ----------
plt.figure(figsize=(12, 5))

# --- Age MAE ---
plt.subplot(1, 2, 1)
plt.plot(history.history["age_mae"], label="Train Age MAE")
plt.plot(history.history["val_age_mae"], label="Val Age MAE")
plt.title("MAE Dự đoán Tuổi")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

# --- Gender Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(history.history["gender_accuracy"], label="Train Gender Acc")
plt.plot(history.history["val_gender_accuracy"], label="Val Gender Acc")
plt.title("Độ chính xác Giới tính")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# --------- LƯU MÔ HÌNH ----------
model.save("age_gender_model_vgg16_final.h5")
print("✅ Mô hình đã lưu: age_gender_model_vgg16_final.h5")
