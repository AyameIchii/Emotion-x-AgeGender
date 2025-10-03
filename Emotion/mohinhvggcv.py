# train_vgg16.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = "datacv/train"
test_dir = "datacv/test"

# Parameters
img_size = (48, 48)
batch_size = 64
epochs = 50

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, color_mode="rgb",
    class_mode="categorical", batch_size=batch_size, shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, color_mode="rgb",
    class_mode="categorical", batch_size=batch_size, shuffle=False
)

num_classes = train_gen.num_classes

# Base VGG16
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

# Freeze tất cả layer
for layer in base_model.layers:
    layer.trainable = False

# Cho phép fine-tune block cuối (block5)
for layer in base_model.layers[-16:]:
    layer.trainable = True

# Thêm head mới
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Tính class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Train
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    class_weight=class_weights
)

# Evaluate
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_gen.class_indices.keys(),
            yticklabels=train_gen.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16 (fine-tuned)")
plt.show()

# Plot accuracy & loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")
plt.show()

# Save accuracy để so sánh
np.save("vgg16_acc.npy", np.array([history.history["val_accuracy"][-1]]))

# 🔹 Lưu mô hình (full: kiến trúc + trọng số + optimizer)
model.save("vgg16_model.h5")
print("💾 Mô hình VGG16 đã được lưu vào vgg16cv_model.h5")
# 🔹 Save confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_gen.class_indices.keys(),
            yticklabels=train_gen.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16 (fine-tuned)")
plt.savefig("vgg16_cm.png")
plt.close()
print("📊 Confusion matrix saved to vgg16_cm.png")

# 🔹 Save summary table for comparison
summary_file = "model_summary.csv"
summary_data = {
    "model": ["VGG16"],
    "accuracy": [history.history["val_accuracy"][-1]],
    "epochs": [epochs],
    "train_samples": [train_gen.samples],
    "test_samples": [test_gen.samples]
}

df_summary = pd.DataFrame(summary_data)

# Nếu file đã tồn tại thì append, ngược lại tạo mới
if os.path.exists(summary_file):
    old_df = pd.read_csv(summary_file)
    df_summary = pd.concat([old_df, df_summary], ignore_index=True)

df_summary.to_csv(summary_file, index=False)
print(f"📑 Summary updated in {summary_file}")