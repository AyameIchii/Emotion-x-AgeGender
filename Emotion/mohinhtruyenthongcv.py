# train_sgd_svm.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # ✅ để lưu model
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import pandas as pd
from sklearn.metrics import hinge_loss
from sklearn.utils.class_weight import compute_class_weight

# Paths
train_dir = "datacv/train"
test_dir = "datacv/test"
img_size = (48, 48)
pca_components = 300

# Load data
def load_data_from_dir(directory, label_map):
    X, y = [], []
    for label_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, label_name)
        if not os.path.isdir(class_dir):
            continue
        label = label_map[label_name]
        print(f"Loading {label_name} → {label}")
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            if not img_path.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            try:
                img = imread(img_path)
                if img.ndim == 3:
                    img = rgb2gray(img)
                img = resize(img, img_size, anti_aliasing=True)
                X.append(img)
                y.append(label)
            except Exception as e:
                print("❌ Error:", img_path, e)
    return np.array(X), np.array(y)

# HOG feature extraction
def extract_hog_features(images):
    features = []
    for img in images:
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
    return np.array(features)

# Label mapping
label_map = {
    "angry": 0, "disgust": 1, "fear": 2,
    "happy": 3, "sad": 4, "surprise": 5, "neutral": 6
}

print("Loading data...")
X_train, y_train = load_data_from_dir(train_dir, label_map)
X_test, y_test = load_data_from_dir(test_dir, label_map)
print("✅ Data loaded:", X_train.shape, X_test.shape)

print("Extracting HOG...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("Applying PCA...")
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train_hog)
X_test_pca = pca.transform(X_test_hog)

print("Scaling features...")
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# Thay đoạn train SGDClassifier bằng custom training loop
print("Training SGDClassifier (SVM) with loss curve...")

classes = np.unique(y_train)

# ✅ Tính class_weight thủ công
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

clf = SGDClassifier(
    loss="hinge",
    max_iter=1,    # chỉ train 1 vòng mỗi lần
    tol=None,
    warm_start=True,
    random_state=42
)

n_epochs = 50
train_accs, val_accs = [], []
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(n_epochs):
    # sample_weight dựa vào class_weight
    sample_weight = np.array([class_weight_dict[label] for label in y_train])

    clf.partial_fit(X_train_pca, y_train, classes=classes, sample_weight=sample_weight)

    # ---- Loss ----
    train_pred_decision = clf.decision_function(X_train_pca)
    train_loss = hinge_loss(y_train, train_pred_decision, labels=clf.classes_)
    train_losses.append(train_loss)

    val_pred_decision = clf.decision_function(X_test_pca)
    val_loss = hinge_loss(y_test, val_pred_decision, labels=clf.classes_)
    val_losses.append(val_loss)

    # ---- Accuracy ----
    train_pred = clf.predict(X_train_pca)
    val_pred = clf.predict(X_test_pca)
    train_accs.append(accuracy_score(y_train, train_pred))
    val_accs.append(accuracy_score(y_test, val_pred))

    print(f"Epoch {epoch+1}/{n_epochs} "
          f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} "
          f"- Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

# ---- Vẽ biểu đồ ----
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
axs[0].plot(range(1, n_epochs+1), train_accs, label="train_acc")
axs[0].plot(range(1, n_epochs+1), val_accs, label="val_acc")
axs[0].set_title("Accuracy")
axs[0].legend()

# Loss
axs[1].plot(range(1, n_epochs+1), train_losses, label="train_loss")
axs[1].plot(range(1, n_epochs+1), val_losses, label="val_loss")
axs[1].set_title("Loss")
axs[1].legend()

plt.tight_layout()
plt.show()


# Evaluate
print("Evaluating...")
y_pred = clf.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print("SGD-SVM Accuracy:", acc)
print(classification_report(y_test, y_pred))

# 🔹 Save accuracy
np.save("sgd_svm_acc.npy", np.array([acc]))

# 🔹 Save classification report
with open("sgd_svm_report.txt", "w") as f:
    f.write("SGD-SVM Accuracy: {:.4f}\n\n".format(acc))
    f.write(classification_report(y_test, y_pred))

# 🔹 Save model + PCA + Scaler
joblib.dump(clf, "sgd_svm_model.pkl")
joblib.dump(pca, "pca_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model, PCA, Scaler đã được lưu!")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SGDClassifier (SVM)")
plt.savefig("sgd_svm_cm.png")
# 🔹 Save summary table for comparison
summary_file = "model_summary.csv"
summary_data = {
    "model": ["SGD-SVM"],
    "accuracy": [acc],
    "n_pca_components": [pca_components],
    "train_samples": [len(y_train)],
    "test_samples": [len(y_test)]
}

df_summary = pd.DataFrame(summary_data)

# Nếu file đã tồn tại thì append, ngược lại tạo mới
if os.path.exists(summary_file):
    old_df = pd.read_csv(summary_file)
    df_summary = pd.concat([old_df, df_summary], ignore_index=True)

df_summary.to_csv(summary_file, index=False)
print(f"📊 Summary saved to {summary_file}")
plt.show()
