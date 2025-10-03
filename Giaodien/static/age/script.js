let selectedFile = null;

// =================== KÉO & THẢ FILE ===================
const uploadArea = document.querySelector(".upload-area");

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    selectedFile = files[0];
    displayOriginalImage(selectedFile);
    document.getElementById("processBtn").disabled = false;
    document.getElementById("fileName").innerText = "Tên file: " + selectedFile.name;
  }
});

// =================== CHỌN FILE ===================
document.getElementById("imageInput").addEventListener("change", function (e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    displayOriginalImage(selectedFile);
    document.getElementById("processBtn").disabled = false;
    document.getElementById("fileName").innerText = "Tên file: " + selectedFile.name;
  }
});

// =================== HIỂN THỊ ẢNH GỐC ===================
function displayOriginalImage(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    document.getElementById("originalImage").src = e.target.result;

    document.querySelector(".upload-area").style.display = "none";
    document.getElementById("previewContainer").style.display = "flex";
    document.getElementById("resetSection").style.display = "block";

    // Reset các phần khác
    document.getElementById("resultImage").style.display = "none";
    document.getElementById("errorMessage").style.display = "none";
    document.getElementById("facesContainer").innerHTML = "";
    document.getElementById("stats").innerHTML = "";
  };
  reader.readAsDataURL(file);
}

// =================== GỬI ẢNH LÊN SERVER ===================
async function processImage() {
  if (!selectedFile) return;

  document.getElementById("loading").style.display = "block";
  document.getElementById("processBtn").disabled = true;

  const startTime = Date.now();

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (data.error) {
      document.getElementById("errorMessage").innerText = data.error;
      document.getElementById("errorMessage").style.display = "block";
      return;
    }

    // Ảnh có bounding box
    document.getElementById("resultImage").src =
      "data:image/jpeg;base64," + data.image_with_box;
    document.getElementById("resultImage").style.display = "block";

    // Hiển thị tất cả khuôn mặt crop
    const facesContainer = document.getElementById("facesContainer");
    facesContainer.innerHTML = "";
    data.results.forEach((face, i) => {
      const div = document.createElement("div");
      div.classList.add("face-card");
      div.innerHTML = `
        <img src="data:image/jpeg;base64,${face.cropped_face}" alt="face ${
        i + 1
      }" />
        <p>${face.gender}, ${face.age} tuổi</p>
        <small>Độ tin cậy: ${face.confidence}%</small>
      `;
      facesContainer.appendChild(div);
    });

    // Thêm stats
    const processTime = Date.now() - startTime;
    document.getElementById("stats").innerHTML = `
      <div class="stat-card"><div class="stat-number">${
        data.results.length
      }</div><div class="stat-label">Số khuôn mặt</div></div>
      <div class="stat-card"><div class="stat-number">${processTime}ms</div><div class="stat-label">Thời gian xử lý</div></div>
    `;

    document.getElementById("resultsSection").style.display = "block";
  } catch (err) {
    document.getElementById("errorMessage").innerText =
      "Lỗi khi xử lý ảnh: " + err.message;
    document.getElementById("errorMessage").style.display = "block";
  } finally {
    document.getElementById("loading").style.display = "none";
    document.getElementById("processBtn").disabled = false;
  }
}

// =================== RESET UPLOAD ===================
function resetUpload() {
  selectedFile = null;
  document.getElementById("imageInput").value = "";

  document.getElementById("previewContainer").style.display = "none";
  document.getElementById("resetSection").style.display = "none";
  document.querySelector(".upload-area").style.display = "block";

  document.getElementById("originalImage").src = "";
  document.getElementById("resultImage").src = "";
  document.getElementById("fileName").innerText = "";
  document.getElementById("facesContainer").innerHTML = "";
  document.getElementById("stats").innerHTML = "";
  document.getElementById("errorMessage").innerText = "";
  document.getElementById("resultsSection").style.display = "none";

  document.getElementById("processBtn").disabled = true;
}

// =================== TOGGLE WEBCAM ===================
function toggleWebcam() {
  const section = document.getElementById("webcam_section");
  const feed = document.getElementById("webcam_feed");

  const uploadArea = document.querySelector(".upload-area");
  const previewContainer = document.getElementById("previewContainer");
  const resetSection = document.getElementById("resetSection");
  const resultsSection = document.getElementById("resultsSection");
  const facesContainer = document.getElementById("facesContainer");

  if (section.style.display === "none" || section.style.display === "") {
    // Bật webcam
    feed.src = "/video_feed";
    section.style.display = "block";
    document.getElementById("toggleBtn").innerText = "❌ Tắt Webcam";

    // Ẩn toàn bộ phần dự đoán bằng ảnh
    uploadArea.style.display = "none";
    previewContainer.style.display = "none";
    resetSection.style.display = "none";
    resultsSection.style.display = "none";
    facesContainer.style.display = "none";
  } else {
    // Tắt webcam
    feed.src = "";
    section.style.display = "none";
    document.getElementById("toggleBtn").innerText = "🎥 Bật Webcam";

    // Hiện lại upload
    uploadArea.style.display = "block";
    facesContainer.style.display = "flex"; // nếu có ảnh crop thì hiện lại
  }
}
