let selectedFile = null;

// ===== KÉO & THẢ =====
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

// ===== CHỌN FILE =====
document.getElementById("imageInput").addEventListener("change", function (e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    displayOriginalImage(selectedFile);
    document.getElementById("processBtn").disabled = false;
    document.getElementById("fileNamePreview").innerText = selectedFile.name;

  }
});

// ===== HIỂN THỊ ẢNH GỐC =====
function displayOriginalImage(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    document.getElementById("originalImage").src = e.target.result;
    document.getElementById("previewContainer").style.display = "flex";
    document.getElementById("resetSection").style.display = "block";
    document.getElementById("origResult").style.display = "block";

    // Reset kết quả cũ
    document.getElementById("resultText").innerText = "";
    document.getElementById("errorMessage").style.display = "none";
    document.getElementById("deepResult").style.display = "none";
    document.getElementById("tradResult").style.display = "none";
  };
  reader.readAsDataURL(file);
}


// ===== GỬI ẢNH LÊN SERVER =====
async function processImage() {
  if (!selectedFile) return;

  document.getElementById("loading").style.display = "block";
  document.getElementById("processBtn").disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict_emotion", { method: "POST", body: formData });
    const data = await response.json();

    // ==== Kiểm tra lỗi từ server ====
    if (!data || data.error) {
      document.getElementById("errorMessage").innerText = data?.error || "❌ Lỗi không xác định.";
      document.getElementById("errorMessage").style.display = "block";
      return;
    }

    // ==== Hiển thị ảnh trả về từ server ====
    if (data.original_image) {
      document.getElementById("originalImage").src = "data:image/jpeg;base64," + data.original_image;
    }

    if (data.deep_image) {
      document.getElementById("deepImage").src = "data:image/jpeg;base64," + data.deep_image;
      document.getElementById("deepResult").style.display = "block";
    }

    if (data.trad_image) {
      document.getElementById("tradImage").src = "data:image/jpeg;base64," + data.trad_image;
      document.getElementById("tradResult").style.display = "block";
    }

    document.getElementById("previewContainer").style.display = "flex";

    // ==== Duyệt từng khuôn mặt để lấy kết quả ====
    if (data.results && data.results.length > 0) {
      let list = "";
      data.results.forEach((r, idx) => {
        list += `<p><strong>Face ${idx + 1}</strong><br>
                   Deep → ${r.deep.emotion} (${r.deep.confidence}%)<br>
                   Truyền thống → ${r.trad.emotion}
                 </p>`;
      });
      document.getElementById("resultText").innerHTML = list;
    }

  } catch (err) {
    document.getElementById("errorMessage").innerText = "Lỗi khi xử lý ảnh: " + err.message;
    document.getElementById("errorMessage").style.display = "block";
  } finally {
    document.getElementById("loading").style.display = "none";
    document.getElementById("processBtn").disabled = false;
  }
}


// ===== RESET =====
function resetUpload() {
  selectedFile = null;
  document.getElementById("imageInput").value = "";
  document.getElementById("previewContainer").style.display = "none";
  document.getElementById("resetSection").style.display = "none";
  document.getElementById("originalImage").src = "";
  document.getElementById("fileName").innerText = "";
  document.getElementById("resultText").innerText = "";
  document.getElementById("errorMessage").innerText = "";
  document.getElementById("errorMessage").style.display = "none";
  document.getElementById("processBtn").disabled = true;
}
