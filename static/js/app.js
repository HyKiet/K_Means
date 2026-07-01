// ============================================================
// app.js - Xử lý tải file CSV, gọi Flask API và vẽ biểu đồ động
// ============================================================

// Hàm tiện ích: Lấy thẻ HTML theo ID
const $ = (id) => document.getElementById(id);

// Hàm tiện ích: Gọi API đến server
const api = (url, opt) => fetch(url, opt).then((r) => r.json());

let clusters = [];
let chart1Instance = null;
let chart2Instance = null;
let selectedFile = null;

// --- 1. Vẽ biểu đồ phân tán (Chart.js, quản lý hủy instance tránh lỗi đè biểu đồ) ---
const AXIS = "#6b7280", GRID = "#eceff3";

function scatterData(data, xKey) {
  return clusters.map((c) => ({
    label: c.label,
    backgroundColor: c.color,
    pointRadius: 4,
    pointHoverRadius: 6,
    data: data.filter((r) => r.Cluster === c.id).map((r) => ({ x: r[xKey], y: r.GPA })),
  }));
}

function makeChart(id, data, xKey, xTitle) {
  if (id === "chart1" && chart1Instance) chart1Instance.destroy();
  if (id === "chart2" && chart2Instance) chart2Instance.destroy();

  const chart = new Chart($(id), {
    type: "scatter",
    data: { datasets: scatterData(data, xKey) },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: AXIS, boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: xTitle, color: AXIS }, ticks: { color: AXIS }, grid: { color: GRID } },
        y: { title: { display: true, text: "GPA", color: AXIS }, ticks: { color: AXIS }, grid: { color: GRID } },
      },
    },
  });

  if (id === "chart1") chart1Instance = chart;
  if (id === "chart2") chart2Instance = chart;
}

function drawCharts(data) {
  makeChart("chart1", data, "StudyHoursPerWeek", "Giờ học / tuần");
  makeChart("chart2", data, "Absences", "Số buổi vắng");
}

// --- 2. Hàm cập nhật toàn bộ Dashboard khi có dữ liệu mới ---
function updateDashboard(res) {
  // Hiển thị các panel kết quả, ẩn placeholder
  $("stats").style.display = "grid";
  $("charts-panel").style.display = "block";
  $("dulieu").style.display = "block";
  $("placeholder-panel").style.display = "none";

  // Cập nhật tổng số lượng sinh viên
  $("total").textContent = res.total;

  // Sắp xếp các cụm từ: Xuất sắc -> Giỏi -> Trung bình -> Yếu (Theo GPA centroid giảm dần)
  clusters = res.clusters.sort((a, b) => b.centroid.gpa - a.centroid.gpa);

  // Cập nhật các thẻ thống kê KPI
  $("stats").innerHTML = clusters.map((c) => `
    <div class="stat">
      <div class="top">
        <span class="name">${c.label}</span>
        <span class="dot" style="background:${c.color}"></span>
      </div>
      <div class="num">${c.count}<span> SV</span></div>
      <div class="sub">Học ~${c.centroid.study_hours}h · vắng ~${c.centroid.absences} · GPA ~${c.centroid.gpa}</div>
    </div>`).join("");

  // Cập nhật bảng dữ liệu sinh viên
  $("table-body").innerHTML = res.data.map((r) => `
    <tr>
      <td>${r.StudentID}</td>
      <td>${r.StudyHoursPerWeek}</td>
      <td>${r.Absences}</td>
      <td>${r.GPA}</td>
      <td><span class="tag" style="background:${r.Color}">${r.Label}</span></td>
    </tr>`).join("");

  // Vẽ lại biểu đồ phân cụm
  drawCharts(res.data);
}

// --- 3. Xử lý Drag & Drop và Tải file CSV ---
const zone = $("upload-zone");
const fileInput = $("file-input");
const btnUpload = $("btn-upload");
const btnClear = $("btn-clear");
const fileName = $("file-name");

// Click vào vùng kéo thả để chọn file
zone.addEventListener("click", () => fileInput.click());

// Kéo file vào vùng thả
zone.addEventListener("dragover", (e) => {
  e.preventDefault();
  zone.classList.add("dragover");
});

zone.addEventListener("dragleave", () => {
  zone.classList.remove("dragover");
});

zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

// Khi người dùng chọn file từ cửa sổ file
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Vui lòng chỉ tải lên tệp tin định dạng .csv");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  btnClear.style.display = "block";
  btnUpload.removeAttribute("disabled");
}

// Bấm nút Xóa file để phục hồi trạng thái cũ (Trống trơn)
btnClear.addEventListener("click", (e) => {
  e.stopPropagation(); // Tránh click kích hoạt mở file dialog
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  fileName.textContent = "Chưa chọn tệp";
  btnClear.style.display = "none";
  btnUpload.setAttribute("disabled", "true");

  // Ẩn tất cả kết quả, hiện lại màn hình placeholder
  $("stats").style.display = "none";
  $("charts-panel").style.display = "none";
  $("dulieu").style.display = "none";
  $("placeholder-panel").style.display = "flex";
}

// Gửi file CSV lên backend phân tích
btnUpload.addEventListener("click", async () => {
  if (!selectedFile) return;

  btnUpload.textContent = "Đang phân tích...";
  btnUpload.setAttribute("disabled", "true");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    }).then(r => r.json());

    if (res.success) {
      updateDashboard(res);
    } else {
      alert("Lỗi: " + res.error);
    }
  } catch (err) {
    alert("Đã xảy ra lỗi khi gửi yêu cầu lên máy chủ.");
    console.error(err);
  } finally {
    btnUpload.textContent = "Phân tích dữ liệu";
    btnUpload.removeAttribute("disabled");
  }
});

// --- 4. Toggle sidebar (responsive) ---
const sidebar = $("sidebar"), overlay = $("overlay");
function toggleSidebar() {
  sidebar.classList.toggle("open");
  overlay.classList.toggle("show");
}
$("menu-btn").addEventListener("click", toggleSidebar);
overlay.addEventListener("click", toggleSidebar);

document.querySelectorAll(".nav a").forEach((a) =>
  a.addEventListener("click", () => {
    if (sidebar.classList.contains("open")) toggleSidebar();
  })
);

// --- Khởi tạo ban đầu (Mặc định trống trơn) ---
(function init() {
  resetUpload();
})();
