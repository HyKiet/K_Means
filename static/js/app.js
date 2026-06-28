// ============================================================
// app.js - Gọi API từ Flask Backend và vẽ giao diện (Dashboard)
// ============================================================

// Hàm tiện ích: Lấy thẻ HTML theo ID (viết tắt giống jQuery)
const $ = (id) => document.getElementById(id);

// Hàm tiện ích: Gọi API đến server và tự động chuyển kết quả về dạng JSON
const api = (url, opt) => fetch(url, opt).then((r) => r.json());

let clusters = [];

// --- 1. Gọi API lấy thông tin 4 cụm -> Render ra các thẻ thống kê ---
async function loadClusters() {
  // Đợi kết quả từ API /api/clusters
  const data = (await api("/api/clusters")).clusters;
  
  // Sắp xếp các cụm từ: Xuất sắc -> Giỏi -> Trung bình -> Yếu (Dựa vào GPA giảm dần)
  clusters = data.sort((a, b) => b.centroid.gpa - a.centroid.gpa);
  
  // Tạo mã HTML động cho từng thẻ thống kê và nhét vào thẻ div có id="stats"
  $("stats").innerHTML = clusters.map((c) => `
    <div class="stat">
      <div class="top">
        <span class="name">${c.label}</span>
        <span class="dot" style="background:${c.color}"></span>
      </div>
      <div class="num" id="count-${c.id}">–<span> SV</span></div>
      <div class="sub">Học ~${c.centroid.study_hours}h · vắng ~${c.centroid.absences} · GPA ~${c.centroid.gpa}</div>
    </div>`).join("");
}

// --- 2. Dataset -> bang + bieu do + dem so luong ---
async function loadDataset() {
  const res = await api("/api/dataset");
  $("total").textContent = res.total;

  $("table-body").innerHTML = res.data.map((r) => `
    <tr>
      <td>${r.StudentID}</td>
      <td>${r.StudyHoursPerWeek}</td>
      <td>${r.Absences}</td>
      <td>${r.GPA}</td>
      <td><span class="tag" style="background:${r.Color}">${r.Label}</span></td>
    </tr>`).join("");

  const counts = {};
  res.data.forEach((r) => (counts[r.Cluster] = (counts[r.Cluster] || 0) + 1));
  clusters.forEach((c) => ($(`count-${c.id}`).innerHTML = `${counts[c.id] || 0}<span> SV</span>`));

  drawCharts(res.data);
}

// --- 3. Bieu do phan tan (Chart.js, theme sang, theo dung thu tu cum) ---
const AXIS = "#6b7280", GRID = "#eceff3";
function scatterData(data, xKey) {
  return clusters.map((c) => ({
    label: c.label, backgroundColor: c.color, pointRadius: 4, pointHoverRadius: 6,
    data: data.filter((r) => r.Cluster === c.id).map((r) => ({ x: r[xKey], y: r.GPA })),
  }));
}
function makeChart(id, data, xKey, xTitle) {
  new Chart($(id), {
    type: "scatter",
    data: { datasets: scatterData(data, xKey) },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: AXIS, boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: xTitle, color: AXIS }, ticks: { color: AXIS }, grid: { color: GRID } },
        y: { title: { display: true, text: "GPA", color: AXIS }, ticks: { color: AXIS }, grid: { color: GRID } },
      },
    },
  });
}
function drawCharts(data) {
  makeChart("chart1", data, "StudyHoursPerWeek", "Giờ học / tuần");
  makeChart("chart2", data, "Absences", "Số buổi vắng");
}

// --- 4. Gửi dữ liệu Dự đoán sinh viên mới ---
async function predict() {
  // Lấy dữ liệu người dùng nhập từ 3 ô input
  const body = {
    study_hours: parseFloat($("study").value),
    absences: parseFloat($("absence").value),
    gpa: parseFloat($("gpa").value),
  };
  
  // Kiểm tra nếu người dùng để trống bất kỳ ô nào
  if ([body.study_hours, body.absences, body.gpa].some(isNaN))
    return alert("Vui lòng nhập đủ 3 giá trị.");

  // Gọi POST request lên Flask API (/api/predict) kèm theo dữ liệu JSON
  const res = await api("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  
  // Xử lý hiển thị kết quả
  const box = $("result");
  box.classList.add("show");
  
  // Nếu API báo lỗi (VD: nhập GPA = 10 -> lỗi vì vượt qua hệ số 4.0)
  if (!res.success) {
    $("r-emoji").textContent = "⚠️";
    $("r-label").textContent = "Dữ liệu chưa hợp lệ";
    $("r-label").style.color = "#b91c1c";
    $("r-desc").textContent = res.error;
    return;
  }
  
  // Nếu thành công -> Cập nhật giao diện trả về đúng nhóm sinh viên đó
  $("r-emoji").textContent = res.icon;
  $("r-label").textContent = res.label;
  $("r-label").style.color = res.color;
  $("r-desc").textContent = res.description;
}
// Gán sự kiện click cho nút "Phân loại"
$("btn-predict").addEventListener("click", predict);

// --- 5. Toggle sidebar (responsive) ---
const sidebar = $("sidebar"), overlay = $("overlay");
function toggleSidebar() {
  sidebar.classList.toggle("open");
  overlay.classList.toggle("show");
}
$("menu-btn").addEventListener("click", toggleSidebar);
overlay.addEventListener("click", toggleSidebar);
// Bam vao 1 muc menu -> dong sidebar (tren mobile)
document.querySelectorAll(".nav a").forEach((a) =>
  a.addEventListener("click", () => {
    if (sidebar.classList.contains("open")) toggleSidebar();
  })
);

// --- Khoi tao ---
(async function init() {
  await loadClusters();
  await loadDataset();
})();
