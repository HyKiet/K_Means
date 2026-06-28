// ============================================================
// app.js — Dashboard Logic
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadClusters();
    loadDataset();
});

// ── Predict Student ──
async function predictStudent() {
    const btn = document.getElementById('btn-predict');
    const spinner = document.getElementById('spinner');
    const btnText = document.getElementById('btn-text');

    const studyHours = parseFloat(document.getElementById('study-hours').value);
    const absences = parseFloat(document.getElementById('absences').value);
    const gpa = parseFloat(document.getElementById('gpa').value);

    if (isNaN(studyHours) || isNaN(absences) || isNaN(gpa)) {
        showAlert('Vui lòng nhập đầy đủ thông tin!', 'error');
        return;
    }

    btn.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Đang phân tích...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ study_hours: studyHours, absences, gpa })
        });
        const data = await res.json();

        if (data.success) {
            displayResult(data);
        } else {
            showAlert(data.error || 'Lỗi server!', 'error');
        }
    } catch (e) {
        showAlert('Không thể kết nối API!', 'error');
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Phân Loại Ngay';
    }
}

// ── Display Result ──
function displayResult(data) {
    const box = document.getElementById('result-container');
    const label = document.getElementById('result-label');
    const desc = document.getElementById('result-desc');
    const icon = document.getElementById('result-icon');

    const emojis = { 'Xuat sac': '🏆', 'Gioi': '⭐', 'Trung binh': '📊', 'Yeu': '⚠️' };
    
    label.textContent = data.label;
    label.style.color = data.color;
    desc.textContent = data.description;
    icon.textContent = emojis[data.label] || '🎓';

    box.style.borderColor = data.color;
    box.classList.add('show');
}

// ── Load Clusters ──
async function loadClusters() {
    try {
        const res = await fetch('/api/clusters');
        const data = await res.json();
        if (data.success) renderClusters(data.clusters);
    } catch (e) {
        console.error('Load clusters err:', e);
    }
}

function renderClusters(clusters) {
    const container = document.getElementById('clusters-container');
    if (!container) return;

    const emojis = { 'Xuat sac': '🏆', 'Gioi': '⭐', 'Trung binh': '📊', 'Yeu': '⚠️' };
    const sorted = [...clusters].sort((a, b) => b.centroid.gpa - a.centroid.gpa);

    container.innerHTML = sorted.map(c => `
        <div class="summary-card" style="border-left-color: ${c.color}">
            <div class="summary-header">
                <div class="summary-title" style="color: ${c.color}">${c.label}</div>
                <div class="summary-icon">${emojis[c.label] || '📊'}</div>
            </div>
            <div class="summary-desc">${c.description}</div>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-val">${c.centroid.study_hours}h</div>
                    <div class="stat-lbl">Giờ học</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">${c.centroid.absences}</div>
                    <div class="stat-lbl">Vắng</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">${c.centroid.gpa}</div>
                    <div class="stat-lbl">GPA</div>
                </div>
            </div>
        </div>
    `).join('');
}

// ── Load Charts ──
async function loadDataset() {
    try {
        const res = await fetch('/api/dataset');
        const data = await res.json();
        if (data.success) {
            renderCharts(data.data);
        }
    } catch (e) {
        console.error('Load dataset err:', e);
    }
}

function renderCharts(students) {
    const groups = {};
    students.forEach(s => {
        if (!groups[s.Label]) groups[s.Label] = { data1: [], data2: [], color: s.Color };
        groups[s.Label].data1.push({ x: s.StudyHoursPerWeek, y: s.GPA });
        groups[s.Label].data2.push({ x: s.Absences, y: s.GPA });
    });

    const chartOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 10, padding: 15, font: { family: 'Inter', size: 11 } } },
            tooltip: {
                backgroundColor: '#1f2937', titleColor: '#f9fafb', bodyColor: '#d1d5db',
                cornerRadius: 6, padding: 10, bodyFont: { family: 'Inter', size: 11 }
            }
        },
        scales: {
            x: { grid: { color: '#f3f4f6' }, ticks: { color: '#6b7280', font: { size: 10 } } },
            y: { grid: { color: '#f3f4f6' }, ticks: { color: '#6b7280', font: { size: 10 } }, min: 0, max: 4.2 }
        }
    };

    const ctx1 = document.getElementById('scatter-chart');
    if (ctx1) {
        new Chart(ctx1, {
            type: 'scatter',
            data: { datasets: Object.entries(groups).map(([label, g]) => ({
                label, data: g.data1, backgroundColor: g.color + '90', borderColor: g.color,
                borderWidth: 1, pointRadius: 5, pointHoverRadius: 7
            })) },
            options: chartOpts
        });
    }

    const ctx2 = document.getElementById('scatter-chart-2');
    if (ctx2) {
        new Chart(ctx2, {
            type: 'scatter',
            data: { datasets: Object.entries(groups).map(([label, g]) => ({
                label, data: g.data2, backgroundColor: g.color + '90', borderColor: g.color,
                borderWidth: 1, pointRadius: 5, pointHoverRadius: 7
            })) },
            options: chartOpts
        });
    }
}

// ── Utils ──
function showAlert(msg, type = 'success') {
    const container = document.getElementById('alert-container');
    if (!container) return;
    
    const el = document.createElement('div');
    el.className = `alert ${type}`;
    el.textContent = msg;
    
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}


