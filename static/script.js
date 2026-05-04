/* ── State ─────────────────────────────────────────── */
const API = '';  // same-origin
let historyData = [];
let activeFilter = 'All';
let donutChart, trendChart, barChart;

const EXAMPLES = {
    hot: {
        name: 'Sarah Connor', company: 'Cyberdyne Corp', source: 'email',
        text: 'Our board has approved the budget. We are ready to sign the contract today and need to move forward ASAP. Please send the paperwork and invoice immediately so we can close this deal before end of quarter.',
    },
    warm: {
        name: 'Miles Dyson', company: 'Neural Corp', source: 'crm',
        text: 'Hi, I came across your platform at the conference last week and I\'m very interested in learning more. Could you send me detailed pricing information for the enterprise plan and schedule a demo for our team next week?',
    },
    cold: {
        name: 'John Unknown', company: 'N/A', source: 'email',
        text: 'Please remove me from your mailing list. I am not interested in your product and do not wish to be contacted again. Unsubscribe immediately.',
    },
};

const VERDICT_META = {
    Hot: { emoji: '🔥', sub: 'High purchase intent — prioritize immediately', cls: 'hot' },
    Warm: { emoji: '✨', sub: 'Engaged and interested — nurture actively', cls: 'warm' },
    Cold: { emoji: '❄️', sub: 'Low intent detected — place in long-term queue', cls: 'cold' },
};

/* ── Init ──────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
    await checkStatus();
    await loadHistory();
    initCharts();
    await refreshAnalytics();

    document.getElementById('leadText').addEventListener('input', e => {
        document.getElementById('charCount').textContent = e.target.value.length;
    });
});

/* ── Status ────────────────────────────────────────── */
async function checkStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const txt = document.getElementById('statusText');
        dot.classList.add('active');
        txt.textContent = data.model_ready
            ? `MiniLM Ready`
            : `Rule Engine Active`;
    } catch {
        document.getElementById('statusText').textContent = 'Offline';
    }
}

/* ── Classify ──────────────────────────────────────── */
async function classifyLead() {
    const text = document.getElementById('leadText').value.trim();
    if (!text) { showToast('Please enter a lead description.', 'error'); return; }

    setLoading(true);

    try {
        const payload = {
            text,
            lead_name: document.getElementById('leadName').value.trim(),
            company: document.getElementById('leadCompany').value.trim(),
            source: document.getElementById('leadSource').value,
        };

        const res = await fetch(`${API}/api/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Classification failed');
        }

        const data = await res.json();
        renderResult(data);
        await loadHistory();
        await refreshAnalytics();
        showToast(`Lead classified as ${data.label} (${(data.confidence * 100).toFixed(0)}% confidence)`, 'success');

    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        setLoading(false);
    }
}

/* ── Render Result ─────────────────────────────────── */
function renderResult(data) {
    const label = data.label;
    const meta = VERDICT_META[label] || VERDICT_META.Warm;
    const probs = data.probabilities || {};
    const conf = data.confidence || 0;

    // Show result, hide empty
    document.getElementById('resultEmpty').classList.add('hidden');
    const rc = document.getElementById('resultContent');
    rc.classList.add('visible');

    // Verdict
    const vb = document.getElementById('verdictBlock');
    vb.className = `verdict ${meta.cls} fade-in`;
    document.getElementById('verdictEmoji').textContent = meta.emoji;
    document.getElementById('verdictLabel').textContent = label;
    document.getElementById('verdictSub').textContent = meta.sub;

    // Engine badge
    const eb = document.getElementById('engineBadge');
    eb.style.display = 'inline-flex';
    eb.textContent = `⚙ ${data.engine || 'rule_based'}`;

    // Confidence
    document.getElementById('confValue').textContent = `${(conf * 100).toFixed(1)}%`;
    const fill = document.getElementById('confFill');
    fill.className = `conf-fill ${meta.cls}`;
    setTimeout(() => { fill.style.width = `${conf * 100}%`; }, 50);

    // Probabilities
    const setProb = (id, pctId, cls, val) => {
        const pct = (val || 0) * 100;
        document.getElementById(pctId).textContent = `${pct.toFixed(1)}%`;
        const bar = document.getElementById(id);
        bar.className = `prob-bar ${cls}`;
        setTimeout(() => { bar.style.width = `${pct}%`; }, 80);
    };
    setProb('probHot', 'probHotPct', 'hot', probs.Hot);
    setProb('probWarm', 'probWarmPct', 'warm', probs.Warm);
    setProb('probCold', 'probColdPct', 'cold', probs.Cold);

    // Signals
    const pills = document.getElementById('signalPills');
    const rules = data.applied_rules || [];
    if (rules.length) {
        pills.innerHTML = rules.map(r => `<span class="signal-pill">${r}</span>`).join('');
        document.getElementById('signalsBlock').style.display = 'block';
    } else {
        document.getElementById('signalsBlock').style.display = 'none';
    }

    // RAG
    const ragBlock = document.getElementById('ragBlock');
    if (data.rag_enabled && data.retrieved_context) {
        ragBlock.style.display = 'block';
        document.getElementById('ragSummary').textContent = data.intelligence_summary || 'Retrieved Business Context:';
        document.getElementById('ragContent').textContent = data.retrieved_context;
    } else {
        ragBlock.style.display = 'none';
    }

    // Meta
    document.getElementById('metaName').textContent = data.lead_name || '—';
    document.getElementById('metaCompany').textContent = data.company || '—';
    document.getElementById('metaSource').textContent = data.source || '—';
    document.getElementById('metaTime').textContent = formatTime(data.timestamp);
}

/* ── History ───────────────────────────────────────── */
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/history`);
        const data = await res.json();
        historyData = data.history || [];
        document.getElementById('historyCount').textContent = `${historyData.length} records`;
        renderHistory(historyData);
    } catch { }
}

function renderHistory(items) {
    const list = document.getElementById('historyList');
    const filtered = activeFilter === 'All'
        ? items
        : items.filter(i => i.label === activeFilter);

    if (!filtered.length) {
        list.innerHTML = '<div class="history-empty">No leads in this category.</div>';
        return;
    }

    list.innerHTML = filtered.map(item => `
    <div class="history-item ${item.label}" onclick="showHistoryDetail('${item.record_id}')">
      <div class="history-item-header">
        <span class="history-tag ${item.label}">${item.label}</span>
        <span class="history-conf">${(item.confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="history-text">${escHtml(item.text)}</div>
      ${item.lead_name ? `<div class="history-time">${escHtml(item.lead_name)} · ${item.company || '—'}</div>` : ''}
      <div class="history-time">${formatTime(item.timestamp)} · via ${item.source || '—'}</div>
    </div>
  `).join('');

    // Update stat cards
    const counts = { Hot: 0, Warm: 0, Cold: 0 };
    items.forEach(i => { if (counts[i.label] !== undefined) counts[i.label]++; });
    document.getElementById('statTotal').textContent = items.length;
    document.getElementById('statHot').textContent = counts.Hot;
    document.getElementById('statWarm').textContent = counts.Warm;
    document.getElementById('statCold').textContent = counts.Cold;
}

function filterHistory(label, btn) {
    activeFilter = label;
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    renderHistory(historyData);
}

function showHistoryDetail(recordId) {
    const item = historyData.find(i => i.record_id === recordId);
    if (item) renderResult(item);
}

/* ── Analytics / Charts ─────────────────────────────── */
function initCharts() {
    const chartDefaults = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        animation: { duration: 700 },
    };

    const gridColor = 'rgba(0,240,200,0.06)';
    const tickColor = '#3d5270';

    // Donut
    donutChart = new Chart(document.getElementById('donutChart'), {
        type: 'doughnut',
        data: {
            labels: ['Hot', 'Warm', 'Cold'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#ff4d6d', '#ffb700', '#3d9eff'],
                borderColor: '#060a12',
                borderWidth: 3,
                hoverOffset: 8,
            }],
        },
        options: {
            ...chartDefaults,
            cutout: '68%',
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: { color: '#7a90b0', font: { family: 'JetBrains Mono', size: 11 }, padding: 12, boxWidth: 10 },
                },
            },
        },
    });

    // Trend (line)
    trendChart = new Chart(document.getElementById('trendChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Confidence',
                data: [],
                borderColor: '#00f0c8',
                backgroundColor: 'rgba(0,240,200,0.06)',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '#00f0c8',
                fill: true,
                tension: 0.4,
            }],
        },
        options: {
            ...chartDefaults,
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'JetBrains Mono', size: 10 } } },
                y: {
                    min: 0, max: 1,
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor,
                        font: { family: 'JetBrains Mono', size: 10 },
                        callback: v => `${(v * 100).toFixed(0)}%`,
                    },
                },
            },
        },
    });

    // Bar
    barChart = new Chart(document.getElementById('barChart'), {
        type: 'bar',
        data: {
            labels: ['Hot', 'Warm', 'Cold'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['rgba(255,77,109,0.7)', 'rgba(255,183,0,0.7)', 'rgba(61,158,255,0.7)'],
                borderColor: ['#ff4d6d', '#ffb700', '#3d9eff'],
                borderWidth: 2,
                borderRadius: 4,
            }],
        },
        options: {
            ...chartDefaults,
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'JetBrains Mono', size: 11 } } },
                y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'JetBrains Mono', size: 10 } } },
            },
        },
    });
}

async function refreshAnalytics() {
    try {
        const res = await fetch(`${API}/api/analytics`);
        const data = await res.json();
        const s = data.session || {};

        // Donut
        donutChart.data.datasets[0].data = [s.Hot || 0, s.Warm || 0, s.Cold || 0];
        donutChart.update();

        // Bar
        barChart.data.datasets[0].data = [s.Hot || 0, s.Warm || 0, s.Cold || 0];
        barChart.update();

        // Trend
        const trend = s.trend || [];
        trendChart.data.labels = trend.map((_, i) => `#${i + 1}`);
        trendChart.data.datasets[0].data = trend.map(t => t.confidence);
        trendChart.update();

    } catch { }
}

/* ── Helpers ───────────────────────────────────────── */
function setLoading(on) {
    const btn = document.getElementById('classifyBtn');
    const label = document.getElementById('btnLabel');
    const spinner = document.getElementById('btnSpinner');
    btn.disabled = on;
    label.style.display = on ? 'none' : 'flex';
    spinner.style.display = on ? 'block' : 'none';
}

function loadExample(type) {
    const ex = EXAMPLES[type];
    if (!ex) return;
    document.getElementById('leadName').value = ex.name;
    document.getElementById('leadCompany').value = ex.company;
    document.getElementById('leadSource').value = ex.source;
    document.getElementById('leadText').value = ex.text;
    document.getElementById('charCount').textContent = ex.text.length;
}

async function clearSession() {
    await fetch(`${API}/api/session/clear`, { method: 'POST' });
    historyData = [];
    document.getElementById('historyList').innerHTML = '<div class="history-empty">Session cleared.</div>';
    document.getElementById('historyCount').textContent = '0 records';
    ['statTotal', 'statHot', 'statWarm', 'statCold'].forEach(id => document.getElementById(id).textContent = '0');
    document.getElementById('resultEmpty').classList.remove('hidden');
    document.getElementById('resultContent').classList.remove('visible');
    await refreshAnalytics();
    showToast('Session cleared.', 'success');
}

function formatTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function escHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function showToast(msg, type = 'success') {
    const c = document.getElementById('toastContainer');
    const div = document.createElement('div');
    div.className = `toast ${type}`;
    div.textContent = msg;
    c.appendChild(div);
    setTimeout(() => { div.style.opacity = '0'; div.style.transition = 'opacity 0.4s'; setTimeout(() => div.remove(), 400); }, 3000);
}
function toggleTheme() {
    const body = document.body;
    const btn = document.getElementById("themeBtn");

    body.classList.toggle("light-mode");

    if (body.classList.contains("light-mode")) {
        localStorage.setItem("theme", "light");
        btn.innerText = "☀️ Light";
    } else {
        localStorage.setItem("theme", "dark");
        btn.innerText = "🌙 Dark";
    }
}

