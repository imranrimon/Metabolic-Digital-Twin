// Metabolic Digital Twin - Mobile Frontend Logic

const API_BASE = "http://localhost:8001";

async function updateTwin() {
    const glucose = parseFloat(document.getElementById('input-glucose').value);
    const hba1c = parseFloat(document.getElementById('input-hba1c').value);

    if (isNaN(glucose)) return alert("Please enter current glucose");

    // 1. Predict Risk
    try {
        const riskRes = await fetch(`${API_BASE}/predict/risk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_features: [45, 28.5, hba1c || 6.5, glucose], // age, bmi, hba1c, g
                cat_features: [1, 3, 0, 0] // gender, smok, hyp, heart
            })
        });
        const riskData = await riskRes.json();
        document.getElementById('risk-val').innerText = (riskData.risk_probability * 100).toFixed(1) + "%";
        document.getElementById('risk-status').innerText = riskData.status + " Risk Profile";
        document.getElementById('risk-status').style.color = riskData.status === "High" ? "#ff4d4d" : "#00d2ff";
    } catch (e) { console.error(e); }

    // 2. Get AI Recommendation
    try {
        const dietRes = await fetch(`${API_BASE}/recommend/diet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_glucose: glucose })
        });
        const dietData = await dietRes.json();
        document.getElementById('rec-title').innerText = dietData.recommendation;
        document.getElementById('rec-desc').innerText = dietData.reason;
    } catch (e) { console.error(e); }

    // 3. Update Chart (Static Mock for Demo logic)
    renderChart(glucose);
}

function renderChart(baseG) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    const labels = ['Now', '15m', '30m', '45m', '60m'];
    const data = [baseG, baseG + 5, baseG + 15, baseG + 8, baseG - 2];

    if (window.myChart) window.myChart.destroy();

    window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Forecast',
                data: data,
                borderColor: '#00d2ff',
                backgroundColor: 'rgba(0, 210, 255, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#b0b3b8' } }
            }
        }
    });
}

// Load Chart.js dynamically
const script = document.createElement('script');
script.src = "https://cdn.jsdelivr.net/npm/chart.js";
script.onload = () => {
    renderChart(110);
};
document.head.appendChild(script);
