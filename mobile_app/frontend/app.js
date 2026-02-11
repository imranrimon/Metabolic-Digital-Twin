// Metabolic Digital Twin - Mobile Frontend Logic

const API_BASE = "http://localhost:8001";

async function updateTwin() {
    const glucose = parseFloat(document.getElementById('input-glucose').value);
    const hba1c = parseFloat(document.getElementById('input-hba1c').value);

    if (isNaN(glucose)) return alert("Please enter current glucose");

    // 1. Predict Risk (Using SOTA Grandmaster Pipeline)
    try {
        const riskRes = await fetch(`${API_BASE}/predict/risk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gender: "Female", // Default/Demo
                age: 45.0,
                hypertension: 0,
                heart_disease: 0,
                smoking_history: "never",
                bmi: 28.5,
                HbA1c_level: hba1c || 6.5,
                blood_glucose_level: parseInt(glucose)
            })
        });
        const riskData = await riskRes.json();

        if (riskData.error) {
            console.error(riskData.error);
            return;
        }

        const prob = (riskData.risk_probability * 100).toFixed(1);
        document.getElementById('risk-val').innerText = prob + "%";
        document.getElementById('risk-status').innerText = riskData.status + " Risk";
        document.getElementById('risk-status').style.color = riskData.status === "High" ? "#ff4d4d" : "#00d2ff";

        // Update subtitle to show model
        if (riskData.model) {
            document.getElementById('risk-status').innerText += ` (${riskData.model})`;
            document.getElementById('risk-status').style.fontSize = "0.8rem";
        }

    } catch (e) { console.error(e); }

    // 2. Get AI Recommendation (Strategy)
    try {
        const dietRes = await fetch(`${API_BASE}/recommend/diet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_glucose: parseFloat(glucose),
                age: 45, // Demo values
                bmi: 28.5,
                gender: 0
            })
        });
        const dietData = await dietRes.json();

        // Update Strategy Title
        document.getElementById('rec-title').innerText = dietData.strategy + " Strategy";
        document.getElementById('rec-desc').innerText = dietData.reason;

    } catch (e) { console.error(e); }

    // 3. Get Detailed Meal Plan (New Endpoint)
    try {
        const mealRes = await fetch(`${API_BASE}/recommend/meals`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_glucose: parseFloat(glucose),
                age: 45,
                bmi: 28.5,
                gender: 0
            })
        });
        const mealData = await mealRes.json();

        // Render Meal Plan
        const planContainer = document.getElementById('meal-plan-container');
        if (!planContainer) {
            const div = document.createElement('div');
            div.id = 'meal-plan-container';
            div.style.marginTop = "1rem";
            document.getElementById('rec-desc').parentNode.appendChild(div);
        }

        const meals = mealData.meals;
        let html = `<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top:10px;">`;

        // Helper to render a card
        const renderCard = (title, items) => {
            let list = items.map(i => `<li style="color:#b0b3b8; font-size:0.8rem;">
                <b>${i.food}</b> <br>
                <span style="font-size:0.75rem; color:#888;">${i.calories} kcal | GI: ${i.gi}</span>
            </li>`).join('');
            return `
            <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:10px;">
                <h4 style="color:#00d2ff; margin-bottom:5px;">${title}</h4>
                <ul style="padding-left:15px; margin:0; line-height:1.4;">${list || '<li style="color:#666">No suggestions</li>'}</ul>
            </div>`;
        };

        html += renderCard('Breakfast', meals.Breakfast);
        html += renderCard('Lunch', meals.Lunch);
        html += renderCard('Dinner', meals.Dinner);
        html += renderCard('Snack', meals.Snack);
        html += `</div> 
        <div style="margin-top:10px; font-size:0.8rem; color:#888; display:flex; justify-content:space-between;">
            <span>Target: ${mealData.caloric_target} kcal</span>
            <span>Macros: C:${mealData.macros.carbs} P:${mealData.macros.protein} F:${mealData.macros.fat}</span>
        </div>`;

        document.getElementById('meal-plan-container').innerHTML = html;

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
