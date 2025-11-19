// frontend/script.js
const API_BASE = "http://localhost:5000";

async function fetchHealth() {
  const el = document.getElementById("health");
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = `서버 상태를 가져오지 못했습니다: ${e}`;
  }
}

async function predict(features) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

function setupForm() {
  const form = document.getElementById("predict-form");
  const resultEl = document.getElementById("result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultEl.textContent = "예측 중...";

    const sepal_length = parseFloat(document.getElementById("sepal_length").value);
    const sepal_width  = parseFloat(document.getElementById("sepal_width").value);
    const petal_length = parseFloat(document.getElementById("petal_length").value);
    const petal_width  = parseFloat(document.getElementById("petal_width").value);

    const features = [sepal_length, sepal_width, petal_length, petal_width];

    // 간단한 입력 검증
    if (features.some(v => Number.isNaN(v))) {
      resultEl.textContent = "모든 값을 올바르게 입력해주세요.";
      return;
    }

    try {
      const data = await predict(features);
      resultEl.textContent =
        `예측 품종: ${data.prediction}\n\n` +
        `클래스 확률:\n` +
        Object.entries(data.probabilities)
          .map(([k, v]) => `- ${k}: ${v}`)
          .join("\n");
    } catch (e) {
      resultEl.textContent = `에러: ${e.message}`;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupForm();
  fetchHealth();
});
