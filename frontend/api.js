const PROD_API_URL = "https://your-backend.onrender.com";

const BASE_URL =
  ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://127.0.0.1:8000"
    : "https://dmw-credit-risk.onrender.com";

async function fetchJSON(path, options = {}) {
    const response = await fetch(`${BASE_URL}${path}`, options);
    if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
    }
    return response.json();
}

export async function getHealthStatus() {
    return fetchJSON("/");
}

export async function getDashboardMetrics() {
    return fetchJSON("/dashboard/metrics");
}

export async function getLifecycleSummary() {
    return fetchJSON("/lifecycle/summary");
}

export async function getEDA() {
    return fetchJSON("/eda");
}

export async function getModelEvaluation() {
    return fetchJSON("/model/evaluation");
}

export async function getWarehouse() {
    return fetchJSON("/warehouse");
}

export async function predictAPI(payload) {
    return fetchJSON("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });
}