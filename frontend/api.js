const PROD_API_URL = "https://your-backend.onrender.com";

const BASE_URL =
  ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://127.0.0.1:8000"
    : "https://dmw-credit-risk.onrender.com";

async function fetchJSON(path, options = {}) {
    const controller = new AbortController();
    const timeoutMs = options.timeout || 120000; // 120s default
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(`${BASE_URL}${path}`, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === "AbortError") {
            throw new Error("Request timeout. The server may be training the model (can take several minutes on first request).");
        }
        throw error;
    }
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
        timeout: 120000,
    });
}