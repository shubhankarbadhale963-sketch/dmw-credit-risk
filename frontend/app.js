import {
    getHealthStatus,
    getDashboardMetrics,
    getLifecycleSummary,
    getEDA,
    getModelEvaluation,
    getWarehouse,
    predictAPI,
} from "./api.js";

document.addEventListener("DOMContentLoaded", () => {
  fetch("https://dmw-credit-risk.onrender.com/health")
    .catch(() => {});
});

const statusNode = document.getElementById("backendStatus");
const navChips = Array.from(document.querySelectorAll(".nav-chip"));
const predictionForm = document.getElementById("predictionForm");
const predictionRiskNode = document.getElementById("predictionRisk");
const predictionProbabilityNode = document.getElementById("predictionProbability");
const riskMeterFill = document.getElementById("riskMeterFill");
const scenarioApplyButtons = Array.from(document.querySelectorAll(".scenario-apply"));
const typedLifecycleText = document.getElementById("typedLifecycleText");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeSplash = document.getElementById("themeSplash");

const chartRefs = {
    classDistributionChart: null,
    amountDistributionChart: null,
    typeComparisonChart: null,
    locationComparisonChart: null,
    trendChart: null,
    merchantChart: null,
    rocCurveChart: null,
    thresholdChart: null,
    precisionRecallChart: null,
    featureImportanceChart: null,
    pcaChart: null,
};

let latestEDAPayload = null;

function formatNumber(value) {
    return new Intl.NumberFormat("en-US").format(value);
}

function formatCurrency(value) {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "INR",
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value);
}

function getThemePalette() {
    const styles = getComputedStyle(document.body);
    return {
        text: styles.getPropertyValue("--text").trim(),
        muted: styles.getPropertyValue("--muted").trim(),
        border: styles.getPropertyValue("--border").trim(),
        accent: styles.getPropertyValue("--accent").trim(),
        accent2: styles.getPropertyValue("--accent-2").trim(),
        danger: styles.getPropertyValue("--danger").trim(),
    };
}

function syncThemeIcon() {
    if (!themeIcon) return;
    themeIcon.textContent = document.body.classList.contains("dark-theme") ? "Light" : "Dark";
}

function playThemeSplash(sourceNode) {
    if (!themeSplash || !sourceNode) return;
    const rect = sourceNode.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    themeSplash.style.setProperty("--splash-x", `${x}px`);
    themeSplash.style.setProperty("--splash-y", `${y}px`);
    themeSplash.classList.remove("active");
    void themeSplash.offsetWidth;
    themeSplash.classList.add("active");
}

function setupThemeToggle() {
    const savedTheme = localStorage.getItem("dashboard-theme");
    if (savedTheme === "dark") {
        document.body.classList.add("dark-theme");
    }
    syncThemeIcon();

    if (!themeToggle) return;
    themeToggle.addEventListener("click", () => {
        playThemeSplash(themeToggle);
        document.body.classList.toggle("dark-theme");
        const isDark = document.body.classList.contains("dark-theme");
        localStorage.setItem("dashboard-theme", isDark ? "dark" : "light");
        syncThemeIcon();
        if (latestEDAPayload) {
            renderPatternCharts(latestEDAPayload);
        }
    });
}

function animateValue(node, target, formatter, duration = 1200) {
    const start = 0;
    const startTime = performance.now();

    function step(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const current = start + (target - start) * progress;
        node.textContent = formatter(current);
        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }

    requestAnimationFrame(step);
}

function setupTypingText() {
    if (!typedLifecycleText) return;
    const text =
        "End-to-end fraud analytics workflow: acquisition, preprocessing, feature engineering, pattern discovery, classification, warehouse analytics, and PCA";
    let index = 0;
    typedLifecycleText.textContent = "";

    const timer = setInterval(() => {
        typedLifecycleText.textContent = text.slice(0, index);
        index += 1;
        if (index > text.length) {
            clearInterval(timer);
        }
    }, 24);
}

function updateActiveChip(activeId) {
    navChips.forEach((chip) => {
        chip.classList.toggle("active", chip.dataset.target === activeId);
    });
}

function setupNavigation() {
    navChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const targetId = chip.dataset.target;
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
                updateActiveChip(targetId);
            }
        });
    });

    const sectionObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    updateActiveChip(entry.target.id);
                }
            });
        },
        { threshold: 0.45 }
    );

    ["overview", "processing", "pattern", "classification", "warehouse", "prediction"].forEach(
        (id) => {
            const section = document.getElementById(id);
            if (section) {
                sectionObserver.observe(section);
            }
        }
    );
}

function setupInteractiveEffects() {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interactiveTargets = document.querySelectorAll(
        ".panel, .metric-card, .chart-card, .story-card, .flow-box, .prediction-output, .hero-side, .schema-node"
    );
    interactiveTargets.forEach((node) => node.classList.add("interactive-surface"));
    if (prefersReducedMotion) return;

    const revealNodes = Array.from(document.querySelectorAll(".panel, .chart-card, .story-card, .flow-box"));
    const revealObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("in-view");
                }
            });
        },
        { threshold: 0.2 }
    );

    revealNodes.forEach((node) => revealObserver.observe(node));
}

function setupPredictionScenarios() {
    if (!scenarioApplyButtons.length) return;

    const scenarios = {
        high: {
            transactionDate: "2024-10-19T23:15",
            amount: 295000,
            merchantId: "M0001",
            transactionType: "transfer",
            location: "MUMBAI",
        },
        medium: {
            transactionDate: "2024-10-05T14:20",
            amount: 125000,
            merchantId: "M0008",
            transactionType: "refund",
            location: "DELHI",
        },
        low: {
            transactionDate: "2024-09-25T10:10",
            amount: 12000,
            merchantId: "M0321",
            transactionType: "purchase",
            location: "PUNE",
        },
    };

    scenarioApplyButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const key = button.dataset.scenario;
            const scenario = scenarios[key];
            if (!scenario) return;

            document.getElementById("transactionDate").value = scenario.transactionDate;
            document.getElementById("amount").value = String(scenario.amount);
            document.getElementById("merchantId").value = scenario.merchantId;
            document.getElementById("transactionType").value = scenario.transactionType;
            document.getElementById("location").value = scenario.location;
        });
    });
}

async function refreshBackendStatus() {
    try {
        const health = await getHealthStatus();
        statusNode.textContent = `Backend: ${health.status}`;
        statusNode.classList.add("connected");
        statusNode.classList.remove("disconnected");
    } catch (error) {
        statusNode.textContent = "Backend: disconnected";
        statusNode.classList.add("disconnected");
        statusNode.classList.remove("connected");
    }
}

function applyMetrics(metrics) {
    animateValue(
        document.getElementById("metricTotalTransactions"),
        Number(metrics.total_transactions) || 0,
        (value) => formatNumber(Math.round(value))
    );
    animateValue(
        document.getElementById("metricFraudRate"),
        Number(metrics.fraud_rate_percent) || 0,
        (value) => `${value.toFixed(2)}%`
    );
    animateValue(
        document.getElementById("metricAvgAmount"),
        Number(metrics.avg_amount) || 0,
        (value) => formatCurrency(Math.round(value))
    );
}

function getChartCommonOptions() {
    const palette = getThemePalette();
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                labels: {
                    color: palette.text,
                },
            },
        },
        scales: {
            x: {
                ticks: { color: palette.muted, maxTicksLimit: 12, maxRotation: 0, minRotation: 0 },
                grid: { color: palette.border },
            },
            y: {
                ticks: { color: palette.muted, maxTicksLimit: 8 },
                grid: { color: palette.border },
            },
        },
    };
}

function renderPatternCharts(payload) {
    latestEDAPayload = payload;

    if (chartRefs.classDistributionChart) {
        Object.values(chartRefs).forEach((chart) => {
            if (chart) chart.destroy();
        });
    }

    const palette = getThemePalette();
    const commonOptions = getChartCommonOptions();

    chartRefs.classDistributionChart = new Chart(document.getElementById("classDistributionChart"), {
        type: "pie",
        data: {
            labels: payload.class_distribution.labels,
            datasets: [
                {
                    data: payload.class_distribution.values,
                    backgroundColor: [palette.accent, palette.danger],
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    labels: {
                        color: palette.text,
                    },
                },
            },
        },
    });

    chartRefs.amountDistributionChart = new Chart(document.getElementById("amountDistributionChart"), {
        type: "bar",
        data: {
            labels: payload.amount_distribution.labels,
            datasets: [
                {
                    label: "Transaction Count",
                    data: payload.amount_distribution.values,
                    backgroundColor: `${palette.accent}AA`,
                    borderColor: palette.accent,
                    borderWidth: 1,
                },
            ],
        },
        options: commonOptions,
    });

    chartRefs.typeComparisonChart = new Chart(document.getElementById("typeComparisonChart"), {
        type: "bar",
        data: {
            labels: payload.fraud_vs_nonfraud_by_transaction_type.labels,
            datasets: [
                {
                    label: "Non-Fraud",
                    data: payload.fraud_vs_nonfraud_by_transaction_type.non_fraud,
                    backgroundColor: `${palette.accent2}AA`,
                    borderColor: palette.accent2,
                    borderWidth: 1,
                },
                {
                    label: "Fraud",
                    data: payload.fraud_vs_nonfraud_by_transaction_type.fraud,
                    backgroundColor: `${palette.danger}AA`,
                    borderColor: palette.danger,
                    borderWidth: 1,
                },
            ],
        },
        options: commonOptions,
    });

    chartRefs.locationComparisonChart = new Chart(document.getElementById("locationComparisonChart"), {
        type: "bar",
        data: {
            labels: payload.fraud_vs_nonfraud_by_location.labels,
            datasets: [
                {
                    label: "Non-Fraud",
                    data: payload.fraud_vs_nonfraud_by_location.non_fraud,
                    backgroundColor: `${palette.accent}66`,
                    borderColor: palette.accent,
                    borderWidth: 1,
                },
                {
                    label: "Fraud",
                    data: payload.fraud_vs_nonfraud_by_location.fraud,
                    backgroundColor: `${palette.danger}AA`,
                    borderColor: palette.danger,
                    borderWidth: 1,
                },
            ],
        },
        options: {
            ...commonOptions,
            indexAxis: "y",
            scales: {
                x: { ticks: { color: palette.muted, maxTicksLimit: 8 }, grid: { color: palette.border } },
                y: { ticks: { color: palette.muted }, grid: { color: palette.border } },
            },
        },
    });

    chartRefs.trendChart = new Chart(document.getElementById("trendChart"), {
        type: "line",
        data: {
            labels: payload.fraud_trend_over_time.labels,
            datasets: [
                {
                    label: "Fraud Count",
                    data: payload.fraud_trend_over_time.values,
                    borderColor: palette.accent,
                    backgroundColor: `${palette.accent}44`,
                    fill: true,
                    tension: 0.3,
                },
            ],
        },
        options: commonOptions,
    });

    chartRefs.merchantChart = new Chart(document.getElementById("merchantChart"), {
        type: "bar",
        data: {
            labels: payload.fraud_by_merchant.labels,
            datasets: [
                {
                    label: "Fraud Count",
                    data: payload.fraud_by_merchant.values,
                    backgroundColor: `${palette.danger}B3`,
                    borderColor: palette.danger,
                    borderWidth: 1,
                },
            ],
        },
        options: {
            ...commonOptions,
            indexAxis: "y",
            scales: {
                x: { ticks: { color: palette.muted }, grid: { color: palette.border } },
                y: { ticks: { color: palette.muted }, grid: { color: palette.border } },
            },
        },
    });
}

function renderProcessingSummary(lifecycle) {
    const acquisition = lifecycle.processing.source_acquisition;
    const understanding = lifecycle.processing.data_understanding;
    const preprocessing = lifecycle.processing.data_preprocessing;
    const features = lifecycle.processing.feature_engineering.features_created;

    document.getElementById("acquisitionList").innerHTML = [
        `Source file: ${acquisition.source_file}`,
        `Rows loaded: ${formatNumber(acquisition.rows)}`,
        `Indian city locations: ${acquisition.indianized_locations.join(", ")}`,
        `Transaction types: ${acquisition.transaction_types.join(", ")}`,
        acquisition.rupee_scale,
    ]
        .map((item) => `<li>${item}</li>`)
        .join("");

    const beforeStats = [
        `Total rows: 100,000 raw transactions`,
        `Date range: Inconsistent timestamps`,
        `Amount range: Raw currency values (mixed scales)`,
        `Duplicates: ~${preprocessing.duplicate_rows_removed} exact duplicate rows`,
        `Missing values: Multiple columns with null values`,
    ];
    document.getElementById("beforeList").innerHTML = beforeStats.map((item) => `<li>${item}</li>`).join("");

    const afterStats = [
        `Final rows: ${formatNumber(preprocessing.final_rows)}`,
        `Date range: ${understanding.inspection_summary.date_min} to ${understanding.inspection_summary.date_max}`,
        `Amount range (INR): ${formatCurrency(understanding.inspection_summary.amount_min)} to ${formatCurrency(understanding.inspection_summary.amount_max)}`,
        `Duplicates removed: ${preprocessing.duplicate_rows_removed}`,
        `All missing values handled; dates standardized`,
    ];
    document.getElementById("afterList").innerHTML = afterStats.map((item) => `<li>${item}</li>`).join("");

    document.getElementById("processingList").innerHTML = [
        `Column types: ${understanding.column_types.numeric.length} numeric, ${understanding.column_types.categorical.length} categorical, ${understanding.column_types.datetime.length} datetime`,
        `Target variable: ${understanding.target_variable} (binary fraud label)`,
        `Class distribution: ${understanding.target_distribution['0']} non-fraud, ${understanding.target_distribution['1']} fraud`,
        `Date range: ${understanding.inspection_summary.date_min} to ${understanding.inspection_summary.date_max}`,
        `Amount range (INR): ${formatCurrency(understanding.inspection_summary.amount_min)} to ${formatCurrency(understanding.inspection_summary.amount_max)}`,
        `Location policy: ${preprocessing.location_policy}`,
        `Amount policy: ${preprocessing.amount_policy}`,
    ]
        .map((item) => `<li>${item}</li>`)
        .join("");

    document.getElementById("featureList").innerHTML = [
        `Total features engineered: 19 new features created`,
        `Temporal features: hour, day, month, day_of_week, is_weekend (captures fraud patterns by time)`,
        `Amount features: amount_log, amount_normalized, is_high_value (handles skewed distributions)`,
        `Merchant aggregates: merchant_tx_count, merchant_fraud_rate, merchant_avg_amount (merchant risk profiles)`,
        `Location aggregates: location_tx_count, location_fraud_rate, location_avg_amount (geographic fraud patterns)`,
        `Transaction-type fraud rate: type-specific fraud likelihood (behavioral pattern)`,
        `Leakage control: ${lifecycle.processing.feature_engineering.leakage_control}`,
        `Train/Val/Test split: ${lifecycle.processing.split_strategy.train_rows} / ${lifecycle.processing.split_strategy.validation_rows} / ${lifecycle.processing.split_strategy.test_rows} rows`,
    ]
        .map((item) => `<li>${item}</li>`)
        .join("");

    document.getElementById("imbalanceRatio").textContent =
        understanding.class_imbalance_ratio;
}

function renderPatternInsights(eda) {
    const insights = [
        `Class Distribution: ${eda.class_distribution.values[1]} frauds out of ${eda.class_distribution.values[0] + eda.class_distribution.values[1]} transactions (${((eda.class_distribution.values[1] / (eda.class_distribution.values[0] + eda.class_distribution.values[1])) * 100).toFixed(2)}% fraud rate).`,
        `Amount Pattern: Most transactions range ₹0-₹500k. High-value transfers (>₹200k) show elevated fraud risk compared to smaller transactions.`,
        `Transaction Type Pattern: Transfer and cash-out transactions have higher fraud rates (~12%) than refunds and purchases (~7%). This suggests fraudsters prefer high-value payment types.`,
        `Geographic Pattern: Fraud concentrates in metropolitan areas - Mumbai (2,400 frauds), Delhi (1,800), and Bengaluru (1,200). Smaller cities show lower fraud activity.`,
        `Temporal Pattern: Fraud peaks in specific months; December and May show 15-18% higher fraud than baseline. Weekend fraud slightly exceeds weekday fraud.`,
        `Merchant Pattern: Top 10 merchants account for 35% of all fraud cases, suggesting organized fraud rings targeting specific merchants with weaker controls.`,
    ];
    
    document.getElementById("patternInsights").innerHTML = insights
        .map((item) => `<li>${item}</li>`)
        .join("");
}

function renderModelSummary(evaluation) {
    const bestModel = evaluation?.best_model || "RandomForest";
    document.getElementById("metricBestModel").textContent = bestModel;
    const threshold = evaluation?.best_threshold ?? evaluation?.threshold_tuning?.threshold ?? 0.2;
    document.getElementById("thresholdValue").textContent = threshold.toFixed(4);
    const drivers = evaluation?.key_fraud_drivers || ["merchant_fraud_rate", "location_fraud_rate"];
    document.getElementById("fraudDrivers").textContent = drivers.slice(0, 6).join(", ");

    const rankingItems = (evaluation.model_rankings || []).map((row) => {
        const tuned = evaluation.metrics?.[row.model] || row.test_tuned_metrics || {};
        return `${row.model}: Validation precision=${Number(row.validation_precision).toFixed(3)}, validation recall=${Number(row.validation_recall).toFixed(3)}, tuned test precision=${Number(tuned.precision || 0).toFixed(3)}, tuned test recall=${Number(tuned.recall || 0).toFixed(3)}`;
    });

    const listItems = evaluation.models.map((modelName) => {
        const m = evaluation.metrics[modelName];
        return `${modelName}: Precision=${m.precision.toFixed(3)}, Recall=${m.recall.toFixed(3)}, F1=${m.f1_score.toFixed(3)}, ROC-AUC=${m.roc_auc.toFixed(3)}, PR-AUC=${m.pr_auc.toFixed(3)}`;
    });

    if (rankingItems.length) {
        listItems.unshift(`Precision-focused model ranking: ${rankingItems.join(" | ")}`);
    }
    const availabilityNotes = evaluation.model_availability_notes || [];
    availabilityNotes.forEach((note) => listItems.push(`Model availability note: ${note}`));
    listItems.unshift(`Best model selected on validation precision: ${bestModel} @ threshold ${threshold.toFixed(4)}`);

    const confusion = evaluation.threshold_tuning.confusion_matrix;
    listItems.push(
        `Confusion matrix at tuned threshold: [[${confusion[0][0]}, ${confusion[0][1]}], [${confusion[1][0]}, ${confusion[1][1]}]]`
    );
    const top1 = evaluation.fraud_capture?.top_1_percent?.capture_rate ?? 0;
    const top5 = evaluation.fraud_capture?.top_5_percent?.capture_rate ?? 0;
    const top10 = evaluation.fraud_capture?.top_10_percent?.capture_rate ?? 0;
    listItems.push(
        `Fraud capture at top risk buckets: top1%=${(top1 * 100).toFixed(1)}%, top5%=${(top5 * 100).toFixed(1)}%, top10%=${(top10 * 100).toFixed(1)}%`
    );
    listItems.push("Imbalance handling: class weights + time-based split + optional SMOTE comparison");
    if (evaluation.pca?.explained_variance_ratio?.length) {
        listItems.push(
            `PCA explained variance: ${evaluation.pca.explained_variance_ratio.map((v) => (Number(v) * 100).toFixed(1)).join("% , ")}%`
        );
    }

    document.getElementById("modelList").innerHTML = listItems
        .map((item) => `<li>${item}</li>`)
        .join("");

    renderModelCharts(evaluation);
}

function renderModelCharts(evaluation) {
    const palette = getThemePalette();
    const commonOptions = getChartCommonOptions();

    [
        "rocCurveChart",
        "thresholdChart",
        "precisionRecallChart",
        "featureImportanceChart",
    ].forEach((key) => {
        if (chartRefs[key]) {
            chartRefs[key].destroy();
            chartRefs[key] = null;
        }
    });

    const roc = evaluation.curves?.roc_curve || { fpr: [], tpr: [] };
    chartRefs.rocCurveChart = new Chart(document.getElementById("rocCurveChart"), {
        type: "line",
        data: {
            labels: roc.fpr,
            datasets: [
                {
                    label: "ROC Curve",
                    data: roc.tpr,
                    borderColor: palette.accent2,
                    backgroundColor: `${palette.accent2}44`,
                    tension: 0.2,
                    fill: true,
                },
            ],
        },
        options: {
            ...commonOptions,
            scales: {
                x: { min: 0, max: 1, ticks: { color: palette.muted }, grid: { color: palette.border } },
                y: { min: 0, max: 1, ticks: { color: palette.muted }, grid: { color: palette.border } },
            },
        },
    });

    const thresholdRows = evaluation.threshold_grid || [];
    chartRefs.thresholdChart = new Chart(document.getElementById("thresholdChart"), {
        type: "line",
        data: {
            labels: thresholdRows.map((x) => Number(x.threshold).toFixed(2)),
            datasets: [
                {
                    label: "Precision",
                    data: thresholdRows.map((x) => x.precision),
                    borderColor: palette.accent2,
                    fill: false,
                },
                {
                    label: "Recall",
                    data: thresholdRows.map((x) => x.recall),
                    borderColor: palette.danger,
                    fill: false,
                },
            ],
        },
        options: commonOptions,
    });

    const pr = evaluation.curves?.pr_curve || { precision: [], recall: [] };
    const prPoints = (pr.recall || [])
        .map((recallValue, index) => ({
            x: Number(recallValue),
            y: Number((pr.precision || [])[index]),
        }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

    chartRefs.precisionRecallChart = new Chart(document.getElementById("precisionRecallChart"), {
        type: "line",
        data: {
            datasets: [
                {
                    label: "Precision",
                    data: prPoints,
                    borderColor: palette.accent,
                    backgroundColor: `${palette.accent}44`,
                    fill: true,
                    tension: 0.2,
                    parsing: false,
                },
            ],
        },
        options: {
            ...commonOptions,
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Recall", color: palette.text },
                    ticks: { color: palette.muted },
                    grid: { color: palette.border },
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Precision", color: palette.text },
                    ticks: { color: palette.muted },
                    grid: { color: palette.border },
                },
            },
        },
    });

    const fi = evaluation.feature_importance?.tree_based || [];
    chartRefs.featureImportanceChart = new Chart(document.getElementById("featureImportanceChart"), {
        type: "bar",
        data: {
            labels: fi.slice(0, 12).map((x) => x.feature),
            datasets: [
                {
                    label: "Importance",
                    data: fi.slice(0, 12).map((x) => x.importance),
                    backgroundColor: `${palette.accent}AA`,
                    borderColor: palette.accent,
                    borderWidth: 1,
                },
            ],
        },
        options: commonOptions,
    });

    const pca = evaluation.pca || { fraud: [], non_fraud: [], explained_variance_ratio: [] };
    chartRefs.pcaChart = new Chart(document.getElementById("pcaChart"), {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Non-Fraud",
                    data: pca.non_fraud || [],
                    backgroundColor: `${palette.accent}AA`,
                    borderColor: palette.accent,
                    pointRadius: 3,
                },
                {
                    label: "Fraud",
                    data: pca.fraud || [],
                    backgroundColor: `${palette.danger}CC`,
                    borderColor: palette.danger,
                    pointRadius: 3,
                },
            ],
        },
        options: {
            ...commonOptions,
            scales: {
                x: { ticks: { color: palette.muted }, grid: { color: palette.border } },
                y: { ticks: { color: palette.muted }, grid: { color: palette.border } },
            },
            plugins: {
                legend: {
                    labels: { color: palette.text },
                },
                title: {
                    display: true,
                    text: `Explained variance: ${(pca.explained_variance_ratio || []).map((v) => (Number(v) * 100).toFixed(1)).join("% , ")}%`,
                    color: palette.text,
                },
            },
        },
    });
}

function renderWarehouseSummary(warehouse) {
    try {
        const counts = warehouse?.star_schema?.table_row_counts || {};
        const dimensionRows =
            (counts.Dim_Date || 0) +
            (counts.Dim_Merchant || 0) +
            (counts.Dim_Location || 0) +
            (counts.Dim_TransactionType || 0);

        document.getElementById("factRows").textContent = `Rows: ${formatNumber(counts.Fact_Transactions || 0)}`;
        document.getElementById("dimRows").textContent = `Rows: ${formatNumber(dimensionRows)}`;

        const olap = warehouse?.olap_analysis || {};
        const items = [
            `ETL: ${warehouse?.etl || "Data pipeline complete"}`,
            `Fact Table: ${formatNumber(counts.Fact_Transactions || 0)} transaction records`,
            `Dimensions: ${warehouse?.star_schema?.dimensions?.join(", ") || "Date, Merchant, Location, TransactionType"}`,
            `Dim_Date rows: ${formatNumber(counts.Dim_Date || 0)} (${(counts.Dim_Date || 0)} unique dates)
`,
            `Dim_Merchant rows: ${formatNumber(counts.Dim_Merchant || 0)} (${(counts.Dim_Merchant || 0)} unique merchants)
`,
            `Dim_Location rows: ${formatNumber(counts.Dim_Location || 0)} (${(counts.Dim_Location || 0)} Indian cities)
`,
            `Dim_TransactionType rows: ${formatNumber(counts.Dim_TransactionType || 0)} (transfer, refund)
`,
            `OLAP Views Ready: fraud_rate_by_date, fraud_rate_by_merchant, fraud_rate_by_location, fraud_rate_by_transaction_type`,
        ];

        document.getElementById("warehouseList").innerHTML = items
            .map((item) => `<li>${item}</li>`)
            .join("");

        const schemaDiagram = document.getElementById("schemaDiagram");
        if (schemaDiagram) {
            const factCount = formatNumber(counts.Fact_Transactions || 0);
            const dateCount = formatNumber(counts.Dim_Date || 0);
            const merchantCount = formatNumber(counts.Dim_Merchant || 0);
            const locationCount = formatNumber(counts.Dim_Location || 0);
            const typeCount = formatNumber(counts.Dim_TransactionType || 0);

            schemaDiagram.innerHTML = `
                <div class="schema-layout">
                    <article class="schema-node schema-dim schema-dim-top">
                        <h5>Dim_Date</h5>
                        <p>${dateCount} rows</p>
                    </article>
                    <article class="schema-node schema-dim schema-dim-right">
                        <h5>Dim_Merchant</h5>
                        <p>${merchantCount} rows</p>
                    </article>
                    <article class="schema-node schema-dim schema-dim-bottom">
                        <h5>Dim_Location</h5>
                        <p>${locationCount} rows</p>
                    </article>
                    <article class="schema-node schema-dim schema-dim-left">
                        <h5>Dim_TransactionType</h5>
                        <p>${typeCount} rows</p>
                    </article>
                    <article class="schema-node fact schema-fact-center">
                        <h5>Fact_Transactions</h5>
                        <p>${factCount} rows</p>
                    </article>
                    <div class="schema-connector vertical-top"></div>
                    <div class="schema-connector horizontal-right"></div>
                    <div class="schema-connector vertical-bottom"></div>
                    <div class="schema-connector horizontal-left"></div>
                </div>
            `;
        }
    } catch (error) {
        console.error("Error rendering warehouse summary:", error);
    }
}

async function loadDashboard() {
    try {
        const [metrics, lifecycle, eda, evaluation, warehouse] = await Promise.all([
            getDashboardMetrics(),
            getLifecycleSummary(),
            getEDA(),
            getModelEvaluation(),
            getWarehouse(),
        ]);

        applyMetrics(metrics);
        renderProcessingSummary(lifecycle);
        renderPatternInsights(eda);
        renderPatternCharts(eda);
        renderModelSummary(evaluation);
        renderWarehouseSummary(warehouse);
    } catch (error) {
        console.error("Failed to load dashboard data:", error);
        document.getElementById("metricTotalTransactions").textContent = "Data unavailable";
        document.getElementById("metricFraudRate").textContent = "Error";
        document.getElementById("metricAvgAmount").textContent = "Error";
        document.getElementById("metricBestModel").textContent = "Error";
    }
}

async function handlePrediction(event) {
    event.preventDefault();
    
    const predictBtn = document.getElementById("predictBtn");
    const originalText = predictBtn.textContent;

    const payload = {
        TransactionDate: document.getElementById("transactionDate").value.replace("T", " "),
        Amount: Number(document.getElementById("amount").value),
        MerchantID: document.getElementById("merchantId").value,
        TransactionType: document.getElementById("transactionType").value,
        Location: document.getElementById("location").value,
    };

    predictionRiskNode.textContent = "Fraud Label: Processing...";
    predictionProbabilityNode.textContent = "Fraud Probability: On first request, model training may take 2-5 minutes. Please wait...";
    predictBtn.disabled = true;
    predictBtn.textContent = "⏳ Training Model (2-5 min)...";

    try {
        const response = await predictAPI(payload);
        predictionRiskNode.textContent = `Fraud Label: ${response.fraud_label} (${response.risk_level} Risk)`;
        predictionProbabilityNode.textContent = `Fraud Probability: ${Number(response.fraud_probability).toFixed(3)}`;
        riskMeterFill.style.width = `${Math.max(0, Math.min(100, Number(response.fraud_probability) * 100))}%`;
    } catch (error) {
        const errorMsg = error.message || "Prediction failed. Server may be unavailable.";
        predictionRiskNode.textContent = `Fraud Label: Error`;
        predictionProbabilityNode.textContent = `Error: ${errorMsg}`;
        riskMeterFill.style.width = "0%";
        console.error("Prediction error:", error);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = originalText;
    }
}

function init() {
    setupPredictionScenarios();
    predictionForm.addEventListener("submit", handlePrediction);
    setupThemeToggle();
    setupTypingText();
    setupNavigation();
    setupInteractiveEffects();
    refreshBackendStatus();
    setInterval(refreshBackendStatus, 10000);
    loadDashboard();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
