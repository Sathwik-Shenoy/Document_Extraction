const apiBaseInput = document.getElementById("api-base");
const apiKeyInput = document.getElementById("api-key");
const fileInput = document.getElementById("file-input");
const filePill = document.getElementById("file-pill");
const dropzone = document.getElementById("dropzone");
const feedback = document.getElementById("feedback");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const refreshHealthBtn = document.getElementById("refresh-health");
const loadSampleBtn = document.getElementById("load-sample");
const sampleReportBtn = document.getElementById("sample-report");
const sampleShoppingBtn = document.getElementById("sample-shopping");

const serviceName = document.getElementById("service-name");
const serviceVersion = document.getElementById("service-version");
const healthStatus = document.getElementById("health-status");
const healthDetail = document.getElementById("health-detail");
const dependencyStatus = document.getElementById("dependency-status");
const dependencyDetail = document.getElementById("dependency-detail");
const apiHealthChip = document.getElementById("api-health-chip");
const metricsChip = document.getElementById("metrics-chip");
const authChip = document.getElementById("auth-chip");
const resultMeta = document.getElementById("result-meta");
const summaryText = document.getElementById("summary-text");
const sentimentText = document.getElementById("sentiment-text");
const confidenceList = document.getElementById("confidence-list");
const entityList = document.getElementById("entity-list");
const extractedText = document.getElementById("extracted-text");
const metadataJson = document.getElementById("metadata-json");

const sampleDocs = {
  report: {
    filename: "internal-report.txt",
    text: [
      "Quarterly update: Acme Corp reached $500000 in revenue on 15 March 2026.",
      "The team delivered strong product growth, improved customer satisfaction, and positive momentum.",
      "Contact: Maya Patel, Director at Acme Corp.",
    ].join(" "),
  },
  shopping: {
    filename: "shopify-training.txt",
    text: [
      "Shopify training plan for Digital Heroes includes product research, homepage design, and checkout testing.",
      "The launch goal is to build one fully functional store with 3 to 10 products and strong trust signals.",
      "Revenue target: $2500 USD by 30 April 2026.",
    ].join(" "),
  },
};

let currentFile = null;

function apiBase() {
  return apiBaseInput.value.trim().replace(/\/+$/, "");
}

function setFeedback(message, tone = "") {
  feedback.textContent = message;
  feedback.className = `feedback ${tone}`.trim();
}

function saveApiKey() {
  localStorage.setItem("doc-extractor-api-key", apiKeyInput.value.trim());
}

function loadApiKey() {
  const saved = localStorage.getItem("doc-extractor-api-key");
  if (saved) {
    apiKeyInput.value = saved;
  }
}

function setFile(file) {
  currentFile = file;
  if (file) {
    filePill.textContent = `${file.name} · ${Math.round(file.size / 1024)} KB`;
    setFeedback(`Ready to analyze ${file.name}.`, "warn");
  } else {
    filePill.textContent = "No file selected";
    setFeedback("Upload a document to begin.");
  }
}

function resetResults() {
  summaryText.textContent = "The summary will appear here.";
  sentimentText.textContent = "Waiting for analysis.";
  confidenceList.innerHTML = "";
  entityList.innerHTML = "";
  extractedText.textContent = "The extracted text will appear here.";
  metadataJson.textContent = "{}";
  resultMeta.textContent = "No analysis yet";
}

function appendStatus(target, value, tone) {
  target.textContent = value;
  target.className = tone;
}

function renderConfidence(scores = {}) {
  confidenceList.innerHTML = "";
  const entries = Object.entries(scores);
  if (!entries.length) {
    confidenceList.innerHTML = "<li>No confidence data returned.</li>";
    return;
  }
  for (const [key, value] of entries) {
    const item = document.createElement("li");
    item.textContent = `${key}: ${Number(value).toFixed(3)}`;
    confidenceList.appendChild(item);
  }
}

function renderEntities(entities = []) {
  entityList.innerHTML = "";
  if (!entities.length) {
    entityList.innerHTML = '<span class="chip">No entities found</span>';
    return;
  }
  for (const entity of entities.slice(0, 12)) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${entity.entity} · ${entity.type}`;
    entityList.appendChild(chip);
  }
}

function renderAnalysis(data) {
  summaryText.textContent = data.summary || "No summary was generated.";
  const sentiment = data.sentiment || {};
  sentimentText.textContent = `${sentiment.label || "unknown"} · score ${Number(sentiment.score || 0).toFixed(3)}\n${sentiment.explanation || ""}`;
  renderConfidence(data.confidence_scores || {});
  renderEntities(data.entities || []);
  extractedText.textContent = data.extracted_text || "";
  metadataJson.textContent = JSON.stringify(data.metadata || {}, null, 2);
  resultMeta.textContent = `${data.file_type || "unknown"} · ${data.document_id || "no id"}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  return { ok: response.ok, status: response.status, data: await response.json() };
}

async function refreshStatus() {
  const base = apiBase();
  try {
    const [health, apiHealth, dependencies, metrics, auth] = await Promise.all([
      fetchJson(`${base}/health`),
      fetchJson(`${base}/api/v1/health`),
      fetchJson(`${base}/health/dependencies`),
      fetchJson(`${base}/api/v1/metrics`),
      fetchJson(`${base}/api/v1/auth-status`),
    ]);

    serviceName.textContent = health.data.service || "Document analyzer";
    serviceVersion.textContent = `v${health.data.version || "?"}`;
    appendStatus(healthStatus, health.data.status || `HTTP ${health.status}`, health.ok ? "ok" : "bad");
    healthDetail.textContent = apiHealth.data.service ? apiHealth.data.service : "";
    appendStatus(dependencyStatus, dependencies.data.extractors?.all_extractors_ready ? "Ready" : "Check needed", dependencies.data.extractors?.all_extractors_ready ? "ok" : "warn");
    const ready = dependencies.data.extractors?.all_extractors_ready;
    dependencyDetail.textContent = ready ? "All extractors available" : "Some extractors need attention";

    appendStatus(apiHealthChip, apiHealth.ok ? "Healthy" : "Down", apiHealth.ok ? "ok" : "bad");
    appendStatus(metricsChip, metrics.ok ? `${metrics.data.total_processed || 0} processed` : "Unavailable", metrics.ok ? "ok" : "bad");
    appendStatus(authChip, auth.data.require_api_key ? "Enabled" : "Optional", auth.data.require_api_key ? "warn" : "ok");
  } catch (error) {
    appendStatus(healthStatus, "Unavailable", "bad");
    healthDetail.textContent = error.message;
    appendStatus(dependencyStatus, "Unavailable", "bad");
    dependencyDetail.textContent = error.message;
    appendStatus(apiHealthChip, "Unavailable", "bad");
    appendStatus(metricsChip, "Unavailable", "bad");
    appendStatus(authChip, "Unavailable", "bad");
  }
}

async function analyzeCurrentFile() {
  if (!currentFile) {
    setFeedback("Choose a file first.", "bad");
    return;
  }

  const base = apiBase();
  const formData = new FormData();
  formData.append("file", currentFile);

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  setFeedback(`Uploading ${currentFile.name} to the analyzer...`, "warn");

  try {
    const headers = {};
    const apiKey = apiKeyInput.value.trim();
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    const response = await fetch(`${base}/api/v1/documents/analyze`, {
      method: "POST",
      headers,
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail?.message || data?.detail || `Request failed with HTTP ${response.status}`;
      throw new Error(typeof message === "string" ? message : JSON.stringify(message));
    }
    renderAnalysis(data);
    setFeedback(`Analysis complete for ${currentFile.name}.`, "ok");
    await refreshStatus();
  } catch (error) {
    setFeedback(error.message, "bad");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze document";
  }
}

function sampleToFile(sample) {
  return new File([sample.text], sample.filename, { type: "text/plain" });
}

function wireSampleButtons() {
  sampleReportBtn.addEventListener("click", () => {
    setFile(sampleToFile(sampleDocs.report));
  });
  sampleShoppingBtn.addEventListener("click", () => {
    setFile(sampleToFile(sampleDocs.shopping));
  });
}

function wireDragDrop() {
  const prevent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, prevent);
    document.body.addEventListener(eventName, prevent);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add("dragover"));
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragover"));
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files?.[0];
    if (file) {
      fileInput.files = event.dataTransfer.files;
      setFile(file);
    }
  });
}

function wireEvents() {
  fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] || null));
  analyzeBtn.addEventListener("click", analyzeCurrentFile);
  clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    setFile(null);
    resetResults();
  });
  refreshHealthBtn.addEventListener("click", refreshStatus);
  loadSampleBtn.addEventListener("click", () => {
    setFile(sampleToFile(sampleDocs.report));
  });
  apiKeyInput.addEventListener("change", saveApiKey);
  apiKeyInput.addEventListener("blur", saveApiKey);
}

function init() {
  loadApiKey();
  wireEvents();
  wireSampleButtons();
  wireDragDrop();
  resetResults();
  refreshStatus();
}

init();
