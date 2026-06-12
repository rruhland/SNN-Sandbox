const els = {
  deviceLine: document.querySelector("#deviceLine"),
  statusValue: document.querySelector("#statusValue"),
  startBtn: document.querySelector("#startBtn"),
  stopBtn: document.querySelector("#stopBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  saveBtn: document.querySelector("#saveBtn"),
  pullBtn: document.querySelector("#pullBtn"),
  loadBtn: document.querySelector("#loadBtn"),
  speedInput: document.querySelector("#speedInput"),
  speedValue: document.querySelector("#speedValue"),
  stateSelect: document.querySelector("#stateSelect"),
  accuracy: document.querySelector("#accuracy"),
  trials: document.querySelector("#trials"),
  trialsPerSecond: document.querySelector("#trialsPerSecond"),
  lastResult: document.querySelector("#lastResult"),
  graphTrial: document.querySelector("#graphTrial"),
  pullStatus: document.querySelector("#pullStatus"),
  inspectorState: document.querySelector("#inspectorState"),
  inspectorMeta: document.querySelector("#inspectorMeta"),
  emptyState: document.querySelector("#emptyState"),
  inspectorWorkspace: document.querySelector("#inspectorWorkspace"),
  architecture: document.querySelector("#architecture"),
  evalList: document.querySelector("#evalList"),
  accuracyCanvas: document.querySelector("#accuracyCanvas"),
  networkCanvas: document.querySelector("#networkCanvas"),
  rasterCanvas: document.querySelector("#rasterCanvas"),
  weightsCanvas: document.querySelector("#weightsCanvas"),
  confusionCanvas: document.querySelector("#confusionCanvas"),
  evalAccuracyCanvas: document.querySelector("#evalAccuracyCanvas"),
};

let liveMetrics = null;
let pulledSnapshot = null;
let renderedLiveSeq = -1;
let renderedPullTrial = -1;
const textCache = new Map();
const canvasCache = new WeakMap();

async function command(name, payload = {}) {
  const res = await fetch(`/api/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  if (name !== "pull_current") liveMetrics = data;
  return data;
}

async function refreshStates() {
  const res = await fetch("/api/states");
  const data = await res.json();
  els.stateSelect.innerHTML = "";
  for (const state of data.states) {
    const option = document.createElement("option");
    option.value = state.name;
    option.textContent = state.name;
    els.stateSelect.appendChild(option);
  }
}

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function setText(el, value) {
  if (!el || textCache.get(el) === value) return;
  el.textContent = value;
  textCache.set(el, value);
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || canvas.width / ratio;
  const cssHeight = canvas.clientHeight || canvas.height / ratio;
  const pixelWidth = Math.max(1, Math.floor(cssWidth * ratio));
  const pixelHeight = Math.max(1, Math.floor(cssHeight * ratio));
  let cached = canvasCache.get(canvas);
  if (!cached) {
    cached = { ctx: canvas.getContext("2d", { alpha: false }), ratio: 0, width: 0, height: 0 };
    canvasCache.set(canvas, cached);
  }
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  if (cached.width !== pixelWidth || cached.height !== pixelHeight || cached.ratio !== ratio) {
    cached.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    cached.width = pixelWidth;
    cached.height = pixelHeight;
    cached.ratio = ratio;
  }
  return { ctx: cached.ctx, width: cssWidth, height: cssHeight, pixelWidth, pixelHeight };
}

function clear(ctx, width, height) {
  ctx.fillStyle = "#0c100f";
  ctx.fillRect(0, 0, width, height);
}

function drawAccuracyGraph(metrics) {
  const { ctx, width, height } = setupCanvas(els.accuracyCanvas);
  clear(ctx, width, height);
  const history = metrics.history || [];
  ctx.strokeStyle = "rgba(159, 176, 166, 0.35)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = 18 + (i / 4) * (height - 42);
    ctx.beginPath();
    ctx.moveTo(18, y);
    ctx.lineTo(width - 16, y);
    ctx.stroke();
  }
  if (history.length < 2) return;
  ctx.strokeStyle = "#37d18f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((point, index) => {
    const x = 20 + (index / Math.max(1, history.length - 1)) * (width - 42);
    const y = height - 24 - Math.max(0, Math.min(1, point.accuracy || 0)) * (height - 48);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderLiveMetrics() {
  const metrics = liveMetrics;
  if (!metrics || metrics.seq === renderedLiveSeq) return;
  renderedLiveSeq = metrics.seq;
  setText(els.statusValue, metrics.running ? "Running" : "Stopped");
  setText(els.deviceLine, `${metrics.device || "-"}${metrics.gpuName ? ` (${metrics.gpuName})` : ""}`);
  setText(els.trials, String(metrics.trials || 0));
  setText(els.trialsPerSecond, String(metrics.trialsPerSecond || 0));
  setText(els.accuracy, percent(metrics.accuracy));
  const last = metrics.lastResult || {};
  setText(els.lastResult, last.pattern ? `${last.pattern} -> ${last.prediction} (${last.correct ? "ok" : "miss"})` : "-");
  setText(els.graphTrial, `Trial ${metrics.trials || 0}`);
  if (document.activeElement !== els.speedInput) {
    els.speedInput.value = metrics.episodesPerTick || 8;
  }
  setText(els.speedValue, String(metrics.episodesPerTick || 8));
  drawAccuracyGraph(metrics);
  if (pulledSnapshot) updateInspectorMeta();
}

function evaluatePulled(snapshot) {
  const config = snapshot.config;
  const inputHidden = snapshot.weights.inputHidden;
  const hiddenOutput = snapshot.weights.hiddenOutput;
  const patterns = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  return patterns.map((pattern) => {
    const encoded = Array(config.inputNeurons).fill(0);
    encoded[pattern[0] === 0 ? 0 : 1] = 1;
    encoded[pattern[1] === 0 ? 2 : 3] = 1;
    const hiddenCurrent = inputHidden[0].map((_, hiddenIndex) =>
      encoded.reduce((sum, value, inputIndex) => sum + value * inputHidden[inputIndex][hiddenIndex], 0)
    );
    const ranked = hiddenCurrent
      .map((value, index) => ({ value, index }))
      .filter((item) => item.value > config.hiddenThreshold)
      .sort((a, b) => b.value - a.value)
      .slice(0, config.hiddenWinners);
    const hiddenSpikes = Array(config.hiddenNeurons).fill(0);
    for (const item of ranked) hiddenSpikes[item.index] = 1;
    const outputDrive = Array(config.outputNeurons).fill(0);
    for (let output = 0; output < config.outputNeurons; output += 1) {
      let current = 0;
      for (let hidden = 0; hidden < config.hiddenNeurons; hidden += 1) {
        current += hiddenSpikes[hidden] * hiddenOutput[hidden][output];
      }
      outputDrive[output] = (current + (current > config.outputThreshold ? 0.05 : 0)) * config.episodeSteps;
    }
    const target = pattern[0] + pattern[1];
    const prediction = outputDrive.indexOf(Math.max(...outputDrive));
    const competitors = outputDrive.filter((_, index) => index !== target);
    return {
      pattern: `${pattern[0]}+${pattern[1]}`,
      target,
      prediction,
      correct: prediction === target,
      outputDrive,
      activeHidden: ranked.length,
      margin: outputDrive[target] - Math.max(...competitors),
    };
  });
}

function pullSummary(snapshot, evalResults) {
  const correct = evalResults.filter((item) => item.correct).length;
  return {
    evalAccuracy: correct / evalResults.length,
    evalResults,
  };
}

function updateInspectorMeta() {
  if (!pulledSnapshot) return;
  const liveTrial = liveMetrics?.trials ?? "-";
  setText(els.inspectorState, `Viewing model from trial ${pulledSnapshot.trial}; trainer currently at trial ${liveTrial}.`);
  setText(
    els.inspectorMeta,
    `Pulled ${new Date(pulledSnapshot.pulledAt).toLocaleTimeString()} | export ${pulledSnapshot.exportMs}ms`
  );
}

function renderPulledInspector(snapshot) {
  const evalResults = evaluatePulled(snapshot);
  const summary = pullSummary(snapshot, evalResults);
  els.emptyState.classList.add("hidden");
  els.inspectorWorkspace.classList.remove("hidden");
  setText(els.pullStatus, `Pulled trial ${snapshot.trial}`);
  setText(els.architecture, `${snapshot.architecture.inputs} -> ${snapshot.architecture.hidden} -> ${snapshot.architecture.outputs}`);
  updateInspectorMeta();
  drawNetwork(snapshot, evalResults);
  drawRaster(snapshot);
  drawWeights(snapshot);
  drawEvaluation(evalResults);
  drawConfusion(snapshot.confusion || []);
  drawEvalAccuracy(summary.evalAccuracy);
}

function drawNetwork(snapshot, evalResults) {
  const { ctx, width, height } = setupCanvas(els.networkCanvas);
  clear(ctx, width, height);
  const recent = snapshot.recentSpikes || [];
  const activity = recent.length
    ? recent[0].map((_, col) => recent.reduce((sum, row) => sum + (row[col] || 0), 0) / recent.length)
    : [];
  const outputs = evalResults.at(-1)?.outputDrive || [0, 0, 0];
  const leftX = 70;
  const midX = width * 0.5;
  const rightX = width - 70;
  const inputY = [height * 0.28, height * 0.42, height * 0.58, height * 0.72];
  const outputY = [height * 0.3, height * 0.5, height * 0.7];
  ctx.strokeStyle = "rgba(88, 166, 255, 0.12)";
  for (const y of inputY) {
    for (let i = 0; i < Math.min(24, activity.length); i += 1) {
      const hy = 35 + (i / 23) * (height - 70);
      ctx.beginPath();
      ctx.moveTo(leftX, y);
      ctx.lineTo(midX, hy);
      ctx.stroke();
    }
  }
  for (let i = 0; i < Math.min(24, activity.length); i += 1) {
    const hy = 35 + (i / 23) * (height - 70);
    const active = activity[i] || 0;
    ctx.fillStyle = `rgba(55, 209, 143, ${0.2 + Math.min(0.8, active * 8)})`;
    ctx.beginPath();
    ctx.arc(midX, hy, 4 + Math.min(8, active * 24), 0, Math.PI * 2);
    ctx.fill();
  }
  inputY.forEach((y, index) => {
    ctx.fillStyle = "#58a6ff";
    ctx.beginPath();
    ctx.arc(leftX, y, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#d7e5dc";
    ctx.fillText(["A0", "A1", "B0", "B1"][index], leftX - 10, y + 28);
  });
  const maxOutput = Math.max(1, ...outputs);
  outputY.forEach((y, index) => {
    ctx.fillStyle = "#f0b84f";
    ctx.globalAlpha = Math.max(0.35, Math.min(1, (outputs[index] || 0) / maxOutput));
    ctx.beginPath();
    ctx.arc(rightX, y, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#d7e5dc";
    ctx.fillText(`${index}`, rightX - 4, y + 32);
  });
}

function drawRaster(snapshot) {
  const raster = snapshot.recentSpikes || [];
  const { ctx, width, height } = setupCanvas(els.rasterCanvas);
  clear(ctx, width, height);
  if (!raster.length) return;
  const rows = raster.length;
  const cols = raster[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  ctx.fillStyle = "#26312d";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#37d18f";
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      if (raster[y][x]) ctx.fillRect(x * cellW, y * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
    }
  }
}

function drawWeights(snapshot) {
  const weights = snapshot.weights.hiddenOutput || [];
  const { ctx, width, height } = setupCanvas(els.weightsCanvas);
  clear(ctx, width, height);
  if (!weights.length) return;
  const rows = weights.length;
  const cols = weights[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const value = Math.min(1, weights[y][x] / 1.4);
      ctx.fillStyle = `rgb(${Math.floor(22 + value * 70)},${Math.floor(34 + value * 190)},${Math.floor(44 + value * 120)})`;
      ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function drawEvaluation(evalResults) {
  els.evalList.innerHTML = "";
  for (const item of evalResults) {
    const div = document.createElement("div");
    div.className = `eval-item ${item.correct ? "ok" : "bad"}`;
    div.innerHTML = `<strong>${item.pattern} = ${item.target}</strong><span>pred ${item.prediction} | margin ${item.margin.toFixed(3)}</span>`;
    els.evalList.appendChild(div);
  }
}

function drawConfusion(matrix) {
  const { ctx, width, height } = setupCanvas(els.confusionCanvas);
  clear(ctx, width, height);
  const max = Math.max(1, ...matrix.flat());
  const size = Math.min(width, height - 28) / 3;
  ctx.font = "12px system-ui";
  for (let y = 0; y < 3; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      const value = matrix[y]?.[x] || 0;
      const intensity = value / max;
      ctx.fillStyle = x === y ? `rgba(55, 209, 143, ${0.18 + intensity * 0.75})` : `rgba(255, 107, 107, ${0.12 + intensity * 0.65})`;
      ctx.fillRect(30 + x * size, 12 + y * size, size - 4, size - 4);
      ctx.fillStyle = "#eef5ef";
      ctx.fillText(String(value), 30 + x * size + size * 0.42, 12 + y * size + size * 0.56);
    }
  }
}

function drawEvalAccuracy(value) {
  const { ctx, width, height } = setupCanvas(els.evalAccuracyCanvas);
  clear(ctx, width, height);
  ctx.fillStyle = "#9fb0a6";
  ctx.font = "15px system-ui";
  ctx.fillText("Evaluation Accuracy", 22, 34);
  ctx.fillStyle = "#eef5ef";
  ctx.font = "42px system-ui";
  ctx.fillText(percent(value), 22, 88);
}

async function pullCurrent() {
  setText(els.pullStatus, "Pulling...");
  const snapshot = await command("pull_current");
  pulledSnapshot = snapshot;
  renderedPullTrial = snapshot.trial;
  renderPulledInspector(snapshot);
}

function loop() {
  renderLiveMetrics();
  requestAnimationFrame(loop);
}

function connectStream() {
  const source = new EventSource("/api/stream?interval=0.05");
  source.onmessage = (event) => {
    liveMetrics = JSON.parse(event.data);
  };
  source.onerror = () => {
    source.close();
    setTimeout(connectStream, 1200);
  };
}

els.startBtn.addEventListener("click", () => command("start"));
els.stopBtn.addEventListener("click", () => command("stop"));
els.resetBtn.addEventListener("click", () => command("reset"));
els.saveBtn.addEventListener("click", async () => {
  await command("save", { name: `snn-state-${Date.now()}` });
  await refreshStates();
});
els.pullBtn.addEventListener("click", pullCurrent);
els.loadBtn.addEventListener("click", () => {
  if (els.stateSelect.value) command("load", { name: els.stateSelect.value });
});
els.speedInput.addEventListener("input", () => {
  setText(els.speedValue, els.speedInput.value);
});
els.speedInput.addEventListener("change", () => command("speed", { episodesPerTick: Number(els.speedInput.value) }));

fetch("/api/snapshot").then((res) => res.json()).then((data) => {
  liveMetrics = data;
});
refreshStates();
connectStream();
requestAnimationFrame(loop);
