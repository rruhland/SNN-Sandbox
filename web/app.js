const els = {
  deviceLine: document.querySelector("#deviceLine"),
  startBtn: document.querySelector("#startBtn"),
  stopBtn: document.querySelector("#stopBtn"),
  stepBtn: document.querySelector("#stepBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  saveBtn: document.querySelector("#saveBtn"),
  loadBtn: document.querySelector("#loadBtn"),
  speedInput: document.querySelector("#speedInput"),
  speedValue: document.querySelector("#speedValue"),
  stateSelect: document.querySelector("#stateSelect"),
  accuracy: document.querySelector("#accuracy"),
  evalAccuracy: document.querySelector("#evalAccuracy"),
  trials: document.querySelector("#trials"),
  lastResult: document.querySelector("#lastResult"),
  architecture: document.querySelector("#architecture"),
  evalList: document.querySelector("#evalList"),
  networkCanvas: document.querySelector("#networkCanvas"),
  rasterCanvas: document.querySelector("#rasterCanvas"),
  weightsCanvas: document.querySelector("#weightsCanvas"),
  confusionCanvas: document.querySelector("#confusionCanvas"),
};

let latestSnapshot = null;
let renderedSeq = -1;

async function command(name, payload = {}) {
  const res = await fetch(`/api/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  latestSnapshot = data;
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

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function clear(ctx, width, height) {
  ctx.fillStyle = "#0c100f";
  ctx.fillRect(0, 0, width, height);
}

function drawRaster(snapshot) {
  const { ctx, width, height } = setupCanvas(els.rasterCanvas);
  clear(ctx, width, height);
  const raster = snapshot.spikeRaster || [];
  if (!raster.length) return;
  const rows = raster.length;
  const cols = raster[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  ctx.fillStyle = "#26312d";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#37d18f";
  for (let y = 0; y < rows; y += 1) {
    const row = raster[y];
    for (let x = 0; x < cols; x += 1) {
      if (row[x]) ctx.fillRect(x * cellW, y * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
    }
  }
}

function drawWeights(snapshot) {
  const { ctx, width, height } = setupCanvas(els.weightsCanvas);
  clear(ctx, width, height);
  const weights = snapshot.hiddenOutputWeights || [];
  if (!weights.length) return;
  const rows = weights.length;
  const cols = weights[0].length;
  const cellW = width / cols;
  const cellH = height / rows;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const value = Math.min(1, weights[y][x] / 1.4);
      const r = Math.floor(22 + value * 70);
      const g = Math.floor(34 + value * 190);
      const b = Math.floor(44 + value * 120);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function drawNetwork(snapshot) {
  const { ctx, width, height } = setupCanvas(els.networkCanvas);
  clear(ctx, width, height);
  const activity = snapshot.hiddenActivity || [];
  const outputs = snapshot.lastResult?.outputDrive || [0, 0, 0];
  const leftX = 70;
  const midX = width * 0.5;
  const rightX = width - 70;
  const inputY = [height * 0.28, height * 0.42, height * 0.58, height * 0.72];
  const outputY = [height * 0.3, height * 0.5, height * 0.7];

  ctx.strokeStyle = "rgba(88, 166, 255, 0.12)";
  ctx.lineWidth = 1;
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

  outputY.forEach((y, index) => {
    const value = outputs[index] || 0;
    ctx.fillStyle = index === snapshot.lastResult?.prediction ? "#37d18f" : "#f0b84f";
    ctx.globalAlpha = Math.max(0.35, Math.min(1, value / Math.max(1, Math.max(...outputs))));
    ctx.beginPath();
    ctx.arc(rightX, y, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#d7e5dc";
    ctx.fillText(`${index}`, rightX - 4, y + 32);
  });
}

function drawConfusion(snapshot) {
  const { ctx, width, height } = setupCanvas(els.confusionCanvas);
  clear(ctx, width, height);
  const matrix = snapshot.confusion || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const max = Math.max(1, ...matrix.flat());
  const size = Math.min(width, height - 28) / 3;
  ctx.font = "12px system-ui";
  for (let y = 0; y < 3; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      const value = matrix[y][x] || 0;
      const intensity = value / max;
      ctx.fillStyle = x === y ? `rgba(55, 209, 143, ${0.18 + intensity * 0.75})` : `rgba(255, 107, 107, ${0.12 + intensity * 0.65})`;
      ctx.fillRect(30 + x * size, 12 + y * size, size - 4, size - 4);
      ctx.fillStyle = "#eef5ef";
      ctx.fillText(String(value), 30 + x * size + size * 0.42, 12 + y * size + size * 0.56);
    }
  }
}

function renderEval(snapshot) {
  els.evalList.innerHTML = "";
  for (const item of snapshot.eval || []) {
    const div = document.createElement("div");
    div.className = `eval-item ${item.correct ? "ok" : "bad"}`;
    div.innerHTML = `<strong>${item.pattern} = ${item.target}</strong><span>pred ${item.prediction} | margin ${item.margin}</span>`;
    els.evalList.appendChild(div);
  }
}

function render(snapshot) {
  els.deviceLine.textContent = `${snapshot.running ? "Running" : "Stopped"} on ${snapshot.device}${snapshot.gpuName ? ` (${snapshot.gpuName})` : ""}`;
  els.accuracy.textContent = percent(snapshot.accuracy);
  els.evalAccuracy.textContent = percent(snapshot.evalAccuracy);
  els.trials.textContent = String(snapshot.trials || 0);
  const last = snapshot.lastResult || {};
  els.lastResult.textContent = last.pattern ? `${last.pattern} -> ${last.prediction} (${last.correct ? "ok" : "miss"})` : "-";
  els.architecture.textContent = `${snapshot.architecture.inputs} -> ${snapshot.architecture.hidden} -> ${snapshot.architecture.outputs}`;
  els.speedInput.value = snapshot.episodesPerTick;
  els.speedValue.textContent = snapshot.episodesPerTick;
  renderEval(snapshot);
  drawNetwork(snapshot);
  drawRaster(snapshot);
  drawWeights(snapshot);
  drawConfusion(snapshot);
}

function loop() {
  if (latestSnapshot && latestSnapshot.seq !== renderedSeq) {
    renderedSeq = latestSnapshot.seq;
    render(latestSnapshot);
  }
  requestAnimationFrame(loop);
}

function connectStream() {
  const source = new EventSource("/api/stream?interval=0.05");
  source.onmessage = (event) => {
    latestSnapshot = JSON.parse(event.data);
  };
  source.onerror = () => {
    source.close();
    setTimeout(connectStream, 1200);
  };
}

els.startBtn.addEventListener("click", () => command("start"));
els.stopBtn.addEventListener("click", () => command("stop"));
els.stepBtn.addEventListener("click", () => command("step", { episodes: 1 }));
els.resetBtn.addEventListener("click", () => command("reset"));
els.saveBtn.addEventListener("click", async () => {
  await command("save", { name: `snn-state-${Date.now()}` });
  await refreshStates();
});
els.loadBtn.addEventListener("click", () => {
  if (els.stateSelect.value) command("load", { name: els.stateSelect.value });
});
els.speedInput.addEventListener("input", () => {
  els.speedValue.textContent = els.speedInput.value;
});
els.speedInput.addEventListener("change", () => command("speed", { episodesPerTick: Number(els.speedInput.value) }));

fetch("/api/snapshot").then((res) => res.json()).then((data) => {
  latestSnapshot = data;
});
refreshStates();
connectStream();
loop();
