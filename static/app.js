document.addEventListener("DOMContentLoaded", () => {
  console.log("app.js loaded");

  const input = document.getElementById("video_file");
  const nameEl = document.getElementById("file_name");
  const btn = document.getElementById("file_btn");

  if (!input || !nameEl || !btn) {
    console.warn("Не найдены элементы:", { input, nameEl, btn });
    return;
  }

  function updateUI() {
    const file = input.files && input.files[0];
    if (file) {
      nameEl.textContent = file.name;
      btn.textContent = "Изменить файл";
      btn.classList.add("is-selected");
    } else {
      nameEl.textContent = "Файл не выбран";
      btn.textContent = "Выбрать файл";
      btn.classList.remove("is-selected");
    }
  }

  input.addEventListener("change", updateUI);
  updateUI();
});

async function pollJobStatus(jobId) {
  const statusEl = document.getElementById("status_text");

  async function tick() {
    try {
      const r = await fetch(`/api/status/${jobId}`, { cache: "no-store" });
      if (!r.ok) throw new Error("bad response");

      const data = await r.json();

      if (data.status === "done" || data.status === "error") {
        window.location.href = `/result/${jobId}`;
        return;
      }

      if (statusEl) statusEl.textContent = "Статус: выполняется…";
    } catch (e) {
      if (statusEl) statusEl.textContent = "Статус: нет связи, повтор…";
    }

    setTimeout(tick, 1000);
  }

  tick();
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.querySelector("[data-page='processing']");
  if (!page) return;

  const jobId = page.getAttribute("data-job-id");
  if (!jobId) return;

  pollJobStatus(jobId);
});

function formatTime(sec) {
  sec = Math.max(0, sec || 0);
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);

  const ss = String(s).padStart(2, "0");
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function clamp(n, a, b) {
  return Math.min(b, Math.max(a, n));
}

function initResultPage() {
  const page = document.querySelector("[data-page='result']");
  if (!page) return;

  const status = page.getAttribute("data-status");
  if (status !== "done") return;

  const markersRaw = page.getAttribute("data-markers") || "[]";
  let markers = [];
  try { markers = JSON.parse(markersRaw); } catch { markers = []; }

  // нормализуем: только числа >=0
  markers = markers
    .map(x => Number(x))
    .filter(x => Number.isFinite(x) && x >= 0)
    .sort((a, b) => a - b);

  const video = document.getElementById("result_video");
  const listEl = document.getElementById("tc_list");
  const seek = document.getElementById("video_seek");
  const curEl = document.getElementById("cur_time");
  const durEl = document.getElementById("dur_time");
  const markerLayer = document.getElementById("timeline_markers");

  if (!video || !listEl || !seek || !curEl || !durEl || !markerLayer) return;

  function jumpTo(t) {
    if (!Number.isFinite(t)) return;
    video.currentTime = clamp(t, 0, video.duration || t);
    video.play?.();
  }

  // список таймкодов
  listEl.innerHTML = "";
  if (markers.length === 0) {
    const li = document.createElement("li");
    li.className = "tc-item";
    li.textContent = "Пока нет таймкодов (после обработки они появятся здесь).";
    listEl.appendChild(li);
  } else {
    for (const t of markers) {
      const li = document.createElement("li");
      li.className = "tc-item";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tc-btn";
      btn.addEventListener("click", () => jumpTo(t));

      const timeSpan = document.createElement("span");
      timeSpan.className = "tc-time";
      timeSpan.textContent = formatTime(t);

      const hintSpan = document.createElement("span");
      hintSpan.className = "tc-jump";
      hintSpan.textContent = "перейти ↗";

      btn.appendChild(timeSpan);
      btn.appendChild(hintSpan);

      li.appendChild(btn);
      listEl.appendChild(li);
    }
  }

  function renderMarkers() {
    const dur = video.duration || 0;
    markerLayer.innerHTML = "";

    if (!dur || markers.length === 0) return;

    for (const t of markers) {
      const x = (t / dur) * 100;

      const dot = document.createElement("div");
      dot.className = "timeline-marker";
      dot.style.left = `${x}%`;
      dot.title = formatTime(t);
      dot.addEventListener("click", () => jumpTo(t));

      markerLayer.appendChild(dot);
    }
  }

  function syncUI() {
    const dur = video.duration || 0;
    const cur = video.currentTime || 0;

    durEl.textContent = formatTime(dur);
    curEl.textContent = formatTime(cur);

    if (dur > 0) {
      // делаем range 0..1000 чтобы было плавно
      const v = Math.round((cur / dur) * 1000);
      seek.value = String(clamp(v, 0, 1000));
    } else {
      seek.value = "0";
    }
  }

  video.addEventListener("loadedmetadata", () => {
    renderMarkers();
    syncUI();
  });

  video.addEventListener("timeupdate", syncUI);

  seek.addEventListener("input", () => {
    const dur = video.duration || 0;
    if (!dur) return;

    const v = Number(seek.value);
    const t = (clamp(v, 0, 1000) / 1000) * dur;
    video.currentTime = t;
  });

  // если metadata уже загружена (иногда так бывает)
  if (video.readyState >= 1) {
    renderMarkers();
    syncUI();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initResultPage();
});


function initRoiPage(){
  const page = document.querySelector("[data-page='roi']");
  if (!page) return;

  const video = document.getElementById("roi_video");
  const box = document.getElementById("roi_box");
  const wrap = document.getElementById("roi_wrap");

  const inX = document.getElementById("roi_x");
  const inY = document.getElementById("roi_y");
  const inW = document.getElementById("roi_w");
  const inH = document.getElementById("roi_h");

  const dimTop = document.querySelector(".roi-dim-top");
  const dimLeft = document.querySelector(".roi-dim-left");
  const dimRight = document.querySelector(".roi-dim-right");
  const dimBottom = document.querySelector(".roi-dim-bottom");

  if (!video || !box || !wrap || !inX || !inY || !inW || !inH) return;

  video.controls = true;

  const MIN_W = 40; // px (на экране)
  const MIN_H = 40;

  let mode = null; // "move" or "resize"
  let handle = null; // "nw","n","ne","w","e","sw","s","se"
  let startMx = 0, startMy = 0;
  let start = { left: 0, top: 0, width: 0, height: 0 };

  function clamp(n, a, b){ return Math.max(a, Math.min(b, n)); }

  function getWrapRect(){
    // ROI двигаем по видимой области видео (его bbox)
    return video.getBoundingClientRect();
  }

  function setBoxRect(left, top, width, height){
    const r = getWrapRect();
    left = clamp(left, 0, r.width - width);
    top  = clamp(top, 0, r.height - height);

    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;

    updateDim();
    updateHiddenInputs();
  }

  function updateDim(){
    if (!dimTop) return;
    const r = getWrapRect();
    const b = box.getBoundingClientRect();

    const left = b.left - r.left;
    const top  = b.top - r.top;
    const w = b.width;
    const h = b.height;

    // top
    dimTop.style.left = "0px";
    dimTop.style.top = "0px";
    dimTop.style.width = `${r.width}px`;
    dimTop.style.height = `${top}px`;

    // left
    dimLeft.style.left = "0px";
    dimLeft.style.top = `${top}px`;
    dimLeft.style.width = `${left}px`;
    dimLeft.style.height = `${h}px`;

    // right
    dimRight.style.left = `${left + w}px`;
    dimRight.style.top = `${top}px`;
    dimRight.style.width = `${Math.max(0, r.width - (left + w))}px`;
    dimRight.style.height = `${h}px`;

    // bottom
    dimBottom.style.left = "0px";
    dimBottom.style.top = `${top + h}px`;
    dimBottom.style.width = `${r.width}px`;
    dimBottom.style.height = `${Math.max(0, r.height - (top + h))}px`;
  }

  function updateHiddenInputs(){
    const r = getWrapRect();
    const b = box.getBoundingClientRect();

    const relX = (b.left - r.left) / r.width;
    const relY = (b.top - r.top) / r.height;
    const relW = b.width / r.width;
    const relH = b.height / r.height;

    const vw = video.videoWidth || 0;
    const vh = video.videoHeight || 0;

    const x = Math.round(relX * vw);
    const y = Math.round(relY * vh);
    const w = Math.round(relW * vw);
    const h = Math.round(relH * vh);

    inX.value = String(Math.max(0, x));
    inY.value = String(Math.max(0, y));
    inW.value = String(Math.max(1, w));
    inH.value = String(Math.max(1, h));
  }

  function beginInteraction(e, newMode, newHandle){
    e.preventDefault();
    mode = newMode;
    handle = newHandle || null;
    startMx = e.clientX;
    startMy = e.clientY;
    start.left = box.offsetLeft;
    start.top = box.offsetTop;
    start.width = box.offsetWidth;
    start.height = box.offsetHeight;
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", endInteraction, { once: true });
  }

  function endInteraction(){
    mode = null;
    handle = null;
    window.removeEventListener("mousemove", onMove);
  }

  function onMove(e){
    const r = getWrapRect();
    const dx = e.clientX - startMx;
    const dy = e.clientY - startMy;

    if (mode === "move"){
      setBoxRect(start.left + dx, start.top + dy, start.width, start.height);
      return;
    }

    if (mode === "resize"){
      let left = start.left;
      let top = start.top;
      let width = start.width;
      let height = start.height;

      const right = start.left + start.width;
      const bottom = start.top + start.height;

      // по X
      if (handle.includes("w")){
        left = clamp(start.left + dx, 0, right - MIN_W);
        width = right - left;
      }
      if (handle.includes("e")){
        const newRight = clamp(right + dx, left + MIN_W, r.width);
        width = newRight - left;
      }

      // по Y
      if (handle.includes("n")){
        top = clamp(start.top + dy, 0, bottom - MIN_H);
        height = bottom - top;
      }
      if (handle.includes("s")){
        const newBottom = clamp(bottom + dy, top + MIN_H, r.height);
        height = newBottom - top;
      }

      // финальный clamp чтобы не вылезло
      width = clamp(width, MIN_W, r.width);
      height = clamp(height, MIN_H, r.height);

      // если после clamp ширина/высота упирается — корректируем left/top
      left = clamp(left, 0, r.width - width);
      top = clamp(top, 0, r.height - height);

      setBoxRect(left, top, width, height);
    }
  }

  // Запуск: после metadata можно корректно работать с videoWidth/videoHeight
  video.addEventListener("loadedmetadata", () => {
    const r = getWrapRect();
    const left = Math.round(r.width * 0.10);
    const top = Math.round(r.height * 0.10);
    const width = Math.round(r.width * 0.55);
    const height = Math.round(r.height * 0.55);
    setBoxRect(left, top, width, height);
  });

  // move (за рамку)
  box.addEventListener("mousedown", (e) => {
    // если клик по ручке — это resize (ниже)
    if (e.target && e.target.classList.contains("roi-handle")) return;
    beginInteraction(e, "move", null);
  });

  // resize (за ручки)
  box.querySelectorAll(".roi-handle").forEach(h => {
    h.addEventListener("mousedown", (e) => {
      const key = e.target.getAttribute("data-h");
      beginInteraction(e, "resize", key);
    });
  });

  // если окно меняет размер/масштаб — пересчитать маску и hidden inputs
  window.addEventListener("resize", () => {
    updateDim();
    updateHiddenInputs();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initRoiPage();
});
