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
