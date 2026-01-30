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
