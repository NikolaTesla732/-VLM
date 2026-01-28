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
