"use strict";
(() => {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const counter = document.getElementById("counter");
  const notesPanel = document.getElementById("notes-panel");
  const notesContent = document.getElementById("notes-content");
  const notesToggle = document.getElementById("notes-toggle");
  let current = 0;

  function scaleDeck() {
    const area = document.querySelector(".viewport");
    const scale = Math.min(area.clientWidth / 1600, area.clientHeight / 900);
    document.documentElement.style.setProperty("--deck-scale", String(Math.max(scale, 0.05)));
  }

  function show(index, updateHash = true) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      slide.classList.toggle("active", i === current);
      slide.setAttribute("aria-hidden", String(i !== current));
    });
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    counter.textContent = `${String(current + 1).padStart(2, "0")} / ${slides.length}`;
    notesContent.textContent = slides[current].querySelector(".notes")?.textContent.trim() || "";
    if (updateHash) history.replaceState(null, "", `#slide-${current + 1}`);
  }

  function toggleNotes() {
    notesPanel.hidden = !notesPanel.hidden;
    notesToggle.setAttribute("aria-pressed", String(!notesPanel.hidden));
  }

  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    } catch {
      // 브라우저에서 전체 화면을 허용하지 않아도 발표 이동은 계속 사용할 수 있습니다.
    }
  }

  previous.addEventListener("click", () => show(current - 1));
  next.addEventListener("click", () => show(current + 1));
  notesToggle.addEventListener("click", toggleNotes);
  document.getElementById("fullscreen").addEventListener("click", fullscreen);
  document.getElementById("print").addEventListener("click", () => window.print());
  document.addEventListener("keydown", event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === " " && event.target.closest?.("button")) return;
    const handlers = {
      ArrowRight: () => show(current + 1), ArrowDown: () => show(current + 1),
      PageDown: () => show(current + 1), " ": () => show(current + 1),
      ArrowLeft: () => show(current - 1), ArrowUp: () => show(current - 1),
      PageUp: () => show(current - 1), Home: () => show(0), End: () => show(slides.length - 1),
      n: toggleNotes, N: toggleNotes, f: fullscreen, F: fullscreen
    };
    if (handlers[event.key]) { event.preventDefault(); handlers[event.key](); }
  });
  window.addEventListener("resize", scaleDeck);
  window.addEventListener("hashchange", () => {
    const match = location.hash.match(/^#slide-(\d+)$/);
    if (match) show(Number(match[1]) - 1, false);
  });
  const initial = location.hash.match(/^#slide-(\d+)$/);
  show(initial ? Number(initial[1]) - 1 : 0, false);
  scaleDeck();
})();
