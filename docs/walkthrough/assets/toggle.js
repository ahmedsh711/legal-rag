// Language toggle: EN / AR / Both. Remembered per browser (convenience only).
(function () {
  const body = document.body;
  const buttons = document.querySelectorAll(".lang-toggle button");
  function apply(lang) {
    body.dataset.lang = lang;
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    try { localStorage.setItem("walkthrough-lang", lang); } catch (_) {}
  }
  let saved = "both";
  try { saved = localStorage.getItem("walkthrough-lang") || "both"; } catch (_) {}
  apply(saved);
  buttons.forEach((b) => b.addEventListener("click", () => apply(b.dataset.lang)));
})();
