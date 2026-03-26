(() => {
  const forceLight = () => {
    try {
      localStorage.setItem("theme", "light");
    } catch (_err) {
      // Ignore storage failures (private mode / blocked storage).
    }
    document.documentElement.dataset.theme = "light";
    if (document.body) {
      document.body.dataset.theme = "light";
    }
  };

  forceLight();

  document.addEventListener("DOMContentLoaded", forceLight, { once: true });

  const observer = new MutationObserver(() => {
    if (document.body && document.body.dataset.theme !== "light") {
      forceLight();
    }
  });

  document.addEventListener(
    "DOMContentLoaded",
    () => {
      if (document.body) {
        observer.observe(document.body, {
          attributes: true,
          attributeFilter: ["data-theme"],
        });
      }
    },
    { once: true }
  );
})();
