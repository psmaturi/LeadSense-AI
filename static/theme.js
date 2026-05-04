function toggleTheme() {
    const body = document.body;
    const btn = document.getElementById("themeBtn");

    body.classList.toggle("light-mode");

    if (body.classList.contains("light-mode")) {
        localStorage.setItem("theme", "light");
        btn.innerText = "☀️ Light";
    } else {
        localStorage.setItem("theme", "dark");
        btn.innerText = "🌙 Dark";
    }
}

// Load saved theme
window.addEventListener("DOMContentLoaded", () => {
    const saved = localStorage.getItem("theme");
    const btn = document.getElementById("themeBtn");

    if (saved === "light") {
        document.body.classList.add("light-mode");
        if (btn) btn.innerText = "☀️ Light";
    } else {
        if (btn) btn.innerText = "🌙 Dark";
    }
});