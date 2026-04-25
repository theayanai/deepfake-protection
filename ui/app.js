(() => {
    document.documentElement.classList.add("deepfake-app-ready");

    const selectRevealNodes = () =>
        document.querySelectorAll(
            ".hero-card, .glass-panel, .dashboard-shell, .metric-card, .signal-block, .analysis-panel, .stat-box, .flow-step, .trace-chip"
        );

    /* ========================= */
    /* 🌊 REVEAL SYSTEM */
    /* ========================= */

    const applyRevealClasses = () => {
        const nodes = selectRevealNodes();
        nodes.forEach((node, index) => {
            node.classList.add("reveal");
            node.classList.add(`stagger-${(index % 4) + 1}`);
        });
    };

    const setupRevealObserver = () => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("is-visible");
                    }
                });
            },
            { threshold: 0.2, rootMargin: "0px 0px -8% 0px" }
        );

        selectRevealNodes().forEach((node) => observer.observe(node));
    };

    /* ========================= */
    /* 🧊 HERO TILT */
    /* ========================= */

    const setupHeroTilt = () => {
        const hero = document.querySelector(".hero-card");
        if (!hero) return;

        const clamp = (n, min, max) => Math.min(Math.max(n, min), max);

        hero.addEventListener("mousemove", (event) => {
            const rect = hero.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;

            const tiltY = clamp((x - 0.5) * 10, -5, 5);
            const tiltX = clamp((0.5 - y) * 8, -4, 4);

            hero.style.setProperty("--tilt-x", `${tiltX}deg`);
            hero.style.setProperty("--tilt-y", `${tiltY}deg`);
        });

        hero.addEventListener("mouseleave", () => {
            hero.style.setProperty("--tilt-x", "0deg");
            hero.style.setProperty("--tilt-y", "0deg");
        });
    };

    /* ========================= */
    /* 🤖 CURSOR + POINTER SYSTEM */
    /* ========================= */

    const cursor = document.querySelector(".ai-cursor");
    const trail = document.querySelector(".ai-cursor-trail");

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;

    let trailX = mouseX;
    let trailY = mouseY;

    const setupPointerSystem = () => {
        document.addEventListener("mousemove", (event) => {
            mouseX = event.clientX;
            mouseY = event.clientY;

            const cursor = document.querySelector(".ai-cursor");
            if (cursor) {
                cursor.style.left = event.clientX + "px";
                cursor.style.top = event.clientY + "px";
            }

            // update background glow (your existing system)
            const root = document.documentElement;
            root.style.setProperty("--pointer-x", ((mouseX / window.innerWidth) * 100).toFixed(2));
            root.style.setProperty("--pointer-y", ((mouseY / window.innerHeight) * 100).toFixed(2));
        });
    };

    const animateCursor = () => {
        if (cursor) {
            cursor.style.left = mouseX + "px";
            cursor.style.top = mouseY + "px";
        }

        if (trail) {
            trailX += (mouseX - trailX) * 0.15;
            trailY += (mouseY - trailY) * 0.15;

            trail.style.left = trailX + "px";
            trail.style.top = trailY + "px";
        }

        requestAnimationFrame(animateCursor);
    };

    /* ========================= */
    /* 🚨 DANGER MODE CONTROL */
    /* ========================= */

    window.enableDangerMode = () => {
        document.body.classList.add("danger-mode");
    };

    window.disableDangerMode = () => {
        document.body.classList.remove("danger-mode");
    };

    const setupLoadingState = () => {
        const buttons = Array.from(document.querySelectorAll("button"));
        const launchButton = buttons.find((btn) => btn.textContent && btn.textContent.includes("Launch Deepfake Sweep"));
        if (!launchButton) return;

        launchButton.onclick = () => {
            document.body.style.cursor = "wait";
            setTimeout(() => {
                document.body.style.cursor = "default";
            }, 2000);
        };
    };

    /* ========================= */
    /* 🚀 INIT */
    /* ========================= */

    applyRevealClasses();
    setupRevealObserver();
    setupHeroTilt();
    setupPointerSystem();
    setupLoadingState();
    animateCursor();
})();