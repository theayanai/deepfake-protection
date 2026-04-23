() => {
    document.documentElement.classList.add("deepfake-app-ready");

    const selectRevealNodes = () =>
        document.querySelectorAll(
            ".hero-card, .glass-panel, .dashboard-shell, .metric-card, .signal-block, .analysis-panel, .stat-box, .flow-step, .trace-chip"
        );

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

    const setupPointerGlow = () => {
        document.addEventListener("mousemove", (event) => {
            const root = document.documentElement;
            const x = (event.clientX / window.innerWidth) * 100;
            const y = (event.clientY / window.innerHeight) * 100;
            root.style.setProperty("--pointer-x", x.toFixed(2));
            root.style.setProperty("--pointer-y", y.toFixed(2));
        });
    };

    applyRevealClasses();
    setupRevealObserver();
    setupHeroTilt();
    setupPointerGlow();
}
