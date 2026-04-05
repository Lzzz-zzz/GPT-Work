const layers = Array.from(document.querySelectorAll(".parallax-layer"));
const cursorGlow = document.querySelector(".cursor-glow");
const liveMetrics = Array.from(document.querySelectorAll(".live-metric"));
const revealItems = Array.from(document.querySelectorAll(".reveal"));
const tiltCards = Array.from(document.querySelectorAll(".tilt-card"));
const baseMap = document.getElementById("baseMap");
const localTime = document.getElementById("localTime");
const terminalFeed = document.getElementById("terminalFeed");
const canvas = document.getElementById("particles");

let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
let scrollY = window.scrollY;

const mapLayout = [
  "....EEE.....",
  "..SSEEEE..P.",
  ".SSSCCEE.PP.",
  ".SCCCLLLPPP.",
  "EELLLMLLPP..",
  "EE.LMMMMLL..",
  "EE.LMMMMLL..",
  ".SLLLMLL..P.",
  ".SSSLLL..PP.",
  "..SS....P..."
];

const mapTypes = {
  E: "energy",
  S: "storage",
  M: "magic",
  P: "portal",
  C: "core",
  L: "link",
  ".": ""
};

const terminalMessages = [
  "ENERGY MATRIX / BALANCED",
  "LASER BUS / ROUTED",
  "MANA CORE / CHARGED",
  "END RELAY / STABLE",
  "PORTAL GRID / SYNCED",
  "CRAFT CPU / RUNNING",
  "SHIELD DOME / ACTIVE",
  "FLUX NET / CLEAN"
];

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function buildMap() {
  if (!baseMap) {
    return;
  }

  mapLayout.forEach((row) => {
    row.split("").forEach((cell) => {
      const tile = document.createElement("div");
      tile.className = `map-cell ${mapTypes[cell] || ""}`.trim();
      baseMap.appendChild(tile);
    });
  });
}

function updateTime() {
  if (!localTime) {
    return;
  }

  localTime.textContent = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(new Date());
}

function updateMetrics() {
  liveMetrics.forEach((metric) => {
    const base = Number(metric.dataset.base || 0);
    const variance = Number(metric.dataset.variance || 0);
    const decimals = Number(metric.dataset.decimals || 0);
    const delta = (Math.random() * 2 - 1) * variance;
    const value = Math.max(0, base + delta);
    metric.textContent = value.toFixed(decimals);
  });
}

function animateValue(element) {
  const target = Number(element.dataset.target || 0);
  const decimals = Number.isInteger(target) ? 0 : 1;
  const duration = 1800;
  let start = null;

  function tick(timestamp) {
    if (!start) {
      start = timestamp;
    }

    const progress = Math.min((timestamp - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    element.textContent = value.toFixed(decimals);

    if (progress < 1) {
      window.requestAnimationFrame(tick);
    }
  }

  window.requestAnimationFrame(tick);
}

function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      entry.target.classList.add("is-visible");

      if (entry.target.dataset.animated === "1") {
        return;
      }

      entry.target.dataset.animated = "1";
      entry.target.querySelectorAll(".live-number").forEach(animateValue);
    });
  }, {
    threshold: 0.18
  });

  revealItems.forEach((item) => observer.observe(item));
}

function setParallax() {
  layers.forEach((layer) => {
    const depth = Number(layer.dataset.depth || 0);
    const offsetX = ((mouseX / window.innerWidth) - 0.5) * depth * -80;
    const offsetY = ((mouseY / window.innerHeight) - 0.5) * depth * -60 + scrollY * depth * -0.08;
    layer.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0) scale(1.06)`;
  });
}

function initParallax() {
  if (prefersReducedMotion) {
    if (cursorGlow) {
      cursorGlow.style.opacity = "0";
    }
    return;
  }

  if (cursorGlow) {
    cursorGlow.style.left = `${mouseX}px`;
    cursorGlow.style.top = `${mouseY}px`;
  }

  document.addEventListener("mousemove", (event) => {
    mouseX = event.clientX;
    mouseY = event.clientY;

    if (cursorGlow) {
      cursorGlow.style.left = `${mouseX}px`;
      cursorGlow.style.top = `${mouseY}px`;
    }

    setParallax();
  });

  document.addEventListener("scroll", () => {
    scrollY = window.scrollY;
    setParallax();
  }, { passive: true });

  window.addEventListener("resize", setParallax);
  setParallax();
}

function initTilt() {
  if (prefersReducedMotion) {
    return;
  }

  tiltCards.forEach((card) => {
    card.addEventListener("mousemove", (event) => {
      const rect = card.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      const rotateY = (x - 0.5) * 9;
      const rotateX = (0.5 - y) * 9;

      card.style.transform = `perspective(1200px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
      card.style.setProperty("--mx", `${x * 100}%`);
      card.style.setProperty("--my", `${y * 100}%`);
    });

    card.addEventListener("mouseleave", () => {
      card.style.transform = "";
      card.style.setProperty("--mx", "50%");
      card.style.setProperty("--my", "50%");
    });
  });
}

function pushTerminalMessage(message) {
  if (!terminalFeed) {
    return;
  }

  const line = document.createElement("div");
  line.className = "terminal-line";

  const stamp = document.createElement("span");
  stamp.className = "stamp";
  stamp.textContent = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(new Date());

  const content = document.createElement("span");
  content.textContent = message;

  line.append(stamp, content);
  terminalFeed.prepend(line);

  while (terminalFeed.children.length > 7) {
    terminalFeed.removeChild(terminalFeed.lastElementChild);
  }
}

function initTerminal() {
  if (!terminalFeed || prefersReducedMotion) {
    return;
  }

  let index = 0;
  window.setInterval(() => {
    pushTerminalMessage(terminalMessages[index % terminalMessages.length]);
    index += 1;
  }, 2400);
}

function initParticles() {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  let width = window.innerWidth;
  let height = window.innerHeight;
  let particles = [];
  const particleCount = Math.min(90, Math.max(48, Math.floor(window.innerWidth / 22)));

  function createParticles() {
    particles = Array.from({ length: particleCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.32,
      vy: -(Math.random() * 0.35 + 0.08),
      r: Math.random() * 1.8 + 0.7,
      hue: Math.random() > 0.7 ? 285 : 190,
      alpha: Math.random() * 0.34 + 0.18
    }));
  }

  function resizeCanvas() {
    width = window.innerWidth;
    height = window.innerHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    createParticles();
  }

  function draw() {
    context.clearRect(0, 0, width, height);

    particles.forEach((particle, index) => {
      particle.x += particle.vx;
      particle.y += particle.vy;

      if (particle.y < -12) {
        particle.y = height + 12;
        particle.x = Math.random() * width;
      }

      if (particle.x < -12) {
        particle.x = width + 12;
      } else if (particle.x > width + 12) {
        particle.x = -12;
      }

      context.beginPath();
      context.fillStyle = `hsla(${particle.hue}, 100%, 72%, ${particle.alpha})`;
      context.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2);
      context.fill();

      for (let next = index + 1; next < particles.length; next += 1) {
        const other = particles[next];
        const dx = particle.x - other.x;
        const dy = particle.y - other.y;
        const distance = Math.hypot(dx, dy);

        if (distance > 120) {
          continue;
        }

        const opacity = (1 - distance / 120) * 0.12;
        context.strokeStyle = `rgba(89, 239, 255, ${opacity})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(particle.x, particle.y);
        context.lineTo(other.x, other.y);
        context.stroke();
      }
    });

    window.requestAnimationFrame(draw);
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  window.requestAnimationFrame(draw);
}

buildMap();
updateTime();
updateMetrics();
initReveal();
initParallax();
initTilt();
initTerminal();
initParticles();

window.setInterval(updateTime, 1000);
window.setInterval(updateMetrics, 2200);
