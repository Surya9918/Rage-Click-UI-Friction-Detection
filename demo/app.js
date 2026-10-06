/**
 * JavaScript Telemetry Collector & Live FFNN Prediction Client
 */

(function () {
  const WINDOW_SECONDS = 5.0;
  const RAPID_CLICK_INTERVAL_MS = 300.0;
  const DIRECTION_CHANGE_ANGLE = 90.0;
  const MOVEMENT_THRESHOLD_PX = 5.0;
  const SCROLL_WINDOW_SECONDS = 2.0;
  const DEFAULT_THRESHOLD = 0.85;
  const INTERVENTION_COOLDOWN_MS = 30000;

  let eventBuffer = [];
  let windowStartTime = performance.now();
  let lastInterventionTime = 0;
  let isSimulating = false;
  let currentThreshold = DEFAULT_THRESHOLD;

  const windowProgressBar = document.getElementById("window-progress-bar");
  const windowTimerText = document.getElementById("window-timer-text");
  const valClickFreq = document.getElementById("val-click-freq");
  const valRapidClicks = document.getElementById("val-rapid-clicks");
  const valCursorVel = document.getElementById("val-cursor-vel");
  const valDirectionChanges = document.getElementById("val-direction-changes");
  const valScrollThrash = document.getElementById("val-scroll-thrash");
  const probPercentage = document.getElementById("prob-percentage");
  const decisionBadge = document.getElementById("decision-badge");
  const gaugeFill = document.getElementById("gauge-fill");
  const predictionCard = document.getElementById("prediction-card");
  const thresholdValDisplay = document.getElementById("threshold-val-display");
  const eventLogBox = document.getElementById("event-log-box");
  const eventCountBadge = document.getElementById("event-count-badge");
  const interventionModal = document.getElementById("intervention-modal");
  const interventionDismissBtn = document.getElementById("intervention-dismiss-btn");
  const interventionHelpBtn = document.getElementById("intervention-help-btn");

  const impRapid = document.getElementById("imp-rapid");
  const impRapidVal = document.getElementById("imp-rapid-val");
  const impDir = document.getElementById("imp-dir");
  const impDirVal = document.getElementById("imp-dir-val");
  const impFreq = document.getElementById("imp-freq");
  const impFreqVal = document.getElementById("imp-freq-val");
  const impScroll = document.getElementById("imp-scroll");
  const impScrollVal = document.getElementById("imp-scroll-val");
  const impVel = document.getElementById("imp-vel");
  const impVelVal = document.getElementById("imp-vel-val");

  const simRageBtn = document.getElementById("sim-rage-btn");
  const simNormalBtn = document.getElementById("sim-normal-btn");

  function formatTimestamp(ts) {
    const d = new Date(Date.now() - (performance.now() - ts));
    const h = String(d.getHours()).padStart(2, "0");
    const m = String(d.getMinutes()).padStart(2, "0");
    const s = String(d.getSeconds()).padStart(2, "0");
    const ms = String(d.getMilliseconds()).padStart(3, "0");
    return `${h}:${m}:${s}.${ms}`;
  }

  function getElementIdentifier(el) {
    if (!el || el === window || el === document) return "window";
    if (el.id) return `#${el.id}`;
    if (el.dataset && el.dataset.track) return el.dataset.track;
    if (el.className && typeof el.className === "string") {
      const cls = el.className.split(" ")[0];
      if (cls) return `.${cls}`;
    }
    return el.tagName ? el.tagName.toLowerCase() : "element";
  }

  function recordEvent(eventType, x = 0, y = 0, target = "window", scrollY = 0) {
    const now = performance.now();
    const event = {
      timestamp: now,
      event_type: eventType,
      x_coordinate: Math.round(x),
      y_coordinate: Math.round(y),
      target_element: target,
      scroll_y: Math.round(scrollY),
    };

    eventBuffer.push(event);
    appendLogEntry(event);

    if (eventCountBadge) {
      eventCountBadge.textContent = `${eventBuffer.length} events in buffer`;
    }
  }

  function appendLogEntry(event) {
    if (!eventLogBox) return;
    const div = document.createElement("div");
    div.className = `log-entry ${event.event_type}`;
    const timeStr = formatTimestamp(event.timestamp);

    let details = "";
    if (event.event_type === "scroll") {
      details = `scrollY=${event.scroll_y}`;
    } else if (event.event_type.startsWith("mouse")) {
      details = `x=${event.x_coordinate} y=${event.y_coordinate} target=${event.target_element}`;
    }

    div.textContent = `${timeStr}  ${event.event_type.padEnd(9)}  ${details}`;
    eventLogBox.appendChild(div);

    while (eventLogBox.childNodes.length > 35) {
      eventLogBox.removeChild(eventLogBox.firstChild);
    }
    eventLogBox.scrollTop = eventLogBox.scrollHeight;
  }

  function extractFeaturesFromBuffer(events, duration) {
    if (!events || events.length === 0) {
      return {
        click_frequency: 0.0,
        rapid_fire_clicks: 0,
        maximum_cursor_velocity: 0.0,
        erratic_direction_changes: 0,
        scroll_thrashing: 0.0,
      };
    }

    const clicks = events.filter((e) => e.event_type === "mousedown" || e.event_type === "click");
    const click_frequency = clicks.length / duration;

    let rapid_fire_clicks = 0;
    for (let i = 1; i < clicks.length; i++) {
      const prev = clicks[i - 1];
      const curr = clicks[i];
      const dt = curr.timestamp - prev.timestamp;
      if (curr.target_element && curr.target_element === prev.target_element && dt <= RAPID_CLICK_INTERVAL_MS) {
        rapid_fire_clicks++;
      }
    }

    const moves = events.filter((e) => e.event_type === "mousemove" || e.event_type === "mousedown");
    let maximum_cursor_velocity = 0.0;
    for (let i = 1; i < moves.length; i++) {
      const p1 = moves[i - 1];
      const p2 = moves[i];
      const dt = (p2.timestamp - p1.timestamp) / 1000.0;
      if (dt > 0.001) {
        const dist = Math.hypot(p2.x_coordinate - p1.x_coordinate, p2.y_coordinate - p1.y_coordinate);
        let vel = dist / dt;
        if (vel > 10000.0) vel = 10000.0;
        if (vel > maximum_cursor_velocity) maximum_cursor_velocity = vel;
      }
    }

    let erratic_direction_changes = 0;
    const vectors = [];
    if (moves.length >= 2) {
      let lastX = moves[0].x_coordinate;
      let lastY = moves[0].y_coordinate;
      for (let i = 1; i < moves.length; i++) {
        const dx = moves[i].x_coordinate - lastX;
        const dy = moves[i].y_coordinate - lastY;
        const dist = Math.hypot(dx, dy);
        if (dist >= MOVEMENT_THRESHOLD_PX) {
          vectors.push({ dx, dy, dist });
          lastX = moves[i].x_coordinate;
          lastY = moves[i].y_coordinate;
        }
      }
    }

    for (let i = 1; i < vectors.length; i++) {
      const v1 = vectors[i - 1];
      const v2 = vectors[i];
      const dot = v1.dx * v2.dx + v1.dy * v2.dy;
      let cosTheta = dot / (v1.dist * v2.dist);
      cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta));
      const angle = (Math.acos(cosTheta) * 180.0) / Math.PI;
      if (angle > DIRECTION_CHANGE_ANGLE) {
        erratic_direction_changes++;
      }
    }

    const scrolls = events.filter((e) => e.event_type === "scroll");
    let scroll_thrashing = 0.0;
    const scrollDeltas = [];
    for (let i = 1; i < scrolls.length; i++) {
      const dy = scrolls[i].scroll_y - scrolls[i - 1].scroll_y;
      if (Math.abs(dy) > 0.1) {
        scrollDeltas.push({ t: scrolls[i].timestamp / 1000.0, dy });
      }
    }

    for (let i = 1; i < scrollDeltas.length; i++) {
      const curr = scrollDeltas[i];
      const prev = scrollDeltas[i - 1];
      if (curr.dy * prev.dy < 0) {
        if (Math.abs(curr.t - prev.t) <= SCROLL_WINDOW_SECONDS) {
          scroll_thrashing += Math.abs(curr.dy);
        }
      }
    }

    return {
      click_frequency: parseFloat(click_frequency.toFixed(2)),
      rapid_fire_clicks: rapid_fire_clicks,
      maximum_cursor_velocity: parseFloat(maximum_cursor_velocity.toFixed(1)),
      erratic_direction_changes: erratic_direction_changes,
      scroll_thrashing: parseFloat(scroll_thrashing.toFixed(1)),
    };
  }

  function updateDashboardFeatures(features) {
    if (valClickFreq) valClickFreq.innerHTML = `${features.click_frequency.toFixed(2)} <span class="unit">clicks/s</span>`;
    if (valRapidClicks) valRapidClicks.innerHTML = `${features.rapid_fire_clicks} <span class="unit">events</span>`;
    if (valCursorVel) valCursorVel.innerHTML = `${Math.round(features.maximum_cursor_velocity)} <span class="unit">px/s</span>`;
    if (valDirectionChanges) valDirectionChanges.innerHTML = `${features.erratic_direction_changes} <span class="unit">turns</span>`;
    if (valScrollThrash) valScrollThrash.innerHTML = `${Math.round(features.scroll_thrashing)} <span class="unit">px rev</span>`;
  }

  function updatePredictionUI(result) {
    const prob = result.friction_probability;
    const isFriction = result.friction_detected;
    const pctStr = (prob * 100).toFixed(1) + "%";

    probPercentage.textContent = pctStr;
    gaugeFill.style.width = Math.min(100, Math.max(0, prob * 100)) + "%";

    if (isFriction) {
      decisionBadge.className = "decision-badge badge-friction";
      decisionBadge.textContent = "⚠ UI FRICTION DETECTED";
      predictionCard.classList.add("friction-active");
      triggerIntervention();
    } else {
      decisionBadge.className = "decision-badge badge-normal";
      decisionBadge.textContent = "NORMAL";
      predictionCard.classList.remove("friction-active");
    }

    if (result.feature_importance) {
      updateSensitivityBars(result.feature_importance);
    }
  }

  function updateSensitivityBars(imp) {
    const rf = Math.round((imp.rapid_fire_clicks || 0.2) * 100);
    const ed = Math.round((imp.erratic_direction_changes || 0.2) * 100);
    const cf = Math.round((imp.click_frequency || 0.2) * 100);
    const st = Math.round((imp.scroll_thrashing || 0.2) * 100);
    const cv = Math.round((imp.maximum_cursor_velocity || 0.2) * 100);

    if (impRapid) impRapid.style.width = `${rf}%`;
    if (impRapidVal) impRapidVal.textContent = `${rf}%`;
    if (impDir) impDir.style.width = `${ed}%`;
    if (impDirVal) impDirVal.textContent = `${ed}%`;
    if (impFreq) impFreq.style.width = `${cf}%`;
    if (impFreqVal) impFreqVal.textContent = `${cf}%`;
    if (impScroll) impScroll.style.width = `${st}%`;
    if (impScrollVal) impScrollVal.textContent = `${st}%`;
    if (impVel) impVel.style.width = `${cv}%`;
    if (impVelVal) impVelVal.textContent = `${cv}%`;
  }

  function triggerIntervention() {
    const now = Date.now();
    if (now - lastInterventionTime > INTERVENTION_COOLDOWN_MS) {
      lastInterventionTime = now;
      if (interventionModal) {
        interventionModal.classList.add("visible");
      }
    }
  }

  function hideIntervention() {
    if (interventionModal) {
      interventionModal.classList.remove("visible");
    }
  }

  if (interventionDismissBtn) interventionDismissBtn.addEventListener("click", hideIntervention);
  if (interventionHelpBtn) {
    interventionHelpBtn.addEventListener("click", () => {
      alert("Help Guide: Let's locate the right serverless package or feature for your workflow!");
      hideIntervention();
    });
  }

  async function sendWindowToBackend(features) {
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      currentThreshold = result.threshold || DEFAULT_THRESHOLD;
      if (thresholdValDisplay) thresholdValDisplay.textContent = currentThreshold.toFixed(2);
      updatePredictionUI(result);
    } catch (err) {
      console.warn("Backend /predict call failed or server starting:", err);
    }
  }

  function completeObservationWindow() {
    const windowEvents = [...eventBuffer];
    eventBuffer = [];
    windowStartTime = performance.now();

    const features = extractFeaturesFromBuffer(windowEvents, WINDOW_SECONDS);
    updateDashboardFeatures(features);
    sendWindowToBackend(features);
  }

  function tickWindowProgress() {
    const elapsedMs = performance.now() - windowStartTime;
    const progress = Math.min(1.0, elapsedMs / (WINDOW_SECONDS * 1000.0));
    const remainingSeconds = Math.max(0.0, WINDOW_SECONDS - elapsedMs / 1000.0).toFixed(1);

    if (windowProgressBar) windowProgressBar.style.width = `${progress * 100}%`;
    if (windowTimerText) windowTimerText.textContent = `Next prediction in ${remainingSeconds}s`;

    if (progress >= 1.0) {
      completeObservationWindow();
    }

    requestAnimationFrame(tickWindowProgress);
  }

  const scrollableContent = document.getElementById("scrollable-content") || window;

  document.addEventListener("mousemove", (e) => {
    recordEvent("mousemove", e.clientX, e.clientY, getElementIdentifier(e.target), scrollableContent.scrollTop || window.scrollY || 0);
  });

  document.addEventListener("mousedown", (e) => {
    recordEvent("mousedown", e.clientX, e.clientY, getElementIdentifier(e.target), scrollableContent.scrollTop || window.scrollY || 0);
  });

  document.addEventListener("mouseup", (e) => {
    recordEvent("mouseup", e.clientX, e.clientY, getElementIdentifier(e.target), scrollableContent.scrollTop || window.scrollY || 0);
  });

  scrollableContent.addEventListener("scroll", (e) => {
    const scrollY = scrollableContent.scrollTop !== undefined ? scrollableContent.scrollTop : window.scrollY;
    recordEvent("scroll", 0, 0, "window", scrollY);
  });

  document.querySelectorAll(".interactive-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      btn.style.transform = "scale(0.95)";
      setTimeout(() => (btn.style.transform = ""), 100);
    });
  });

  if (simRageBtn) {
    simRageBtn.addEventListener("click", () => {
      if (isSimulating) return;
      isSimulating = true;
      simRageBtn.disabled = true;

      appendLogEntry({
        timestamp: performance.now(),
        event_type: "system",
        target_element: "SIMULATION",
        x_coordinate: 0,
        y_coordinate: 0,
        scroll_y: 0,
      });

      let clickCount = 0;
      const totalClicks = 8;
      const targetId = "#buy-now-btn";
      const buyBtn = document.getElementById("buy-now-btn");
      const rect = buyBtn ? buyBtn.getBoundingClientRect() : { left: 400, top: 300 };

      const interval = setInterval(() => {
        clickCount++;
        const jx = rect.left + 20 + (Math.random() * 20 - 10);
        const jy = rect.top + 15 + (Math.random() * 20 - 10);

        recordEvent("mousedown", jx, jy, targetId, scrollableContent.scrollTop || 0);
        recordEvent("mouseup", jx, jy, targetId, scrollableContent.scrollTop || 0);

        recordEvent("mousemove", jx + 80, jy - 50, targetId, scrollableContent.scrollTop || 0);
        recordEvent("mousemove", jx - 60, jy + 70, targetId, scrollableContent.scrollTop || 0);

        if (clickCount >= totalClicks) {
          clearInterval(interval);
          const currScroll = scrollableContent.scrollTop || 0;
          recordEvent("scroll", 0, 0, "window", currScroll + 400);
          recordEvent("scroll", 0, 0, "window", currScroll - 350);
          recordEvent("scroll", 0, 0, "window", currScroll + 450);

          setTimeout(() => {
            isSimulating = false;
            simRageBtn.disabled = false;
            completeObservationWindow();
          }, 300);
        }
      }, 140);
    });
  }

  if (simNormalBtn) {
    simNormalBtn.addEventListener("click", () => {
      if (isSimulating) return;
      isSimulating = true;
      simNormalBtn.disabled = true;

      const steps = 15;
      let step = 0;
      const startX = 200, startY = 200, endX = 600, endY = 400;

      const interval = setInterval(() => {
        step++;
        const ratio = step / steps;
        const curX = (1 - ratio) * startX + ratio * endX;
        const curY = (1 - ratio) * startY + ratio * endY;

        recordEvent("mousemove", curX, curY, "body", 0);

        if (step === 8) {
          recordEvent("mousedown", curX, curY, "#nav-products", 0);
          recordEvent("mouseup", curX, curY, "#nav-products", 0);
        }

        if (step >= steps) {
          clearInterval(interval);
          setTimeout(() => {
            isSimulating = false;
            simNormalBtn.disabled = false;
            completeObservationWindow();
          }, 400);
        }
      }, 150);
    });
  }

  requestAnimationFrame(tickWindowProgress);
})();
