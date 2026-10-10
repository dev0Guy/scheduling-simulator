// Builds docs/presentations/experiments_deck.pptx: one slide per experiment, then a summary table.
// Run: NODE_PATH=<dir with pptxgenjs> node scripts/docs_tools/make_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/Users/guyarieli/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/7c34bc6a-bd8c-4259-baee-0a9a41b396da/7107bde6-8620-4827-8d45-5bd494cff7fa/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const OUT = path.join(__dirname, "..", "..", "docs", "presentations", "experiments_deck.pptx");

const THEME = {
  name: "Scheduling Experiments",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2330", lt1: "FFFFFF", dk2: "14213D", lt2: "F1F5F9",
    accent1: "1C7293", accent2: "2E9E5B", accent3: "E0A100",
    accent4: "C0392B", accent5: "8FB3E8", accent6: "5C6B7A",
    hlink: "1C7293", folHlink: "5C6B7A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "PPO vs SJF: scheduling experiments";
const C = pres.SchemeColor;

// ---- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: "14213D" },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9, h: 0.75, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.25, w: 0.5, h: 0.25, fontFace: "Calibri", fontSize: 10, color: C.accent6, align: "right" },
});

// ---- helpers
const NEW_FILL = "DDF3E6", NEW_LINE = C.accent2, OLD_FILL = "EAF1FB", OLD_LINE = C.accent5;

function stack(slide, rows, x, y, w) {
  const pitch = 0.5, h = 0.42;
  rows.forEach((r, i) => {
    const yy = y + i * pitch;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: yy, w, h, rectRadius: 0.07,
      fill: { color: r.isNew ? NEW_FILL : OLD_FILL },
      line: { color: r.isNew ? NEW_LINE : OLD_LINE, width: r.isNew ? 2 : 1 },
      objectName: `stack-box-${i + 1}`,
    });
    slide.addText(
      [
        { text: r.title, options: { bold: true, fontSize: 11, color: C.text2, breakLine: true } },
        { text: r.sub, options: { fontSize: 10, color: C.accent6 } },
      ],
      { x: x + 0.1, y: yy, w: w - 0.2, h, margin: 0, valign: "middle", isTextBox: true, objectName: `stack-text-${i + 1}` }
    );
    if (i < rows.length - 1) {
      slide.addShape(pres.shapes.ISOSCELES_TRIANGLE, {
        x: x + w / 2 - 0.07, y: yy + h + 0.005, w: 0.14, h: 0.07, rotate: 180,
        fill: { color: C.accent6 }, line: { color: C.accent6, width: 0.5 }, objectName: `stack-arrow-${i + 1}`,
      });
    }
  });
}

function legend(slide) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 5.2, w: 0.2, h: 0.15, rectRadius: 0.03, fill: { color: NEW_FILL }, line: { color: NEW_LINE, width: 1.5 }, objectName: "legend-new" });
  slide.addText("changed in this model", { x: 0.75, y: 5.15, w: 1.8, h: 0.25, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.6, y: 5.2, w: 0.2, h: 0.15, rectRadius: 0.03, fill: { color: OLD_FILL }, line: { color: OLD_LINE, width: 1 }, objectName: "legend-old" });
  slide.addText("carried over", { x: 2.85, y: 5.15, w: 1.5, h: 0.25, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
}

function statCard(slide, x, y, w, h, o) {
  const good = o.margin > 0;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.1, fill: { color: "F1F5F9" }, line: { color: good ? C.accent2 : C.accent4, width: 1.5 },
    shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: "result-card",
  });
  slide.addText(`${o.margin > 0 ? "+" : "−"}${Math.abs(o.margin).toFixed(2)}%`, {
    x: x + 0.15, y: y + 0.08, w: 1.9, h: 0.6, fontFace: "Cambria", fontSize: 32, bold: true,
    color: good ? C.accent2 : C.accent4, margin: 0, valign: "middle", isTextBox: true, objectName: "result-margin",
  });
  slide.addText("vs SJF (mean reward)", { x: x + 0.15, y: y + 0.66, w: 1.9, h: 0.25, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
  slide.addText(
    [
      { text: `reward ${o.reward}   wait ${o.wait}   completion ${o.comp}`, options: { breakLine: true } },
      { text: o.wins },
    ],
    { x: x + 2.05, y: y + 0.1, w: w - 2.15, h: 0.85, fontSize: 11, color: C.text1, margin: 0, valign: "middle", isTextBox: true, paraSpaceAfter: 3, objectName: "result-details" }
  );
  if (o.note) {
    slide.addText(o.note, { x: x + 0.15, y: y + 1.0, w: w - 0.3, h: h - 1.05, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "result-note" });
  }
}

function rightPanel(slide, o) {
  slide.addText(o.changeLabel || "What changed", { x: 5.3, y: 1.2, w: 4.2, h: 0.25, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  slide.addText(o.change, { x: 5.3, y: 1.45, w: 4.2, h: 0.85, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "change-text" });
  slide.addText("What we saw", { x: 5.3, y: 2.35, w: 4.2, h: 0.25, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  slide.addText(o.finding, { x: 5.3, y: 2.6, w: 4.2, h: 0.85, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "finding-text" });
  statCard(slide, 5.3, 3.55, 4.2, 1.55, o.result);
}

const R = (title, sub, isNew) => ({ title, sub, isNew: !!isNew });
const ENC = R("Job encoder", "20 → 256 → 256, shared across jobs");
const POOL = R("Attention pooling", "weighted sum of jobs → global summary");
const SCORE = R("Scorer per job", "[global | machine | job] 768 → 128 → 1");
const LOGITS = R("11 action logits", "skip + one per job, softmax");
const IN20 = R("Per-job input (20)", "status, size, wait, ttl, arrival (no usage)");
const FIT = (n) => R("Fit attention" + (n ? " + mask" : ""), n ? "weight = 0 where job usage > machine room" : "Q: job usage, K/V: machine room, sigmoid", n);
const CONCAT = R("Concat + project", "[summary 256 | fit 64] → 256");
const SELF = (n) => R("Job self-attention", "4 heads, finished jobs masked", n);

function contentSlide(section, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}


// ---- E1 experiment-log helpers
const VERDICT_COLOR = { Kept: "2E9E5B", Reverted: "C0392B", Superseded: "5C6B7A", Neutral: "5C6B7A", Stopped: "E0A100", "Not adopted": "C0392B", Ceiling: "1C7293" };
function logTable(slide, rows, colW, y) {
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "14213D" }, align: "left" } });
  const data = [[H("#"), H("Experiment"), H("Result"), H("Verdict")]];
  rows.forEach((r) => {
    data.push([
      { text: r[0], options: { bold: true, align: "center" } },
      { text: r[1], options: { align: "left" } },
      { text: r[2], options: { align: "left" } },
      { text: r[3], options: { bold: true, color: VERDICT_COLOR[r[3]] || "1B2330", align: "center" } },
    ]);
  });
  slide.addTable(data, { x: 0.5, y: y || 1.15, w: 9, colW, fontSize: 11, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08], objectName: "log-table" });
}
function logSlide(section, title, rows, note, notes) {
  const s = contentSlide(section, title, notes);
  logTable(s, rows, [0.5, 2.9, 4.5, 1.1]);
  if (note) s.addText(note, { x: 0.5, y: 4.9, w: 9, h: 0.4, fontSize: 11, color: C.accent6, margin: 0, isTextBox: true, objectName: "log-note" });
  return s;
}

// ================= 1. title
pres.addSection({ title: "Intro" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Intro" });
  s.addText("Teaching PPO to schedule like SJF", { x: 0.7, y: 1.0, w: 8.6, h: 1.9, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, margin: 0, valign: "bottom", isTextBox: true, objectName: "deck-title" });
  s.addText("One experiment per slide: what changed, how the network looks, and what it scored", { x: 0.7, y: 3.1, w: 8.2, h: 0.9, fontSize: 20, color: "CADCFC", margin: 0, valign: "top", isTextBox: true, objectName: "deck-subtitle" });
  s.addText("Cluster scheduling simulator · models M0 to M10 · runs from Oct 2 to Oct 6", { x: 0.7, y: 4.6, w: 8.6, h: 0.35, fontSize: 14, color: "8FB3E8", margin: 0, isTextBox: true, objectName: "deck-footer" });
  s.addNotes("Overview: a PPO agent is compared with Shortest-Job-First (SJF) and an exact optimal solver on a one-machine scheduling task. Each slide adds one change and reports the result.");
}

// ================= 2. setup
{
  const s = contentSlide("Intro", "The setup", "Metric: margin vs SJF = (PPO - SJF)/|SJF| on mean reward over 5000 deterministic episodes.");
  const cards = [
    ["Task", ["1 machine, capacity 255, 3 resources", "10 jobs arrive in the first ~10 ticks", "Each job runs 1 to 10 ticks", "Action: skip, or start one of the 10 jobs"]],
    ["Reward", ["Skip: −1 per active job", "Start a job: +1 (time does not advance)", "Failed start: −10", "All done: +100, timeout: −100"]],
    ["Training and eval", ["PPO, 2M steps, lr 7e-6, batch 64", "Entropy coef 0.02 → 0.002", "Eval: 5000 episodes, seed 42 + episode", "Compare with SJF and exact optimal"]],
  ];
  cards.forEach((c, i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.9, h: 3.6, rectRadius: 0.1, fill: { color: "F1F5F9" }, line: { color: C.accent5, width: 1 }, objectName: `setup-card-${i + 1}` });
    s.addText(c[0], { x: x + 0.2, y: 1.45, w: 2.5, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true });
    s.addText(c[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[1].length - 1 } })), { x: x + 0.2, y: 1.95, w: 2.55, h: 2.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: `setup-bullets-${i + 1}` });
  });
}

// ================= 3. environments
{
  const s = contentSlide("Intro", "Two environments", "E1: every job uses the whole machine. E4: each job uses 63, 128 or 255 of the 255 capacity, so several small jobs can share it.");
  const scale = 5.2 / 255, x0 = 0.5;
  const colors = { 63: "8FB3E8", 128: "F2C14E", 255: "F28B82" };
  const rows = [
    ["E1: every job uses 255 (one job at a time)", [255], true, 1.25],
    ["E4: 63 + 63 + 128 = 254", [63, 63, 128], true, 2.05],
    ["E4: 63 + 128 = 191", [63, 128], true, 2.75],
    ["E4: 128 + 128 = 256 > 255", [128, 128], false, 3.45],
    ["E4: 255 needs the machine alone (63 + 255 > 255)", [63, 255], false, 4.15],
  ];
  rows.forEach(([label, items, ok, y], r) => {
    s.addText(label, { x: x0, y: y - 0.3, w: 6, h: 0.28, fontSize: 12, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y, w: 255 * scale, h: 0.4, fill: { color: "F1F5F9" }, line: { color: C.accent6, width: 1, dashType: "dash" }, objectName: `env-capacity-${r + 1}` });
    let x = x0;
    items.forEach((v, k) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: v * scale, h: 0.4, fill: { color: colors[v] }, line: { color: C.accent6, width: 1 }, objectName: `env-job-${r + 1}-${k + 1}` });
      s.addText(String(v), { x, y, w: v * scale, h: 0.4, fontSize: 12, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      x += v * scale;
    });
    const end = x0 + Math.max(items.reduce((a, b) => a + b, 0), 255) * scale + 0.15;
    s.addText(ok ? "fits" : "does not fit", { x: end, y, w: 1.6, h: 0.4, fontSize: 12, bold: true, color: ok ? C.accent2 : C.accent4, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Jobs on E4 are 63, 128 or 255 with equal chance. A good policy has to check what fits, not only which job is shortest.", { x: 0.5, y: 4.75, w: 8.4, h: 0.5, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
}

// ================= E1 results (best original model)
pres.addSection({ title: "E1 reference" });
{
  const s = contentSlide("E1 reference", "E1 results: the original best model", "g231w4jg (stellar-dawn-304): masked job self-attention + queue_pressure. 5000 deterministic episodes against SJF and the exact DP optimal.");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "14213D" } } });
  s.addText("Against SJF", { x: 0.5, y: 1.15, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addTable(
    [
      [H("Measure"), H("Value")],
      ["Ties", "82.00% (4100)"],
      ["PPO wins", "14.14% (707)"],
      ["SJF wins", "3.86% (193)"],
      ["Mean reward margin", "+1.14%"],
      ["Wait-time margin", "−0.83%"],
      ["Completion-time margin", "−0.62%"],
    ],
    { x: 0.5, y: 1.5, w: 4.3, colW: [2.5, 1.8], fontSize: 12, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, rowH: 0.36, valign: "middle", objectName: "e1-vs-sjf-table" }
  );
  s.addText("Against the exact optimal", { x: 5.2, y: 1.15, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addTable(
    [
      [H("Measure"), H("Value")],
      ["Optimal mean reward", "−131.0788"],
      ["PPO mean reward", "−131.6342"],
      ["SJF mean reward", "−133.1472"],
      ["Exact optimal schedule, PPO", "90.68% (4534)"],
      ["Exact optimal schedule, SJF", "80.96% (4048)"],
      ["Gap to optimal closed by PPO", "73.1% reward, 69.7% wait"],
    ],
    { x: 5.2, y: 1.5, w: 4.3, colW: [2.3, 2.0], fontSize: 12, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, rowH: 0.36, valign: "middle", objectName: "e1-vs-optimal-table" }
  );
  s.addText("PPO beats SJF by a small margin and sits much closer to the true optimum: it finds the optimal schedule on 90.68% of episodes against SJF's 80.96%.", { x: 0.5, y: 4.3, w: 9, h: 0.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "e1-takeaway" });
}

// ================= 4. E1 originals
{
  const s = contentSlide("E1 reference", "E1: rebuilding the original models", "On E1 the original PPO matched SJF. We rebuilt M0 (np7aenua) and M1 (g231w4jg) to confirm that recent changes were what broke it.");
  stack(s, [R("Per-job input (20)", "status, size, wait, ttl, arrival"), ENC, SELF(false), POOL, SCORE, LOGITS], 0.5, 1.2, 4.5);
  s.addText("M0 (baseline). M1 adds queue_pressure and job self-attention.", { x: 0.5, y: 4.3, w: 4.5, h: 0.5, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true });
  s.addText("Results on E1", { x: 5.3, y: 1.2, w: 4.2, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "14213D" } } });
  s.addTable(
    [
      [hdr("Run"), hdr("Model"), hdr("Reward"), hdr("vs SJF")],
      ["mkl0zwp7", "SJF", "−133.15", "0%"],
      ["6b195fqa", "M0", "−131.84", "+0.98%"],
      ["khz1l51q", "M1", "n/a", "−0.15%"],
      ["script", "Optimal", "−131.08", "+1.56%"],
    ],
    { x: 5.3, y: 1.55, w: 4.2, colW: [1.2, 0.9, 1.1, 1.0], fontSize: 12, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, rowH: 0.34, valign: "middle", objectName: "e1-table" }
  );
  s.addText("Both rebuilt models land within about 1% of SJF, so the E1 task itself is solved. The later regression came from the dual-attention and packing changes.", { x: 5.3, y: 3.5, w: 4.2, h: 1.6, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}

// ================= E1 experiment log
pres.addSection({ title: "E1 experiment log" });
logSlide("E1 experiment log", "E1 log: methodology fixes", [
  ["F1", "Decouple arrival time and size in the job generator", "Both came from one jittered value, so \"arrived first\" and \"smallest\" were almost the same job and SJF won for free.", "Kept"],
  ["F2", "Deterministic evaluation (was sampling actions)", "The same trained model scored from +1.03% to −6.44% depending on the sampling. Several earlier \"results\" were noise.", "Kept"],
  ["F3", "5000 evaluation episodes (was 100) and W&B history samples = 10 000", "100 episodes cannot separate real effects from noise at ~1% margins; W&B silently capped at ~500 points.", "Kept"],
], "These fixes came first because every later number depends on them.", "Everything after this point is trustworthy in a way earlier results were not.");

logSlide("E1 experiment log", "E1 log: task shape and features (#1 to #5)", [
  ["1", "Widen and skew arrivals (0 to 20 ticks)", "Ties fell 73% to 59%: spreading jobs out reduced contention and lengthened episodes.", "Reverted"],
  ["2", "3 machines and 15 jobs", "Partial run: PPO lost badly (92% SJF wins) before it was stopped.", "Reverted"],
  ["3", "Add arrival_time (time until arrival) as a feature", "First genuine win: 73% ties, +1.03% (+0.94% after the eval fix).", "Kept"],
  ["4", "Drop ttl (thought redundant with size)", "Regressed: 52% ties, −14.67%. ttl is the only hint of when the running job frees the machine.", "Reverted"],
  ["5", "queue_pressure, global only (critic input)", "Ties 82.66% to 77.18%, margin +0.74%: more reactive, not more accurate.", "Superseded"],
], null, "#3 and #4 show that a feature that looks redundant can carry the only signal for a decision.");

logSlide("E1 experiment log", "E1 log: attention and reward (#6 to #10)", [
  ["6", "queue_pressure, global and per job", "New best at the time: 82.74% ties, +1.08%, fewer SJF wins (185).", "Kept"],
  ["7", "Job self-attention, unmasked", "Fewest SJF wins (159) but PPO wins also fell (661): more SJF-like, +0.75%.", "Superseded"],
  ["8", "Job self-attention with finished jobs masked out as keys", "Best on E1: 707 PPO wins, 193 SJF wins, 82.00% ties, +1.14%. Improved wins and losses together.", "Kept"],
  ["9", "Deeper job-context encoder and deeper scorer", "Large regression: −17.51%, SJF won 41.34%. More depth hurt at this scale.", "Reverted"],
  ["10", "End the episode on a failed allocation", "Roughly neutral: 75% ties, +0.67%.", "Neutral"],
], "#8 is the model rebuilt as g231w4jg and the best result on E1.");

logSlide("E1 experiment log", "E1 log: training and optimal (#11 to #14)", [
  ["11", "Higher entropy (0.05 to 0.01)", "+0.80% against +0.74%: within noise. Exploration was not the bottleneck.", "Reverted"],
  ["12", "More training steps (2M to 4M)", "No effect: the curve was flat by ~10k updates. Convergence, not undertraining.", "Reverted"],
  ["13", "Higher gamma and lambda (0.999 / 0.99)", "Mild regression: PPO wins 643 (from 707), margin +0.61%.", "Reverted"],
  ["14", "Exact DP optimal solver (Pareto frontier over subsets)", "PPO closes ~73% of the SJF-to-optimal gap and finds the optimal schedule on 90.68% of episodes against SJF's 80.96%.", "Ceiling"],
], "The DP solver is exact because the simulator exposes every job's arrival and size from step 0.");

logSlide("E1 experiment log", "E1 log: dual attention (#15)", [
  ["15a", "Dual attention network (machine + job-machine attention) on 2 machines", "Stopped before completion to test the architecture on one machine first.", "Stopped"],
  ["15b", "Same network on 1 machine (machine attention is a no-op there)", "Ties 82.00% to 75.74%; PPO wins 707 to 844 and SJF wins 193 to 369. Margin +0.97%.", "Not adopted"],
  ["15c", "Gated cross-attention (LayerScale), 1M steps", "Ties 84.36%, SJF wins only 158, margin +0.79%. Gate worked; run got half the steps, so unproven.", "Not adopted"],
], "More decisive, not more accurate: both wins and losses rose together in 15b.");

logSlide("E1 experiment log", "Earlier packing tasks (#16 to #21)", [
  ["16", "Random per-job usage 10 to 50% (E2); compare against the old SJF baseline", "+1.01% only against the old exclusive-occupation baseline; gamma = 1.0 gave −18.07%.", "Neutral"],
  ["17", "Fresh SJF baseline, no usage feature", "SJF won 97.42%, margin −154.14%: the policy could not see what fits.", "Neutral"],
  ["18", "Add a usage feature", "−54.43% recovered to −52.43%; SJF wins 97.42% to 85.92%.", "Kept"],
  ["19", "Longer training, fit_slack feature, narrower usage 40 to 60% (E3)", "Margin −25.33%.", "Kept"],
  ["20", "Explicit per-pair fit term in the scorer", "Margin −15.77%: still behind SJF.", "Kept"],
  ["21", "Learned fit gate", "Regression to −89.98%.", "Reverted"],
], "This is the path that led to the E4 experiments.");

// ================= 5. baselines on E4
pres.addSection({ title: "E4 experiments" });
{
  const s = contentSlide("E4 experiments", "E4 baselines: SJF and the optimal", "SJF run rjzcixv2. The optimal solver searches subsets at every arrival and completion, validated against brute force on 150 small instances; replaying it in the real environment gave 0 mismatches over 5000 episodes.");
  const cardsData = [
    ["SJF", "rjzcixv2", "−50.43", "wait 10.09 · completion 16.33", "Always starts the shortest job that fits. The reference for every margin.", C.accent6],
    ["Exact optimal", "search script", "−44.41", "wait 9.49 · completion 15.73", "Searches every capacity-feasible subset at each arrival or completion. +11.9% over SJF: the ceiling.", C.accent1],
  ];
  cardsData.forEach((c, i) => {
    const x = 0.5 + i * 4.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 4.4, h: 3.4, rectRadius: 0.1, fill: { color: "F1F5F9" }, line: { color: c[5], width: 1.5 }, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: `baseline-card-${i + 1}` });
    s.addText(c[0], { x: x + 0.25, y: 1.45, w: 3.9, h: 0.4, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(c[1], { x: x + 0.25, y: 1.85, w: 3.9, h: 0.3, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true });
    s.addText(c[2], { x: x + 0.25, y: 2.2, w: 3.9, h: 0.8, fontFace: "Cambria", fontSize: 44, bold: true, color: c[5], margin: 0, valign: "middle", isTextBox: true });
    s.addText("mean reward", { x: x + 0.25, y: 3.0, w: 3.9, h: 0.25, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
    s.addText(c[3], { x: x + 0.25, y: 3.3, w: 3.9, h: 0.3, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(c[4], { x: x + 0.25, y: 3.7, w: 3.9, h: 0.9, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Every PPO result that follows is measured against SJF on the same 5000 job sets.", { x: 0.5, y: 4.85, w: 9, h: 0.35, fontSize: 14, color: C.accent6, margin: 0, isTextBox: true });
}

// ================= model slides
function modelSlide(title, rows, o, notes) {
  const s = contentSlide("E4 experiments", title, notes);
  stack(s, rows, 0.5, 1.2, 4.5);
  rightPanel(s, o);
  legend(s);
  return s;
}

modelSlide("M0: cannot see usage (control)",
  [R("Per-job input (20)", "status, size, wait, ttl, arrival"), ENC, POOL, SCORE, LOGITS],
  {
    change: "Nothing new. The E1 model run on E4, so it has no way to see job usage.",
    finding: "It cannot tell what fits, so it behaves like a one-job-at-a-time policy.",
    result: { margin: -164.6, reward: "−133.44", wait: "18.39", comp: "24.63", wins: "wins 56 / SJF 4837 / ties 107", note: "Run cf2zhwti" },
  }, "Control run cf2zhwti. Almost identical to M0's numbers on E1 because it never sees usage.");

modelSlide("M5: usage as raw inputs",
  [R("Per-job input (21)", "M0 features + job usage / 255", true), R("Job encoder", "21 → 256 → 256, now sees usage", true), POOL,
   R("Scorer per job", "+ job usage + machine room: 770 → 128 → 1", true), LOGITS],
  {
    changeLabel: "Change vs M0",
    change: "Job usage and machine room are given as plain numbers, to the encoder and the scorer.",
    finding: "Large recovery, from −164.6% to −45.8%, but still far behind SJF.",
    result: { margin: -45.81, reward: "−73.53", wait: "12.38", comp: "18.62", wins: "wins 266 / SJF 4528 / ties 206", note: "Run z4sayojp · failed starts 0.02 per episode" },
  }, "M5 gives usage as raw numbers. Machine room = min(capacity - usage)/255, job usage = max(usage)/255.");

modelSlide("M6: fit attention",
  [IN20, ENC, R("Fit attention", "Q: job usage, K/V: machine room, sigmoid", true), R("Concat + project", "[summary 256 | fit 64] → 256", true), POOL, SCORE, LOGITS],
  {
    changeLabel: "Change vs M5",
    change: "Usage leaves the encoder. A fit attention pairs each job's usage with the machine's room.",
    finding: "Pairing the two numbers beats feeding them separately: −45.8% to −37.0%.",
    result: { margin: -36.98, reward: "−69.07", wait: "11.60", comp: "17.83", wins: "wins 497 / SJF 4125 / ties 378", note: "Run iusicvre · failed starts 0.34 per episode" },
  }, "A sigmoid is used instead of a softmax because with one machine a softmax is always 1.");

modelSlide("M7: job self-attention",
  [IN20, ENC, R("Fit attention", "Q: job usage, K/V: machine room, sigmoid"), CONCAT, SELF(true), POOL, SCORE, LOGITS],
  {
    changeLabel: "Change vs M6",
    change: "Every job attends to every other job (4 heads, finished jobs masked) after the fit step.",
    finding: "SJF's rule is relative to other jobs. This was the biggest jump before the mask: −37.0% to −16.9%.",
    result: { margin: -16.87, reward: "−58.94", wait: "10.83", comp: "17.07", wins: "wins 855 / SJF 3105 / ties 1040", note: "Run emt46quc · failed starts 0.06 per episode" },
  }, "Self-attention lets a job compare itself with the other jobs, which is the comparison SJF makes.");

modelSlide("M8: multiply instead of concat",
  [IN20, ENC, R("Fit attention", "Q: job usage, K/V: machine room, sigmoid"), R("Multiply per channel", "summary × (1 + tanh(Linear(fit)))", true), SELF(false), POOL, SCORE, LOGITS],
  {
    changeLabel: "Change vs M7",
    change: "The fit output scales each channel of the job summary instead of being concatenated.",
    finding: "Worse than M7 at −25.3%, within the ~10-point seed noise. No benefit shown; rolled back.",
    result: { margin: -25.3, reward: "−63.20", wait: "11.36", comp: "17.60", wins: "wins 483 / SJF 3841 / ties 676", note: "Run bv1hk6lb · failed starts 0.003 per episode" },
  }, "Multiplicative (FiLM-style) fusion. Not adopted.");

modelSlide("M9: feasibility mask",
  [IN20, ENC, R("Fit attention + mask", "weight = 0 where job usage > machine room", true), CONCAT, SELF(false), POOL, SCORE, LOGITS],
  {
    changeLabel: "Change vs M7",
    change: "A job and machine pair gets zero weight when the job needs more than the machine has free.",
    finding: "Jobs only draw on machines they can run on. First model to beat SJF: 800 wins, 662 losses.",
    result: { margin: 2.28, reward: "−49.28", wait: "9.97", comp: "16.20", wins: "wins 800 / SJF 662 / ties 3538", note: "Run etudt4qo · failed starts 0.012 per episode · current best" },
  }, "M9 is the current best. A job that fits nowhere gets an all-zero fit vector.");

modelSlide("M10: more entropy",
  [IN20, ENC, R("Fit attention + mask", "weight = 0 where job usage > machine room"), CONCAT, SELF(false), POOL, SCORE, R("Training change", "entropy coefficient 0.05 → 0.01 (was 0.02 → 0.002)", true)],
  {
    changeLabel: "Change vs M9",
    change: "Same network. The entropy coefficient is raised to keep the policy exploring longer.",
    finding: "No effect: +2.31% against +2.28%. Entropy still ends near 0.05. Rolled back to M9.",
    result: { margin: 2.31, reward: "−49.27", wait: "9.96", comp: "16.20", wins: "wins 788 / SJF 730 / ties 3482", note: "Run ne5qv1d2 · failed starts 0.016 per episode" },
  }, "Entropy was not the limiting factor.");

// ================= position augmentation
{
  const s = contentSlide("E4 experiments", "Check: job position", "Tested but not run. The model is already permutation-equivariant, so the augmentation would only cost compute.");
  s.addText("Tried: train on every shuffled job order", { x: 0.5, y: 1.2, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText("Each sample is copied once per cyclic shift of the 10 job slots, with the chosen action moved to the job's new slot.", { x: 0.5, y: 1.5, w: 4.6, h: 1.1, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Result", { x: 0.5, y: 2.8, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText("Shuffling the jobs changes the logits only by moving them with the jobs, within 5e-7 (rounding). The model has no positional encoding and shares weights across jobs, so shuffled copies add nothing.", { x: 0.5, y: 3.1, w: 4.6, h: 1.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.6, y: 1.4, w: 3.9, h: 3.2, rectRadius: 0.1, fill: { color: "F1F5F9" }, line: { color: C.accent1, width: 1.5 }, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: "equivariance-card" });
  s.addText("5e-7", { x: 5.8, y: 1.6, w: 3.5, h: 1.1, fontFace: "Cambria", fontSize: 54, bold: true, color: C.accent1, margin: 0, valign: "middle", isTextBox: true });
  s.addText("largest logit difference between a shuffled input and the original, moved the same way", { x: 5.8, y: 2.75, w: 3.5, h: 0.9, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Not run: 10 times the compute, no new information", { x: 5.8, y: 3.8, w: 3.5, h: 0.6, fontSize: 12, bold: true, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
}

// ================= progress chart
pres.addSection({ title: "Summary" });
{
  const s = contentSlide("Summary", "Progress step by step", "Margin vs SJF in the order the experiments were run. Only the last two beat SJF.");
  const labels = ["M0", "M5", "M6", "M7", "M8", "M9", "M10"];
  const vals = [-164.6, -45.81, -36.98, -16.87, -25.3, 2.28, 2.31];
  s.addChart(pres.charts.BAR, [{ name: "Margin vs SJF (%)", labels, values: vals }], {
    x: 0.5, y: 1.15, w: 5.6, h: 4.0, barDir: "col",
    chartColors: ["C0392B", "C0392B", "C0392B", "C0392B", "C0392B", "2E9E5B", "2E9E5B"],
    showTitle: true, title: "Mean reward margin vs SJF (%)", titleFontSize: 12, titleColor: "14213D", titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: "1B2330", dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.0",
    catAxisLabelColor: "1B2330", catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: "5C6B7A", valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt", valAxisMinVal: -180, valAxisMaxVal: 20,
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisLabelPos: "low",
  });
  s.addText("Change between experiments", { x: 6.4, y: 1.2, w: 3.1, h: 0.3, fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  const steps = [
    ["M0 → M5", "+118.8 pts", "add usage inputs", true],
    ["M5 → M6", "+8.8 pts", "fit attention", true],
    ["M6 → M7", "+20.1 pts", "job self-attention", true],
    ["M7 → M8", "−8.4 pts", "multiply (noise)", false],
    ["M7 → M9", "+19.2 pts", "feasibility mask", true],
    ["M9 → M10", "+0.03 pts", "more entropy (none)", false],
  ];
  steps.forEach((r, i) => {
    const y = 1.55 + i * 0.58;
    s.addText(r[0], { x: 6.4, y, w: 1.1, h: 0.5, fontSize: 12, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(r[1], { x: 7.5, y, w: 0.95, h: 0.5, fontSize: 12, bold: true, color: r[3] ? C.accent2 : C.accent4, margin: 0, valign: "middle", isTextBox: true });
    s.addText(r[2], { x: 8.45, y, w: 1.1, h: 0.5, fontSize: 10, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText("Points are margin points vs SJF; M8 and M10 are inside the ~10-point seed noise.", { x: 6.4, y: 5.05, w: 3.1, h: 0.4, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
}

// ================= summary table
{
  const s = contentSlide("Summary", "Summary table (E4)", "All runs: 5000 deterministic episodes, same job sets as SJF rjzcixv2.");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "14213D" }, align: "center" } });
  const G = (t) => ({ text: t, options: { color: "2E9E5B", bold: true, align: "center" } });
  const B = (t) => ({ text: t, options: { color: "C0392B", align: "center" } });
  const c = (t) => ({ text: t, options: { align: "center" } });
  const l = (t) => ({ text: t, options: { align: "left" } });
  const rows = [
    [H("Run"), H("Model"), H("Change vs previous"), H("Reward"), H("vs SJF"), H("Wait"), H("Compl."), H("W / L / T")],
    [c("rjzcixv2"), c("SJF"), l("baseline"), c("−50.43"), c("0%"), c("10.09"), c("16.33"), c("n/a")],
    [c("script"), c("Optimal"), l("exact search"), c("−44.41"), G("+11.9%"), c("9.49"), c("15.73"), c("n/a")],
    [c("cf2zhwti"), c("M0"), l("cannot see usage"), c("−133.44"), B("−164.6%"), c("18.39"), c("24.63"), c("56/4837/107")],
    [c("z4sayojp"), c("M5"), l("usage as raw inputs"), c("−73.53"), B("−45.81%"), c("12.38"), c("18.62"), c("266/4528/206")],
    [c("iusicvre"), c("M6"), l("fit attention"), c("−69.07"), B("−36.98%"), c("11.60"), c("17.83"), c("497/4125/378")],
    [c("emt46quc"), c("M7"), l("+ job self-attention"), c("−58.94"), B("−16.87%"), c("10.83"), c("17.07"), c("855/3105/1040")],
    [c("bv1hk6lb"), c("M8"), l("multiply instead of concat"), c("−63.20"), B("−25.3%"), c("11.36"), c("17.60"), c("483/3841/676")],
    [c("etudt4qo"), c("M9"), l("+ feasibility mask"), c("−49.28"), G("+2.28%"), c("9.97"), c("16.20"), c("800/662/3538")],
    [c("ne5qv1d2"), c("M10"), l("more entropy"), c("−49.27"), G("+2.31%"), c("9.96"), c("16.20"), c("788/730/3482")],
  ];
  s.addTable(rows, { x: 0.5, y: 1.2, w: 9, colW: [0.95, 0.7, 2.25, 0.8, 0.85, 0.6, 0.65, 1.2], fontSize: 11, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, rowH: 0.34, valign: "middle", objectName: "summary-table" });
  s.addText("W / L / T = PPO wins, SJF wins, ties. M8 and M10 differ from their parents by less than the ~10-point seed noise.", { x: 0.5, y: 4.75, w: 9, h: 0.4, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true });
}

// ================= E1 summary table
{
  const s = contentSlide("Summary", "Summary table (E1)", "Margins are against the SJF baseline that was run alongside each model (nb11ox0z for the original runs, mkl0zwp7 for the rebuilt ones).");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "14213D" }, align: "center" } });
  const G = (t) => ({ text: t, options: { color: "2E9E5B", bold: true, align: "center" } });
  const B = (t) => ({ text: t, options: { color: "C0392B", align: "center" } });
  const c = (t) => ({ text: t, options: { align: "center" } });
  const l = (t) => ({ text: t, options: { align: "left" } });
  const rows = [
    [H("Run"), H("Model"), H("What it is"), H("vs SJF"), H("Ties"), H("W / L")],
    [c("g231w4jg"), c("M1"), l("masked job self-attention + queue_pressure (original best)"), G("+1.14%"), c("82.00%"), c("707 / 193")],
    [c("ddsv1ozh"), c("15b"), l("dual attention, 1 machine"), G("+0.97%"), c("75.74%"), c("844 / 369")],
    [c("cxtxjusj"), c("15c"), l("gated cross-attention, 1M steps"), G("+0.79%"), c("84.36%"), c("624 / 158")],
    [c("6b195fqa"), c("M0"), l("rebuilt np7aenua (rerun on E1)"), G("+0.98%"), c("n/a"), c("n/a")],
    [c("khz1l51q"), c("M1"), l("rebuilt g231w4jg (rerun on E1)"), B("−0.15%"), c("n/a"), c("n/a")],
    [c("script"), c("Optimal"), l("exact DP (rerun SJF mkl0zwp7, −133.15)"), G("+1.56%"), c("n/a"), c("n/a")],
  ];
  s.addTable(rows, { x: 0.5, y: 1.2, w: 9, colW: [1.05, 0.8, 4.05, 0.9, 1.0, 1.2], fontSize: 12, color: "1B2330", border: { type: "solid", color: "C9D3DD", pt: 0.75 }, rowH: 0.4, valign: "middle", objectName: "e1-summary-table" });
  s.addText("W / L = PPO wins and SJF wins over 5000 episodes. Every E1 result is within about 1.5% of SJF: E1 has little room above the heuristic.", { x: 0.5, y: 4.2, w: 9, h: 0.7, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}

// ================= takeaways
{
  const s = contentSlide("Summary", "What we learned and what is next", "The biggest gains came from telling the model what fits. Mixing changes made no difference.");
  const cols = [
    ["What worked", C.accent2, ["Seeing what fits: −164.6% to −37.0%", "Comparing jobs: −16.9%", "Masking jobs that cannot run: +2.3%, the first win over SJF"]],
    ["What did not", C.accent4, ["Multiplying instead of concatenating", "A higher entropy coefficient", "Job-position augmentation (no gain possible)"]],
    ["Next", C.accent1, ["A second seed of M9 to confirm +2.3%", "Action masking on the logits", "Behaviour-cloning from SJF", "Ablate each attention step"]],
  ];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.9, h: 3.7, rectRadius: 0.1, fill: { color: "F1F5F9" }, line: { color: c[1], width: 1.5 }, objectName: `takeaway-card-${i + 1}` });
    s.addText(c[0], { x: x + 0.2, y: 1.45, w: 2.5, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: c[1], margin: 0, isTextBox: true });
    s.addText(c[2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[2].length - 1 } })), { x: x + 0.2, y: 1.95, w: 2.55, h: 2.95, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: `takeaway-bullets-${i + 1}` });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
