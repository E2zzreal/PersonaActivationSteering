const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "PersonaSteer Team";
pres.title = "PersonaSteer Experiment Report";

// Color palette: Midnight Executive
const C = {
  navy: "1E2761",
  ice: "CADCFC",
  white: "FFFFFF",
  dark: "0F1635",
  accent: "4A90D9",
  red: "E74C3C",
  green: "27AE60",
  gray: "8B95A5",
  lightgray: "F0F2F5",
  gold: "F5B041",
};

const mkShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

// ═══════════════════════════════════════════════════════
// SLIDE 1: Title
// ═══════════════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.dark };
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.2, w: 10, h: 1.425, fill: { color: C.navy } });
s1.addText("PersonaSteer", {
  x: 0.8, y: 1.0, w: 8.4, h: 1.2,
  fontSize: 44, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});
s1.addText("Dynamic Activation Steering for Multi-Persona Dialogue", {
  x: 0.8, y: 2.2, w: 8.4, h: 0.6,
  fontSize: 18, fontFace: "Calibri", color: C.ice, margin: 0,
});
s1.addText("Experiment Report  |  2026-04-21", {
  x: 0.8, y: 4.55, w: 4, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.ice, margin: 0,
});
s1.addText("Qwen3-4B  |  Big Five Personality  |  LLM Judge Evaluation", {
  x: 0.8, y: 4.95, w: 6, h: 0.4,
  fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
});

// ═══════════════════════════════════════════════════════
// SLIDE 2: Architecture
// ═══════════════════════════════════════════════════════
let s2 = pres.addSlide();
s2.background = { color: C.white };
s2.addText("Architecture Overview", {
  x: 0.8, y: 0.3, w: 8, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Architecture flow boxes
const boxes = [
  { label: "Personality Text", sub: "Big Five Description", y: 0.2, color: C.accent },
  { label: "Big Five [O,C,E,A,N]", sub: "5D Vector Input", y: 0.2, color: C.gold },
  { label: "HyperNetwork", sub: "Encoder + 5D Branch + MLP", y: 1.6, color: C.navy },
  { label: "DynamicGate", sub: "Per-layer gate coefficients", y: 2.8, color: C.navy },
  { label: "Qwen3-4B Backbone", sub: "Frozen, 36 layers, hook injection", y: 4.0, color: C.dark },
];

// Left column: text path
s2.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: 1.2, w: 3.2, h: 0.7, fill: { color: C.accent }, shadow: mkShadow() });
s2.addText("Personality Text", { x: 1.0, y: 1.2, w: 3.2, h: 0.45, fontSize: 13, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
s2.addText("Big Five Description", { x: 1.0, y: 1.55, w: 3.2, h: 0.3, fontSize: 10, color: C.ice, align: "center", margin: 0 });

// Right column: 5D path
s2.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: 1.2, w: 3.2, h: 0.7, fill: { color: C.gold }, shadow: mkShadow() });
s2.addText("Big Five [O,C,E,A,N]", { x: 5.8, y: 1.2, w: 3.2, h: 0.45, fontSize: 13, bold: true, color: C.dark, align: "center", valign: "middle", margin: 0 });
s2.addText("5D Structured Vector", { x: 5.8, y: 1.55, w: 3.2, h: 0.3, fontSize: 10, color: "7D6608", align: "center", margin: 0 });

// Arrows down
s2.addShape(pres.shapes.LINE, { x: 2.6, y: 1.9, w: 0, h: 0.4, line: { color: C.gray, width: 2 } });
s2.addShape(pres.shapes.LINE, { x: 7.4, y: 1.9, w: 0, h: 0.4, line: { color: C.gray, width: 2 } });

// HyperNetwork
s2.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: 2.3, w: 8.0, h: 0.8, fill: { color: C.navy }, shadow: mkShadow() });
s2.addText([
  { text: "HyperNetwork", options: { bold: true, fontSize: 14, color: C.white, breakLine: true } },
  { text: "Frozen Encoder + Big Five 5D Branch + Projector MLP", options: { fontSize: 10, color: C.ice } },
], { x: 1.0, y: 2.3, w: 8.0, h: 0.8, align: "center", valign: "middle", margin: 0 });

// Arrow
s2.addShape(pres.shapes.LINE, { x: 5.0, y: 3.1, w: 0, h: 0.3, line: { color: C.gray, width: 2 } });

// DynamicGate
s2.addShape(pres.shapes.RECTANGLE, { x: 2.5, y: 3.4, w: 5.0, h: 0.6, fill: { color: "2C3E6B" }, shadow: mkShadow() });
s2.addText("DynamicGate  |  g_i = sigmoid(MLP(v_t))  |  Layers [16..23]", {
  x: 2.5, y: 3.4, w: 5.0, h: 0.6, fontSize: 11, color: C.white, align: "center", valign: "middle", margin: 0,
});

// Arrow
s2.addShape(pres.shapes.LINE, { x: 5.0, y: 4.0, w: 0, h: 0.3, line: { color: C.gray, width: 2 } });

// Backbone
s2.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: 4.3, w: 8.0, h: 0.8, fill: { color: C.dark }, shadow: mkShadow() });
s2.addText([
  { text: "Qwen3-4B Backbone (Frozen)", options: { bold: true, fontSize: 14, color: C.white, breakLine: true } },
  { text: "h'_i = h_i + g_i * proj_i(v_t)  via forward hooks", options: { fontSize: 10, color: C.ice } },
], { x: 1.0, y: 4.3, w: 8.0, h: 0.8, align: "center", valign: "middle", margin: 0 });

// ═══════════════════════════════════════════════════════
// SLIDE 3: Experiment Journey (score progression)
// ═══════════════════════════════════════════════════════
let s3 = pres.addSlide();
s3.background = { color: C.white };
s3.addText("Experiment Journey", {
  x: 0.8, y: 0.3, w: 8, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

s3.addChart(pres.charts.BAR, [{
  name: "Score",
  labels: ["ALOE Gold", "Baseline", "Claude SFT\nStage2", "Big Five\nPersonaSteer", "Big Five\nBaseline"],
  values: [2.600, 2.833, 3.033, 3.503, 3.500],
}], {
  x: 0.5, y: 1.2, w: 9, h: 3.8, barDir: "col",
  chartColors: [C.red, C.gray, C.accent, C.green, C.gold],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.navy,
  catAxisLabelColor: "4A5568", valAxisLabelColor: "4A5568",
  valGridLine: { color: "E2E8F0", size: 0.5 },
  catGridLine: { style: "none" },
  showLegend: false,
  valAxisMinVal: 2.0, valAxisMaxVal: 4.0,
  chartArea: { fill: { color: C.white } },
});

s3.addText("Strict v3 Rubric  |  GPT-5.4 Judge  |  'Good but generic = 3'", {
  x: 0.5, y: 5.1, w: 9, h: 0.3, fontSize: 10, color: C.gray, align: "center", margin: 0,
});

// ═══════════════════════════════════════════════════════
// SLIDE 4: Critical Finding
// ═══════════════════════════════════════════════════════
let s4 = pres.addSlide();
s4.background = { color: C.dark };
s4.addText("Critical Finding", {
  x: 0.8, y: 0.3, w: 8, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

// Big numbers
s4.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.3, w: 4, h: 2.2, fill: { color: C.navy }, shadow: mkShadow() });
s4.addText("3.503", { x: 0.8, y: 1.3, w: 4, h: 1.4, fontSize: 60, bold: true, color: C.green, align: "center", valign: "middle", margin: 0 });
s4.addText("PersonaSteer + Injection", { x: 0.8, y: 2.5, w: 4, h: 0.5, fontSize: 14, color: C.ice, align: "center", margin: 0 });
s4.addText("197 samples, strict v3 rubric", { x: 0.8, y: 2.9, w: 4, h: 0.4, fontSize: 10, color: C.gray, align: "center", margin: 0 });

s4.addText("vs", { x: 4.5, y: 2.0, w: 1, h: 0.6, fontSize: 20, color: C.gray, align: "center", valign: "middle", margin: 0 });

s4.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.3, w: 4, h: 2.2, fill: { color: C.navy }, shadow: mkShadow() });
s4.addText("3.500", { x: 5.2, y: 1.3, w: 4, h: 1.4, fontSize: 60, bold: true, color: C.gold, align: "center", valign: "middle", margin: 0 });
s4.addText("Baseline (Prompt Only)", { x: 5.2, y: 2.5, w: 4, h: 0.5, fontSize: 14, color: C.ice, align: "center", margin: 0 });
s4.addText("50 samples, same personalities", { x: 5.2, y: 2.9, w: 4, h: 0.4, fontSize: 10, color: C.gray, align: "center", margin: 0 });

// Conclusion box
s4.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 3.9, w: 8.4, h: 1.2, fill: { color: "1A1A2E" } });
s4.addText([
  { text: "Injection provides +0.003 gain (not significant)", options: { bold: true, fontSize: 16, color: C.red, breakLine: true } },
  { text: "", options: { fontSize: 8, breakLine: true } },
  { text: "The 3.033 → 3.503 improvement comes entirely from better personality descriptions\n(Big Five structured format), not from the injection mechanism.", options: { fontSize: 12, color: C.ice } },
], { x: 1.2, y: 3.9, w: 7.6, h: 1.2, valign: "middle", margin: 0 });

// ═══════════════════════════════════════════════════════
// SLIDE 5: Probing Analysis
// ═══════════════════════════════════════════════════════
let s5 = pres.addSlide();
s5.background = { color: C.white };
s5.addText("Layer Probing: Where Personality Lives", {
  x: 0.8, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Chart: R² per dimension across layers (simplified as bar chart of peak layers)
s5.addChart(pres.charts.BAR, [
  { name: "R² Score", labels: ["C\nConscient.", "A\nAgreeabl.", "E\nExtraver.", "N\nNeurotic.", "O\nOpenness"], values: [0.815, 0.711, 0.685, 0.396, 0.386] },
], {
  x: 0.5, y: 1.2, w: 5, h: 3.5, barDir: "col",
  chartColors: [C.accent],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.navy,
  catAxisLabelColor: "4A5568", valAxisLabelColor: "4A5568",
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false,
  valAxisMinVal: 0, valAxisMaxVal: 1.0,
  chartArea: { fill: { color: C.white } },
});

// Insights on right
const insights = [
  ["C (Conscientiousness)", "R²=0.815, peaks at layer 19\nStrongest signal — structured vs chaotic language"],
  ["E (Extraversion)", "R²=0.685, increases with depth\nSurfaces in deep semantic processing"],
  ["O & N", "R²<0.40 throughout\nModel least sensitive to these"],
  ["Recommended", "Injection window: Layer [16..23]\nOverall R²=0.559"],
];
let iy = 1.3;
for (const [title, body] of insights) {
  s5.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: iy, w: 3.7, h: 0.8, fill: { color: C.lightgray } });
  s5.addText([
    { text: title, options: { bold: true, fontSize: 11, color: C.navy, breakLine: true } },
    { text: body, options: { fontSize: 9, color: "4A5568" } },
  ], { x: 6.0, y: iy + 0.05, w: 3.3, h: 0.7, valign: "top", margin: 0 });
  iy += 0.9;
}

// ═══════════════════════════════════════════════════════
// SLIDE 6: Big Five Personalities
// ═══════════════════════════════════════════════════════
let s6 = pres.addSlide();
s6.background = { color: C.white };
s6.addText("16 Big Five Personalities (cos similarity: 0.037)", {
  x: 0.8, y: 0.3, w: 9, h: 0.7,
  fontSize: 24, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Per-persona scores table
const personaData = [
  [
    { text: "Persona", options: { bold: true, color: C.white, fill: { color: C.navy } } },
    { text: "Score", options: { bold: true, color: C.white, fill: { color: C.navy } } },
    { text: "Key Trait", options: { bold: true, color: C.white, fill: { color: C.navy } } },
  ],
  ["Dreamer", "4.08", "High O, Low C, High N"],
  ["Artist", "4.00", "High O, Low C, High N"],
  ["Stoic", "3.82", "Low O, High C, Low E, Low N"],
  ["Hermit", "3.64", "Low E (-0.9)"],
  ["Entertainer", "3.62", "High E (+0.9)"],
  ["Explorer", "3.58", "High O, High E"],
  ["Caretaker", "3.23", "High A, High N"],
  ["Commander", "2.93", "Low A (-0.6)"],
  ["Perfectionist", "2.55", "High N, High C, Low E"],
];
s6.addTable(personaData, {
  x: 0.8, y: 1.1, w: 8.4,
  colW: [2.5, 1.2, 4.7],
  fontSize: 11, color: "333333",
  border: { pt: 0.5, color: "DDDDDD" },
  rowH: [0.35, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32],
});

s6.addText("Extreme personalities (Dreamer, Artist, Stoic) score highest — distinctive speech patterns are easiest to learn", {
  x: 0.8, y: 4.6, w: 8.4, h: 0.5, fontSize: 11, italic: true, color: C.gray, margin: 0,
});

// Compare with ALOE
s6.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 5.0, w: 4, h: 0.4, fill: { color: C.lightgray } });
s6.addText("ALOE 57 personalities: encoder cos = 0.961", { x: 0.8, y: 5.0, w: 4, h: 0.4, fontSize: 10, color: C.red, align: "center", valign: "middle", margin: 0 });
s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 5.0, w: 4, h: 0.4, fill: { color: C.lightgray } });
s6.addText("Big Five 16 personalities: 5D cos = 0.037", { x: 5.2, y: 5.0, w: 4, h: 0.4, fontSize: 10, color: C.green, align: "center", valign: "middle", margin: 0 });

// ═══════════════════════════════════════════════════════
// SLIDE 7: Diagnosis Chain
// ═══════════════════════════════════════════════════════
let s7 = pres.addSlide();
s7.background = { color: C.white };
s7.addText("Diagnosis Chain: Finding the Bottleneck", {
  x: 0.8, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

const steps = [
  { q: "Is injection mechanism the bottleneck?", a: "NO — Embedding Table (A1) and gate=50% (B) both score 3.0", color: C.green },
  { q: "Is Claude generation quality too low?", a: "NO — Claude scores 4.0 with simple prompt", color: C.green },
  { q: "Is personality description the issue?", a: "YES — Encoder cos=0.961, all ALOE personalities identical", color: C.red },
  { q: "Does Big Five fix it?", a: "YES for descriptions (3.5) — but NO for injection (baseline = 3.5 too)", color: C.gold },
];

let sy = 1.2;
for (let i = 0; i < steps.length; i++) {
  const s = steps[i];
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: sy, w: 0.08, h: 0.95, fill: { color: s.color } });
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.88, y: sy, w: 8.32, h: 0.95, fill: { color: C.lightgray } });
  s7.addText([
    { text: `Q${i+1}: ${s.q}`, options: { bold: true, fontSize: 13, color: C.navy, breakLine: true } },
    { text: s.a, options: { fontSize: 11, color: "4A5568" } },
  ], { x: 1.1, y: sy + 0.05, w: 7.9, h: 0.85, valign: "middle", margin: 0 });
  sy += 1.05;
}

// ═══════════════════════════════════════════════════════
// SLIDE 8: Future Directions
// ═══════════════════════════════════════════════════════
let s8 = pres.addSlide();
s8.background = { color: C.dark };
s8.addText("Future Research Directions", {
  x: 0.8, y: 0.3, w: 8, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

const directions = [
  {
    title: "A. No-Prompt Injection",
    desc: "Remove personality from system prompt.\nv_t becomes the ONLY personality signal.\nTests injection's independent value.",
    accent: C.accent,
  },
  {
    title: "B. Multi-Turn Consistency",
    desc: "10-20 turn dialogues.\nDoes prompt-based personality fade?\nv_t injection should maintain consistency.",
    accent: C.green,
  },
  {
    title: "C. Continuous Interpolation",
    desc: "Smoothly transition E=+0.9 → E=-0.9.\nObserve gradual style change.\nPrompts can't do continuous control.",
    accent: C.gold,
  },
];

let dx = 0.6;
for (const d of directions) {
  s8.addShape(pres.shapes.RECTANGLE, { x: dx, y: 1.3, w: 2.8, h: 3.5, fill: { color: C.navy }, shadow: mkShadow() });
  s8.addShape(pres.shapes.RECTANGLE, { x: dx, y: 1.3, w: 2.8, h: 0.06, fill: { color: d.accent } });
  s8.addText(d.title, { x: dx + 0.2, y: 1.5, w: 2.4, h: 0.5, fontSize: 15, bold: true, color: C.white, margin: 0 });
  s8.addText(d.desc, { x: dx + 0.2, y: 2.1, w: 2.4, h: 2.2, fontSize: 12, color: C.ice, margin: 0 });
  dx += 3.1;
}

s8.addText("When does injection outperform prompting?", {
  x: 0.8, y: 5.0, w: 8.4, h: 0.4,
  fontSize: 14, italic: true, color: C.gray, align: "center", margin: 0,
});

// ═══════════════════════════════════════════════════════
// SLIDE 9: Key Takeaways
// ═══════════════════════════════════════════════════════
let s9 = pres.addSlide();
s9.background = { color: C.navy };
s9.addText("Key Takeaways", {
  x: 0.8, y: 0.3, w: 8, h: 0.7,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

const takeaways = [
  "Big Five structured personalities dramatically improve persona alignment\n(cos 0.961 → 0.037, score 3.03 → 3.50)",
  "Probing reveals personality info peaks at layers 16-23\n(C dimension R²=0.815, E increases with depth)",
  "Injection mechanism provides no gain when personality is in the prompt\n(3.503 vs 3.500, not significant)",
  "Future value: no-prompt control, multi-turn consistency, continuous interpolation",
];

let ty = 1.2;
for (let i = 0; i < takeaways.length; i++) {
  s9.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: ty, w: 8.4, h: 0.9, fill: { color: "253270" } });
  s9.addText([
    { text: `${i+1}`, options: { bold: true, fontSize: 24, color: C.accent } },
    { text: `   ${takeaways[i]}`, options: { fontSize: 13, color: C.ice } },
  ], { x: 1.0, y: ty, w: 8.0, h: 0.9, valign: "middle", margin: 0 });
  ty += 1.0;
}

// Write
const outputPath = "/home/kemove/Desktop/PersonaSteer/docs/PersonaSteer_Report_2026-04-21.pptx";
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log("PPT saved to: " + outputPath);
});
