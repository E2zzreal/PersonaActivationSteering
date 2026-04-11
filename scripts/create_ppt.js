const pptxgen = require("pptxgenjs");

// Create presentation
let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'PersonaSteer Team';
pres.title = 'PersonaSteer 训练与分析报告';

// Color palette - Midnight Executive
const colors = {
  primary: "1E2761",    // Navy
  secondary: "CADCFC",  // Ice blue
  accent: "0D9488",     // Teal
  white: "FFFFFF",
  dark: "0F172A",
  text: "334155",
  lightBg: "F8FAFC",
  success: "10B981",
  warning: "F59E0B",
  danger: "EF4444"
};

// ==================== Slide 1: Title ====================
let slide1 = pres.addSlide();
slide1.background = { color: colors.primary };

// Title
slide1.addText("PersonaSteer", {
  x: 0.5, y: 1.8, w: 9, h: 1,
  fontSize: 54, fontFace: "Arial", color: colors.white, bold: true,
  align: "center"
});

// Subtitle
slide1.addText("人格引导模型训练与分析报告", {
  x: 0.5, y: 2.8, w: 9, h: 0.6,
  fontSize: 28, fontFace: "Arial", color: colors.secondary,
  align: "center"
});

// Date
slide1.addText("2026-04-11", {
  x: 0.5, y: 4.5, w: 9, h: 0.4,
  fontSize: 16, fontFace: "Arial", color: colors.secondary,
  align: "center"
});

// Decorative line
slide1.addShape(pres.shapes.LINE, {
  x: 3, y: 3.6, w: 4, h: 0,
  line: { color: colors.accent, width: 3 }
});

// ==================== Slide 2: Overview ====================
let slide2 = pres.addSlide();
slide2.background = { color: colors.lightBg };

// Header bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.primary }
});
slide2.addText("项目概述", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial", color: colors.white, bold: true
});

// Content
slide2.addText("目标", {
  x: 0.5, y: 1.1, w: 4, h: 0.4,
  fontSize: 20, fontFace: "Arial", color: colors.primary, bold: true
});

slide2.addText([
  { text: "基于 HyperNetwork 的人格引导模型", options: { bullet: true, breakLine: true } },
  { text: "通过注入层使LLM生成符合特定人格特质的回复", options: { bullet: true, breakLine: true } },
  { text: "渐进式三阶段训练", options: { bullet: true } }
], {
  x: 0.5, y: 1.5, w: 4.5, h: 1.5,
  fontSize: 14, fontFace: "Arial", color: colors.text
});

// Tech specs box
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.1, w: 4.3, h: 2.2,
  fill: { color: colors.white },
  shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.1 }
});

slide2.addText("技术架构", {
  x: 5.4, y: 1.2, w: 4, h: 0.4,
  fontSize: 18, fontFace: "Arial", color: colors.accent, bold: true
});

slide2.addText([
  { text: "基础模型: Qwen3-4B", options: { breakLine: true } },
  { text: "核心组件: HyperNetwork", options: { breakLine: true } },
  { text: "注入层: Injection Layers", options: { breakLine: true } },
  { text: "训练阶段: Stage 1→2→3", options: { breakLine: true } }
], {
  x: 5.4, y: 1.6, w: 4, h: 1.6,
  fontSize: 13, fontFace: "Arial", color: colors.text
});

// Config comparison
slide2.addText("三种训练配置", {
  x: 0.5, y: 3.5, w: 9, h: 0.4,
  fontSize: 18, fontFace: "Arial", color: colors.primary, bold: true
});

slide2.addTable([
  [
    { text: "配置", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "注入层数", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "特点", options: { fill: { color: colors.primary }, color: colors.white, bold: true } }
  ],
  ["Neuroticism", "3层", "最少注入，专注神经质"],
  ["Minimal", "6层", "中等注入，平衡配置"],
  ["Baseline", "8层", "最多注入，全覆盖"]
], {
  x: 0.5, y: 3.9, w: 9, h: 1.5,
  fontSize: 12, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// ==================== Slide 3: Evaluation Results ====================
let slide3 = pres.addSlide();
slide3.background = { color: colors.lightBg };

// Header
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.primary }
});
slide3.addText("评估结果 - LLM Judge 评分", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial", color: colors.white, bold: true
});

// Score table
slide3.addText("人格一致性评分 (1-5分)", {
  x: 0.5, y: 1.0, w: 9, h: 0.4,
  fontSize: 16, fontFace: "Arial", color: colors.primary, bold: true
});

slide3.addTable([
  [
    { text: "配置", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "Stage 1", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "Stage 2", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "Stage 3", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "平均分", options: { fill: { color: colors.primary }, color: colors.white, bold: true } }
  ],
  [
    { text: "Minimal", options: { bold: true, color: colors.accent } },
    "3.0", "3.1",
    { text: "3.3", options: { bold: true } },
    { text: "3.13", options: { bold: true, fill: { color: "D1FAE5" } } }
  ],
  ["Baseline", "3.2", "2.8", "2.89", "2.96"],
  ["Neuroticism", "2.8", "3.0", "2.7", "2.83"]
], {
  x: 0.5, y: 1.4, w: 5.5, h: 1.6,
  fontSize: 13, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// Loss table
slide3.addText("训练 Loss 对比", {
  x: 6.2, y: 1.0, w: 3.5, h: 0.4,
  fontSize: 16, fontFace: "Arial", color: colors.primary, bold: true
});

slide3.addTable([
  [
    { text: "配置", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "平均Loss", options: { fill: { color: colors.primary }, color: colors.white, bold: true } }
  ],
  [
    { text: "Neuroticism", options: { bold: true } },
    { text: "3.08", options: { fill: { color: "FEF3C7" } } }
  ],
  ["Minimal", "3.14"],
  ["Baseline", "3.22"]
], {
  x: 6.2, y: 1.4, w: 3.3, h: 1.2,
  fontSize: 13, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// Key insight box
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.2, w: 9, h: 1.1,
  fill: { color: colors.white },
  shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 }
});

slide3.addText("💡 关键发现", {
  x: 0.7, y: 3.3, w: 8.6, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.accent, bold: true
});

slide3.addText("Loss-Score 相关系数: 0.345 — 低训练 Loss 不一定带来高人格一致性评分", {
  x: 0.7, y: 3.7, w: 8.6, h: 0.5,
  fontSize: 14, fontFace: "Arial", color: colors.text
});

// Chart area
slide3.addChart(pres.charts.BAR, [
  { name: "Stage 1", labels: ["Minimal", "Baseline", "Neuroticism"], values: [3.0, 3.2, 2.8] },
  { name: "Stage 2", labels: ["Minimal", "Baseline", "Neuroticism"], values: [3.1, 2.8, 3.0] },
  { name: "Stage 3", labels: ["Minimal", "Baseline", "Neuroticism"], values: [3.3, 2.89, 2.7] }
], {
  x: 0.5, y: 4.4, w: 9, h: 1.0,
  barDir: "col",
  chartColors: ["0D9488", "14B8A6", "5EEAD4"],
  showLegend: true,
  legendPos: "r",
  showValue: false,
  catAxisLabelColor: colors.text,
  valAxisLabelColor: colors.text,
  valGridLine: { color: "E2E8F0", size: 0.5 },
  catGridLine: { style: "none" }
});

// ==================== Slide 4: Key Findings ====================
let slide4 = pres.addSlide();
slide4.background = { color: colors.lightBg };

// Header
slide4.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.primary }
});
slide4.addText("关键发现", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial", color: colors.white, bold: true
});

// Finding cards
const findings = [
  { title: "最佳配置", value: "Minimal", desc: "LLM Judge评分最高 (3.13)\nStage 3达到最佳效果", color: colors.success },
  { title: "Loss最低", value: "Neuroticism", desc: "训练Loss最低 (3.08)\n但评分不是最高", color: colors.warning },
  { title: "最稳定", value: "Baseline", desc: "评分标准差最小 (0.80)\n波动较小", color: colors.accent }
];

findings.forEach((f, i) => {
  const x = 0.5 + i * 3.1;

  // Card background
  slide4.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.0, w: 2.9, h: 2.2,
    fill: { color: colors.white },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 }
  });

  // Accent bar
  slide4.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.0, w: 0.08, h: 2.2,
    fill: { color: f.color }
  });

  // Title
  slide4.addText(f.title, {
    x: x + 0.2, y: 1.1, w: 2.6, h: 0.35,
    fontSize: 14, fontFace: "Arial", color: colors.text
  });

  // Value
  slide4.addText(f.value, {
    x: x + 0.2, y: 1.45, w: 2.6, h: 0.5,
    fontSize: 24, fontFace: "Arial", color: f.color, bold: true
  });

  // Description
  slide4.addText(f.desc, {
    x: x + 0.2, y: 2.0, w: 2.6, h: 1.0,
    fontSize: 11, fontFace: "Arial", color: colors.text
  });
});

// Problem section
slide4.addText("⚠️ 核心问题：思考过程泄露", {
  x: 0.5, y: 3.4, w: 9, h: 0.4,
  fontSize: 18, fontFace: "Arial", color: colors.danger, bold: true
});

// Problem example
slide4.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.9, w: 4.3, h: 1.5,
  fill: { color: "FEF2F2" },
  line: { color: colors.danger, width: 1 }
});

slide4.addText("❌ 问题回复:", {
  x: 0.7, y: 4.0, w: 4, h: 0.3,
  fontSize: 12, fontFace: "Arial", color: colors.danger, bold: true
});

slide4.addText('"Okay, the user is asking... I need to respond in a friendly way. First, I should..."', {
  x: 0.7, y: 4.3, w: 4, h: 1.0,
  fontSize: 11, fontFace: "Arial", color: colors.text, italic: true
});

// Expected example
slide4.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 3.9, w: 4.3, h: 1.5,
  fill: { color: "D1FAE5" },
  line: { color: colors.success, width: 1 }
});

slide4.addText("✅ 期望回复:", {
  x: 5.4, y: 4.0, w: 4, h: 0.3,
  fontSize: 12, fontFace: "Arial", color: colors.success, bold: true
});

slide4.addText('"That sounds great! I\'ve been working on some interesting projects. How about you?"', {
  x: 5.4, y: 4.3, w: 4, h: 1.0,
  fontSize: 11, fontFace: "Arial", color: colors.text, italic: true
});

// ==================== Slide 5: Post-processing Results ====================
let slide5 = pres.addSlide();
slide5.background = { color: colors.lightBg };

// Header
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.primary }
});
slide5.addText("后处理优化尝试", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial", color: colors.white, bold: true
});

// Method description
slide5.addText("过滤方法", {
  x: 0.5, y: 1.0, w: 9, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.primary, bold: true
});

slide5.addText([
  { text: "移除 \"Okay, the user...\" 开头", options: { bullet: true, breakLine: true } },
  { text: "移除 \"首先，我需要...\" 中文思考", options: { bullet: true, breakLine: true } },
  { text: "移除 \"用户:\" 后续生成内容", options: { bullet: true } }
], {
  x: 0.5, y: 1.4, w: 4.5, h: 1.0,
  fontSize: 13, fontFace: "Arial", color: colors.text
});

// Results table
slide5.addText("后处理效果对比", {
  x: 0.5, y: 2.5, w: 9, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.primary, bold: true
});

slide5.addTable([
  [
    { text: "配置", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "原评分", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "新评分", options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
    { text: "变化", options: { fill: { color: colors.primary }, color: colors.white, bold: true } }
  ],
  [
    { text: "minimal_stage2", options: { bold: true } },
    "3.10",
    { text: "3.40", options: { bold: true } },
    { text: "+0.30", options: { color: colors.success, bold: true } }
  ],
  ["minimal_stage1", "3.00", "3.10", { text: "+0.10", options: { color: colors.success } }],
  ["baseline_stage2", "2.80", "2.90", { text: "+0.10", options: { color: colors.success } }],
  ["minimal_stage3", "3.30", "2.90", { text: "-0.40", options: { color: colors.danger } }],
  ["baseline_stage3", "2.89", "2.56", { text: "-0.33", options: { color: colors.danger } }]
], {
  x: 0.5, y: 2.9, w: 5.5, h: 2.0,
  fontSize: 12, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// Limitations
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 6.2, y: 2.5, w: 3.3, h: 2.4,
  fill: { color: colors.white },
  shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 }
});

slide5.addText("局限性", {
  x: 6.4, y: 2.6, w: 3, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: colors.danger, bold: true
});

slide5.addText([
  { text: "✓ 移除部分思考开头", options: { breakLine: true } },
  { text: "✓ 移除后续对话生成", options: { breakLine: true } },
  { text: "✗ 剩余仍含思考句式", options: { breakLine: true } },
  { text: "✗ 治标不治本", options: { breakLine: true } }
], {
  x: 6.4, y: 3.0, w: 3, h: 1.8,
  fontSize: 11, fontFace: "Arial", color: colors.text
});

// ==================== Slide 6: Improvements ====================
let slide6 = pres.addSlide();
slide6.background = { color: colors.lightBg };

// Header
slide6.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.primary }
});
slide6.addText("改进建议", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial", color: colors.white, bold: true
});

// Short-term
slide6.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 2.8,
  fill: { color: colors.white },
  shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 }
});

slide6.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 0.5,
  fill: { color: colors.accent }
});

slide6.addText("短期方案 (推理时)", {
  x: 0.7, y: 1.1, w: 4, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.white, bold: true
});

slide6.addTable([
  [
    { text: "方法", options: { fill: { color: "E2E8F0" }, bold: true } },
    { text: "难度", options: { fill: { color: "E2E8F0" }, bold: true } },
    { text: "效果", options: { fill: { color: "E2E8F0" }, bold: true } }
  ],
  ["后处理过滤", "低", "中"],
  ["Stop Tokens", "低", "中高"],
  ["Prompt优化", "中", "高"]
], {
  x: 0.7, y: 1.6, w: 3.9, h: 1.5,
  fontSize: 12, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// Long-term
slide6.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 2.8,
  fill: { color: colors.white },
  shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 }
});

slide6.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 0.5,
  fill: { color: colors.primary }
});

slide6.addText("长期方案 (训练时)", {
  x: 5.4, y: 1.1, w: 4, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.white, bold: true
});

slide6.addTable([
  [
    { text: "方法", options: { fill: { color: "E2E8F0" }, bold: true } },
    { text: "难度", options: { fill: { color: "E2E8F0" }, bold: true } },
    { text: "效果", options: { fill: { color: "E2E8F0" }, bold: true } }
  ],
  ["数据清洗", "中", "高"],
  ["SFT数据重构", "高", "最高"],
  ["DPO偏好优化", "高", "高"]
], {
  x: 5.4, y: 1.6, w: 3.9, h: 1.5,
  fontSize: 12, fontFace: "Arial",
  border: { pt: 0.5, color: "CBD5E1" },
  fill: { color: colors.white },
  color: colors.text,
  align: "center",
  valign: "middle"
});

// Recommendation
slide6.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.0, w: 9, h: 1.3,
  fill: { color: "D1FAE5" },
  line: { color: colors.success, width: 2 }
});

slide6.addText("⭐ 推荐配置", {
  x: 0.7, y: 4.1, w: 8.6, h: 0.35,
  fontSize: 16, fontFace: "Arial", color: colors.success, bold: true
});

slide6.addText("Minimal (6层注入) — LLM Judge评分最高 (3.13)，Stage 3达到最佳效果 (3.3)，平衡了注入层数与泛化能力", {
  x: 0.7, y: 4.5, w: 8.6, h: 0.7,
  fontSize: 14, fontFace: "Arial", color: colors.text
});

// ==================== Slide 7: Conclusion ====================
let slide7 = pres.addSlide();
slide7.background = { color: colors.primary };

// Title
slide7.addText("结论", {
  x: 0.5, y: 0.8, w: 9, h: 0.6,
  fontSize: 36, fontFace: "Arial", color: colors.white, bold: true,
  align: "center"
});

// Conclusions
const conclusions = [
  "Minimal 配置表现最佳，建议后续实验以此为基础",
  "思考过程泄露是主要问题，需从训练数据层面解决",
  "Loss 不是唯一指标，应结合 LLM Judge 评分综合评估",
  "后处理有一定效果但有限，根本解决需改进训练流程"
];

conclusions.forEach((c, i) => {
  slide7.addShape(pres.shapes.OVAL, {
    x: 1.5, y: 1.6 + i * 0.9, w: 0.35, h: 0.35,
    fill: { color: colors.accent }
  });

  slide7.addText((i + 1).toString(), {
    x: 1.5, y: 1.6 + i * 0.9, w: 0.35, h: 0.35,
    fontSize: 14, fontFace: "Arial", color: colors.white, bold: true,
    align: "center", valign: "middle"
  });

  slide7.addText(c, {
    x: 2.0, y: 1.6 + i * 0.9, w: 7, h: 0.7,
    fontSize: 18, fontFace: "Arial", color: colors.white
  });
});

// Decorative line
slide7.addShape(pres.shapes.LINE, {
  x: 3, y: 5.0, w: 4, h: 0,
  line: { color: colors.accent, width: 3 }
});

// Thank you
slide7.addText("Thank You", {
  x: 0.5, y: 5.1, w: 9, h: 0.4,
  fontSize: 20, fontFace: "Arial", color: colors.secondary,
  align: "center", italic: true
});

// Save presentation
pres.writeFile({ fileName: "/home/kemove/Desktop/PersonaSteer/docs/PersonaSteer_Training_Report.pptx" })
  .then(() => console.log("PPT created successfully!"))
  .catch(err => console.error("Error:", err));