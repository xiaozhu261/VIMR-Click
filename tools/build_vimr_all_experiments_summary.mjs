import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outDir = path.join(root, "outputs", "vimr_all_experiments_summary");
const previewDir = path.join(outDir, "previews");
const xlsxPath = path.join(outDir, "VIMR_all_experiments_summary.xlsx");
const reportPath = path.join(outDir, "VIMR_all_experiments_report.md");
const qaPath = path.join(outDir, "workbook_qa.json");

await fs.mkdir(previewDir, { recursive: true });

async function loadJson(relativePath) {
  return JSON.parse(await fs.readFile(path.join(root, relativePath), "utf8"));
}

const sources = {
  old: "outputs/vimr_coco_protocol/results.json",
  s0: "outputs/vimr_safe_update_sweep/S0/results.json",
  s1: "outputs/vimr_safe_update_sweep/S1/results.json",
  s2: "outputs/vimr_safe_update_sweep/S2/results.json",
  s3: "outputs/vimr_safe_update_sweep/S3/results.json",
  fusionOnly: "outputs/vimr_component_ablation/fusion_only/results.json",
  viewportComponent: "outputs/vimr_component_ablation/viewport/results.json",
  historyComponent: "outputs/vimr_component_ablation/history/results.json",
  residualComponent: "outputs/vimr_component_ablation/residual/results.json",
  viewportSignal: "outputs/vimr_viewport_signal_ablation/results.json",
  fullMatched: "outputs/vimr_router_detail_ablation/full_matched/results.json",
  router: "outputs/vimr_router_detail_ablation/router/results.json",
  routerDetail: "outputs/vimr_router_detail_ablation/router_detail/results.json",
  routerUsage: "outputs/vimr_router_detail_ablation/router_usage.json",
  quick: "outputs/vimr_coco_quick/summary.json",
  smokeRouter: "outputs/_smoke_router_detail/results.json",
  smokeViewport: "outputs/_smoke_viewport_signal/results.json"
};

const data = {};
for (const [key, value] of Object.entries(sources)) {
  data[key] = await loadJson(value);
}

const metricOrder = [
  "mIoU@1", "mIoU@2", "mIoU@3", "mIoU@5",
  "NoC@85", "Success@85/5", "NoC@90", "Success@90/5",
  "RegressionRate", "MeanRegressionMagnitude",
  "FinalSuccess@85/5", "LostAfterSuccess@85/5",
  "FinalSuccess@90/5", "LostAfterSuccess@90/5",
  "RegressionRate@4", "RegressionRate@5",
  "LateRegressionRate@4-5", "LateMeanRegressionMagnitude@4-5",
  "LateSevereRegressionRate@4-5>0.05",
  "LateSevereRegressionRate@4-5>0.10"
];

const keyMetrics = [
  "mIoU@1", "mIoU@2", "mIoU@3", "mIoU@5",
  "NoC@85", "NoC@90", "Success@85/5", "Success@90/5",
  "RegressionRate", "MeanRegressionMagnitude",
  "LateRegressionRate@4-5", "LateMeanRegressionMagnitude@4-5",
  "LateSevereRegressionRate@4-5>0.05"
];

function metricDirection(metric) {
  return metric.startsWith("mIoU") || metric.startsWith("Success") || metric.startsWith("FinalSuccess") ? "↑" : "↓";
}

function aggregateEntry(block, metric) {
  const entry = block[metric];
  if (!entry) return { mean: null, std: null, baseline: null, delta: null };
  const mean = entry.vimr_mean ?? entry.mean ?? null;
  const std = entry.vimr_std ?? entry.std ?? null;
  const baseline = entry.baseline ?? null;
  const delta = entry.delta_mean ?? entry.delta_vs_baseline ?? (mean != null && baseline != null ? mean - baseline : null);
  return { mean, std, baseline, delta };
}

function normalRun(family, model, sourceKey, status, comparisonGroup, note) {
  const result = data[sourceKey];
  const seedItems = Object.values(result.seeds || {});
  const firstSeed = seedItems[0] || {};
  const aggregate = {};
  for (const metric of metricOrder) aggregate[metric] = aggregateEntry(result.aggregate || {}, metric);
  return {
    family, model, sourceKey, status, comparisonGroup, note,
    source: sources[sourceKey],
    protocol: result.protocol || {},
    aggregate,
    seeds: seedItems.map(item => ({
      seed: item.seed,
      bestEpoch: item.best_epoch,
      parameters: item.trainable_vimr_parameters ?? item.trainable_parameters ?? null,
      metrics: item.vimr_metrics || item.metrics || {},
      epochHistory: item.epoch_history || []
    })),
    parameters: firstSeed.trainable_vimr_parameters ?? firstSeed.trainable_parameters ?? null,
    bestEpochs: seedItems.map(item => item.best_epoch).join("/")
  };
}

function viewportRun(model, mode, status, note) {
  const result = data.viewportSignal;
  const seedItems = Object.values(result.modes[mode] || {});
  const aggregate = {};
  for (const metric of metricOrder) aggregate[metric] = aggregateEntry(result.aggregates[mode] || {}, metric);
  return {
    family: "Viewport 信号", model, sourceKey: "viewportSignal", status,
    comparisonGroup: "P4-viewport-signal", note,
    source: sources.viewportSignal,
    protocol: result.protocol || {},
    aggregate,
    seeds: seedItems.map(item => ({
      seed: item.seed,
      bestEpoch: item.best_epoch,
      parameters: item.trainable_parameters ?? null,
      metrics: item.metrics || {},
      epochHistory: item.epoch_history || []
    })),
    parameters: seedItems[0]?.trainable_parameters ?? null,
    bestEpochs: seedItems.map(item => item.best_epoch).join("/")
  };
}

const uniqueRuns = [
  normalRun("历史正式协议", "R0 旧 Gate≈0.5", "old", "历史正式", "P1-old-gate", "旧初始化与后续正式实验不完全匹配，只作历史参考"),
  normalRun("Safe Update", "S0 (0.0/0.0)", "s0", "正式", "P2-safe-update", "无 safe/monotonic loss，gate init=0.1"),
  normalRun("Safe Update", "S1 (0.2/0.1)", "s1", "正式", "P2-safe-update", "保守 gate 下的原始权重"),
  normalRun("Safe Update", "S2 (0.5/0.2)", "s2", "正式", "P2-safe-update", "中等 Safe Update 权重"),
  normalRun("Safe Update", "S3 (1.0/0.5)", "s3", "正式", "P2-safe-update", "强 Safe Update；组件消融 Full VIMR 亦引用本结果"),
  normalRun("组件消融", "Fusion-only", "fusionOnly", "正式", "P3-components", "固定全局 viewport；保留与 +Viewport 相同的 encoder/fusion 参数量"),
  normalRun("组件消融", "+Viewport", "viewportComponent", "正式", "P3-components", "仅启用 viewport；safe/history/residual 关闭"),
  normalRun("组件消融", "+History", "historyComponent", "正式", "P3-components", "累计：viewport + history"),
  normalRun("组件消融", "+Residual", "residualComponent", "正式", "P3-components", "累计：viewport + history + residual，safe 关闭"),
  viewportRun("Default viewport", "default", "正式", "始终全局 viewport"),
  viewportRun("Synthetic viewport", "synthetic", "正式", "当前点击中心策略"),
  viewportRun("Shuffled viewport", "shuffled", "正式", "跨实例打乱完整 viewport 轨迹"),
  viewportRun("Correct oracle", "correct", "正式", "GT 实例框 oracle；不是实际用户遥测"),
  normalRun("Router/Detail", "Full matched", "fullMatched", "正式", "P5-router-matched", "与 Router/Detail 同代码初始化序列的匹配对照"),
  normalRun("Router/Detail", "+G/O/D Router", "router", "正式", "P5-router-matched", "加入 global/object/detail router"),
  normalRun("Router/Detail", "+Router + Detail Expert", "routerDetail", "正式", "P5-router-matched", "在 Router 上加入 Detail Expert")
];

const laterBaseline = data.s3.baseline;
const oldBaseline = data.old.baseline;

function baselineAggregate(metrics) {
  const result = {};
  for (const metric of metricOrder) {
    result[metric] = { mean: metrics[metric] ?? null, std: 0, baseline: metrics[metric] ?? null, delta: 0 };
  }
  return result;
}

const laterBaselineRun = {
  family: "Frozen baseline", model: "Frozen baseline", status: "正式", comparisonGroup: "P2/P3/P4",
  note: "后续正式协议共用的冻结 EfficientViT-SAM baseline", source: sources.s3,
  aggregate: baselineAggregate(laterBaseline), parameters: 0, bestEpochs: "—"
};

const oldBaselineRun = {
  family: "历史正式协议", model: "Frozen baseline (旧运行)", status: "历史正式", comparisonGroup: "P1-old-gate",
  note: "旧正式运行的 baseline；与后续 baseline 存在小差异", source: sources.old,
  aggregate: baselineAggregate(oldBaseline), parameters: 0, bestEpochs: "—"
};

function meanOf(run, metric) {
  return run.aggregate[metric]?.mean ?? null;
}

function stdOf(run, metric) {
  return run.aggregate[metric]?.std ?? null;
}

function colLetter(index) {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

const colors = {
  navy: "#16324F",
  blue: "#2F75B5",
  teal: "#2A9D8F",
  green: "#5B8C5A",
  orange: "#E9A23B",
  red: "#C94C4C",
  lightBlue: "#DCEAF7",
  lightGreen: "#E2F0D9",
  lightOrange: "#FCE4D6",
  lightRed: "#F4CCCC",
  lightGray: "#EEF1F4",
  border: "#CCD3DA",
  text: "#1F2933"
};

const workbook = Workbook.create();
const sheetNames = ["总览", "实验索引", "关键指标", "Safe Update", "组件消融", "Viewport", "Router Detail", "Seed 明细", "Epoch 明细", "非正式检查"];
const sheets = {};
for (const name of sheetNames) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheets[name] = sheet;
}

function titleBlock(sheet, title, subtitle, lastCol) {
  sheet.mergeCells("A2:" + lastCol + "2");
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2:" + lastCol + "2").format = {
    fill: colors.navy,
    font: { bold: true, color: "#FFFFFF", size: 18 },
    verticalAlignment: "center"
  };
  sheet.getRange("A2:" + lastCol + "2").format.rowHeight = 31;
  sheet.mergeCells("A3:" + lastCol + "3");
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3:" + lastCol + "3").format = {
    fill: colors.lightBlue,
    font: { color: colors.text, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center"
  };
  sheet.getRange("A3:" + lastCol + "3").format.rowHeight = 35;
}

function styleHeader(range) {
  range.format = {
    fill: colors.blue,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: colors.border }
  };
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format = {
    font: { color: colors.text, size: 10 },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: colors.border }
  };
}

function writeTable(sheet, startRow, headers, rows, tableName) {
  const endCol = colLetter(headers.length - 1);
  const endRow = startRow + rows.length;
  sheet.getRange("A" + startRow + ":" + endCol + startRow).values = [headers];
  if (rows.length) sheet.getRange("A" + (startRow + 1) + ":" + endCol + endRow).values = rows;
  styleHeader(sheet.getRange("A" + startRow + ":" + endCol + startRow));
  if (rows.length) styleBody(sheet.getRange("A" + (startRow + 1) + ":" + endCol + endRow));
  const table = sheet.tables.add("A" + startRow + ":" + endCol + endRow, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedColumns = false;
  return { startRow, endRow, endCol, table };
}

function protocolSeeds(protocol) {
  if (Array.isArray(protocol.training_seeds)) return protocol.training_seeds.join(",");
  if (protocol.seed != null) return String(protocol.seed);
  return "";
}

function componentLabel(protocol) {
  if (!protocol.components) return "";
  return Object.entries(protocol.components).filter(([, enabled]) => enabled).map(([name]) => name).join("+");
}

function comparisonRow(run, noteOverride) {
  return [
    run.model,
    run.parameters,
    run.bestEpochs,
    meanOf(run, "mIoU@1"),
    meanOf(run, "mIoU@2"),
    meanOf(run, "mIoU@3"),
    meanOf(run, "mIoU@5"),
    null,
    meanOf(run, "NoC@85"),
    meanOf(run, "NoC@90"),
    meanOf(run, "Success@85/5"),
    meanOf(run, "Success@90/5"),
    meanOf(run, "RegressionRate"),
    meanOf(run, "MeanRegressionMagnitude"),
    meanOf(run, "LateRegressionRate@4-5"),
    meanOf(run, "LateSevereRegressionRate@4-5>0.05"),
    noteOverride || run.note
  ];
}

function addComparisonSheet(sheet, title, subtitle, runs, referenceIndex, tableName) {
  titleBlock(sheet, title, subtitle, "Q");
  const headers = [
    "模型", "训练参数", "最佳 epoch", "mIoU@1", "mIoU@2", "mIoU@3", "mIoU@5",
    "Δ mIoU@5 vs 参考", "NoC@85", "NoC@90", "Success@85", "Success@90",
    "Regression Rate", "Mean Regression Mag.", "Late Regression", "Late Severe >0.05", "说明"
  ];
  const rows = runs.map(run => comparisonRow(run));
  const table = writeTable(sheet, 6, headers, rows, tableName);
  const firstDataRow = 7;
  const referenceRow = firstDataRow + referenceIndex;
  for (let i = 0; i < rows.length; i++) {
    const row = firstDataRow + i;
    sheet.getRange("H" + row).formulas = [["=G" + row + "-$G$" + referenceRow]];
  }
  sheet.getRange("B7:B" + table.endRow).setNumberFormat("#,##0");
  sheet.getRange("D7:H" + table.endRow).setNumberFormat("0.0000");
  sheet.getRange("I7:J" + table.endRow).setNumberFormat("0.000");
  sheet.getRange("K7:P" + table.endRow).setNumberFormat("0.0000");
  sheet.getRange("A" + referenceRow + ":Q" + referenceRow).format.fill = colors.lightBlue;
  sheet.freezePanes.freezeRows(6);
  const widths = [24, 13, 12, 11, 11, 11, 11, 15, 10, 10, 11, 11, 14, 17, 14, 16, 48];
  widths.forEach((width, i) => sheet.getRange(colLetter(i) + "1:" + colLetter(i) + table.endRow).format.columnWidth = width);
  return table;
}

// 总览
const overview = sheets["总览"];
titleBlock(overview, "VIMR-Click 全部实验汇总", "COCO 2017 小规模固定协议；正式比较采用同数据划分、点击序列、训练步数及兼容初始化。n=3，仅作描述性结论。", "N");
const s3Run = uniqueRuns.find(run => run.sourceKey === "s3");
const fusionOnlyRun = uniqueRuns.find(run => run.sourceKey === "fusionOnly");
const viewportComponentRun = uniqueRuns.find(run => run.sourceKey === "viewportComponent");
const routerRun = uniqueRuns.find(run => run.sourceKey === "router");
const detailRun = uniqueRuns.find(run => run.sourceKey === "routerDetail");
const matchedRun = uniqueRuns.find(run => run.sourceKey === "fullMatched");
const trajectoryHeaders = ["点击次数", "Frozen baseline", "Fusion-only", "+Viewport", "S3 Safe Update", "+Router"];
const trajectoryRows = [1, 2, 3, 5].map(click => [
  click,
  laterBaseline["mIoU@" + click],
  meanOf(fusionOnlyRun, "mIoU@" + click),
  meanOf(viewportComponentRun, "mIoU@" + click),
  meanOf(s3Run, "mIoU@" + click),
  meanOf(routerRun, "mIoU@" + click)
]);
writeTable(overview, 6, trajectoryHeaders, trajectoryRows, "OverviewTrajectory");
overview.getRange("B7:F10").setNumberFormat("0.0000");
const chart = overview.charts.add("line", overview.getRange("A6:F10"));
chart.title = "mIoU 随点击次数变化";
chart.titleTextStyle.fontSize = 12;
chart.titleTextStyle.typeface = "Arial";
chart.legend = { position: "top", textStyle: { typeface: "Arial", fontSize: 10 } };
chart.xAxis = { axisType: "textAxis", textStyle: { typeface: "Arial", fontSize: 10 } };
chart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: "Arial", fontSize: 10 } };
chart.setPosition("G6", "N20");

const decisions = [
  ["主模型选择", "S3 / Router 双口径", "S3 的 NoC@85=2.918、严重 late regression=0.0117；Router 在 matched 对照上 mIoU@5 +0.0202", "若论文强调安全交互，用 S3；若强调精度增益，报告 Router，同时披露稳定性代价"],
  ["Safe Update", "不是单一最优权重", "S1 mIoU@5 最高；S0 回退频率最低；S3 NoC@85 和严重回退最好", "按多目标权衡描述，不宣称 λ 越大越好"],
  ["提升来源", "Fusion-only 解释了大部分总提升", "Fusion-only mIoU@5=0.7853；+Viewport=0.7997，净增益仅 +0.0144，且一个 seed 为负", "Frozen baseline 只衡量总系统提升；模块归因必须从 Fusion-only 开始"],
  ["Viewport 语义", "证据不足", "Synthetic−Default=+0.0329，但 Synthetic−Shuffled 仅 +0.0012 且并非每个 seed 为正", "不能声称已学到独立 zoom/pan 意图"],
  ["Detail Expert", "当前不建议保留为主模型", "比 Router 的 mIoU@5 低 0.0019，早期 IoU 与 late regression 更差", "先解决路由塌缩，再重新训练"],
  ["科研风险", "初始化与 seed 方差", "旧 S3 不能直接对 Router；matched Full 的 mIoU@5 std=0.0424", "扩大 seed 与验证集，报告置信区间/配对检验"]
];
overview.getRange("A13:D13").values = [["问题", "结论", "关键证据", "建议写法/行动"]];
overview.getRange("A14:D19").values = decisions;
styleHeader(overview.getRange("A13:D13"));
styleBody(overview.getRange("A14:D19"));
overview.getRange("A14:D19").format.rowHeight = 54;
overview.getRange("A14:A19").format.font = { bold: true, color: colors.navy };
overview.getRange("A21:F21").values = [["模型", "mIoU@5", "std", "NoC@85", "Regression", "Late Regression"]];
overview.getRange("A22:F27").values = [
  ["Frozen baseline", laterBaseline["mIoU@5"], 0, laterBaseline["NoC@85"], laterBaseline.RegressionRate, laterBaseline["LateRegressionRate@4-5"]],
  ["Fusion-only", meanOf(fusionOnlyRun, "mIoU@5"), stdOf(fusionOnlyRun, "mIoU@5"), meanOf(fusionOnlyRun, "NoC@85"), meanOf(fusionOnlyRun, "RegressionRate"), meanOf(fusionOnlyRun, "LateRegressionRate@4-5")],
  ["+Viewport", meanOf(viewportComponentRun, "mIoU@5"), stdOf(viewportComponentRun, "mIoU@5"), meanOf(viewportComponentRun, "NoC@85"), meanOf(viewportComponentRun, "RegressionRate"), meanOf(viewportComponentRun, "LateRegressionRate@4-5")],
  ["S3 Safe Update", meanOf(s3Run, "mIoU@5"), stdOf(s3Run, "mIoU@5"), meanOf(s3Run, "NoC@85"), meanOf(s3Run, "RegressionRate"), meanOf(s3Run, "LateRegressionRate@4-5")],
  ["Full matched", meanOf(matchedRun, "mIoU@5"), stdOf(matchedRun, "mIoU@5"), meanOf(matchedRun, "NoC@85"), meanOf(matchedRun, "RegressionRate"), meanOf(matchedRun, "LateRegressionRate@4-5")],
  ["+Router", meanOf(routerRun, "mIoU@5"), stdOf(routerRun, "mIoU@5"), meanOf(routerRun, "NoC@85"), meanOf(routerRun, "RegressionRate"), meanOf(routerRun, "LateRegressionRate@4-5")]
];
styleHeader(overview.getRange("A21:F21"));
styleBody(overview.getRange("A22:F27"));
overview.getRange("B22:F27").setNumberFormat("0.0000");
overview.getRange("A27:F27").format.fill = colors.lightGreen;
overview.getRange("A1:A30").format.columnWidth = 23;
overview.getRange("B1:B30").format.columnWidth = 18;
overview.getRange("C1:C30").format.columnWidth = 24;
overview.getRange("D1:D30").format.columnWidth = 55;
overview.getRange("E1:F30").format.columnWidth = 14;
overview.freezePanes.freezeRows(3);

// 实验索引
const indexSheet = sheets["实验索引"];
titleBlock(indexSheet, "实验索引与可比性", "只有同一可比组内的结果可做严格增量归因；历史、quick 与 smoke 不进入论文主排名。", "R");
const indexHeaders = [
  "状态", "实验族", "模型", "可比组", "训练实例", "验证实例", "epochs", "最大点击", "seeds", "split seed",
  "λ_safe", "λ_monotonic", "gate init", "启用模块", "冻结基座", "最佳 epoch", "来源", "备注"
];
const indexRows = uniqueRuns.map(run => {
  const p = run.protocol;
  return [
    run.status, run.family, run.model, run.comparisonGroup,
    p.train_instances ?? null, p.validation_instances ?? p.eval_instances ?? null,
    p.epochs ?? null, p.max_clicks ?? null, protocolSeeds(p), p.split_seed ?? null,
    p.lambda_safe ?? null, p.lambda_monotonic ?? null, p.gate_init_probability ?? null,
    componentLabel(p), p.base_modules_frozen ?? true, run.bestEpochs, run.source, run.note
  ];
});
indexRows.push(
  ["工程验证", "Quick", "Quick baseline/VIMR", "不比较", 64, 32, 1, 3, "42", "", "", "", "", "快速端到端验证", true, "1", sources.quick, "样本、点击数与 seed 均不足"],
  ["工程验证", "Smoke", "Router+Detail smoke", "不比较", 2, 2, 1, 3, "7", 2026, 1, 0.5, 0.1, "全部模块", true, "1", sources.smokeRouter, "只验证代码可运行"],
  ["工程验证", "Smoke", "Viewport signal smoke", "不比较", 2, 2, 1, 5, "7", 2026, "", "", "", "viewport only", true, "1", sources.smokeViewport, "只验证四种模式管线"]
);
const indexTable = writeTable(indexSheet, 6, indexHeaders, indexRows, "ExperimentIndex");
indexSheet.getRange("A7:A" + indexTable.endRow).conditionalFormats.add("containsText", { text: "工程验证", format: { fill: colors.lightOrange, font: { color: colors.red } } });
indexSheet.getRange("A7:A" + indexTable.endRow).conditionalFormats.add("containsText", { text: "历史", format: { fill: colors.lightGray } });
indexSheet.freezePanes.freezeRows(6);
const indexWidths = [12, 14, 26, 20, 11, 11, 8, 10, 11, 11, 10, 13, 10, 30, 10, 15, 48, 52];
indexWidths.forEach((w, i) => indexSheet.getRange(colLetter(i) + "1:" + colLetter(i) + indexTable.endRow).format.columnWidth = w);

// 关键指标长表
const metricSheet = sheets["关键指标"];
titleBlock(metricSheet, "正式实验关键指标（长表）", "每一行是一个模型-指标汇总；Baseline 与 Δ 均来自对应运行，避免跨协议误差。", "M");
const metricHeaders = ["状态", "实验族", "模型", "可比组", "指标", "方向", "均值", "标准差", "运行内 Baseline", "Δ vs Baseline", "参数量", "最佳 epoch", "来源"];
const metricRows = [];
for (const run of [laterBaselineRun, oldBaselineRun, ...uniqueRuns]) {
  for (const metric of metricOrder) {
    const entry = run.aggregate[metric];
    if (!entry || entry.mean == null) continue;
    metricRows.push([
      run.status, run.family, run.model, run.comparisonGroup, metric, metricDirection(metric),
      entry.mean, entry.std, entry.baseline, entry.delta, run.parameters, run.bestEpochs, run.source
    ]);
  }
}
const metricTable = writeTable(metricSheet, 6, metricHeaders, metricRows, "FormalMetrics");
metricSheet.getRange("G7:J" + metricTable.endRow).setNumberFormat("0.000000");
metricSheet.getRange("K7:K" + metricTable.endRow).setNumberFormat("#,##0");
metricSheet.freezePanes.freezeRows(6);
const metricWidths = [12, 16, 27, 20, 34, 8, 14, 14, 16, 15, 14, 15, 48];
metricWidths.forEach((w, i) => metricSheet.getRange(colLetter(i) + "1:" + colLetter(i) + metricTable.endRow).format.columnWidth = w);

// Safe Update
const safeRuns = [
  uniqueRuns.find(run => run.sourceKey === "s0"),
  uniqueRuns.find(run => run.sourceKey === "s1"),
  uniqueRuns.find(run => run.sourceKey === "s2"),
  uniqueRuns.find(run => run.sourceKey === "s3"),
  uniqueRuns.find(run => run.sourceKey === "old")
];
const safeTable = addComparisonSheet(
  sheets["Safe Update"],
  "Safe Update 权重扫描",
  "S0–S3 为严格同协议比较，参考行为 S0；R0 旧 Gate≈0.5 仅列作历史证据，不能据此归因 gate init。",
  safeRuns,
  0,
  "SafeUpdateComparison"
);
sheets["Safe Update"].getRange("A11:Q11").format.fill = colors.lightGray;
sheets["Safe Update"].getRange("A14:Q16").values = [
  ["观察", "S1 的 mIoU@5 最高；S0 的 Regression/Late Regression 最低；S3 的 NoC@85、Success@85 和严重 late regression 更优。"],
  ["结论", "权重增大没有单调改善。S3 是偏向用户效率与严重回退控制的多目标选择，不是所有指标的全局最优。"],
  ["论文措辞", "Safe Update 显著降低回退幅度与严重回退；对普通回退频率和峰值 mIoU 的影响取决于权重。"]
];
sheets["Safe Update"].mergeCells("B14:Q14");
sheets["Safe Update"].mergeCells("B15:Q15");
sheets["Safe Update"].mergeCells("B16:Q16");
styleBody(sheets["Safe Update"].getRange("A14:Q16"));
sheets["Safe Update"].getRange("A14:A16").format = { fill: colors.lightBlue, font: { bold: true, color: colors.navy }, borders: { preset: "all", style: "thin", color: colors.border } };

// 组件消融
const componentRuns = [
  laterBaselineRun,
  fusionOnlyRun,
  uniqueRuns.find(run => run.sourceKey === "viewportComponent"),
  uniqueRuns.find(run => run.sourceKey === "historyComponent"),
  uniqueRuns.find(run => run.sourceKey === "residualComponent"),
  Object.assign({}, s3Run, { model: "Full VIMR (S3 reference)", note: "外部引用 S3；相同后续正式协议" })
];
addComparisonSheet(
  sheets["组件消融"],
  "累计组件消融",
  "Baseline → Fusion-only → Viewport → History → Residual → Safe Update。最后一行为 S3 引用；Δ 列相对 Frozen baseline。",
  componentRuns,
  0,
  "ComponentAblation"
);
sheets["组件消融"].getRange("A14:Q18").values = [
  ["增量", "Fusion-only 相对其运行内 Frozen baseline：mIoU@5 +0.1061，说明训练适配贡献了大部分总提升。"],
  ["增量", "+Viewport 相对 Fusion-only：mIoU@5 +0.0144；逐 seed 为 +0.0326/-0.0058/+0.0164，增益不完全一致。"],
  ["增量", "+History 相对 +Viewport：mIoU@5 +0.0124；+Residual 再 +0.0059，但回退率上升。"],
  ["增量", "加入 Safe Update 后 mIoU@5 -0.0007，Regression 0.4471→0.3604，贡献主要是稳定性。"],
  ["限制", "Fusion-only 为本次新运行，+Viewport 为复用旧结果；两次 Frozen baseline 相差 0.0021，提示轻微 GPU 非确定性。"]
];
sheets["组件消融"].mergeCells("B14:Q14");
sheets["组件消融"].mergeCells("B15:Q15");
sheets["组件消融"].mergeCells("B16:Q16");
sheets["组件消融"].mergeCells("B17:Q17");
sheets["组件消融"].mergeCells("B18:Q18");
styleBody(sheets["组件消融"].getRange("A14:Q18"));
sheets["组件消融"].getRange("A14:A17").format = { fill: colors.lightGreen, font: { bold: true, color: colors.navy }, borders: { preset: "all", style: "thin", color: colors.border } };
sheets["组件消融"].getRange("A18").format = { fill: colors.lightOrange, font: { bold: true, color: colors.red }, borders: { preset: "all", style: "thin", color: colors.border } };

// Viewport
const viewportRuns = [
  uniqueRuns.find(run => run.model === "Default viewport"),
  uniqueRuns.find(run => run.model === "Synthetic viewport"),
  uniqueRuns.find(run => run.model === "Shuffled viewport"),
  uniqueRuns.find(run => run.model === "Correct oracle")
];
addComparisonSheet(
  sheets["Viewport"],
  "Viewport 信号消融",
  "四种模式共享冻结 baseline 生成的点击序列；参考 Default。Correct 是 GT 框 oracle，不代表真实用户轨迹。",
  viewportRuns,
  0,
  "ViewportSignal"
);
const paired = data.viewportSignal.paired_contrasts;
const criterion = data.viewportSignal.viewport_signal_criterion;
sheets["Viewport"].getRange("A14:F14").values = [["配对对比", "mIoU@5 均值差", "std", "seed 42", "seed 43", "seed 44"]];
const pairedRows = [
  ["Synthetic - Default", paired.synthetic_minus_default["mIoU@5"].mean, paired.synthetic_minus_default["mIoU@5"].std, ...paired.synthetic_minus_default["mIoU@5"].per_seed],
  ["Synthetic - Shuffled", paired.synthetic_minus_shuffled["mIoU@5"].mean, paired.synthetic_minus_shuffled["mIoU@5"].std, ...paired.synthetic_minus_shuffled["mIoU@5"].per_seed],
  ["Correct - Synthetic", paired.correct_minus_synthetic["mIoU@5"].mean, paired.correct_minus_synthetic["mIoU@5"].std, ...paired.correct_minus_synthetic["mIoU@5"].per_seed]
];
sheets["Viewport"].getRange("A15:F17").values = pairedRows;
styleHeader(sheets["Viewport"].getRange("A14:F14"));
styleBody(sheets["Viewport"].getRange("A15:F17"));
sheets["Viewport"].getRange("B15:F17").setNumberFormat("0.0000");
sheets["Viewport"].getRange("A19:Q21").values = [
  ["预注册判据", criterion.definition],
  ["结果", "Synthetic−Default=" + criterion["synthetic_minus_default_mIoU@5"].toFixed(4) + "；Synthetic−Shuffled=" + criterion["synthetic_minus_shuffled_mIoU@5"].toFixed(4) + "；每个 seed 均为正=" + criterion.positive_for_every_seed],
  ["结论", "未通过。当前结果只说明非默认 viewport 条件有帮助，不能证明模型利用了与点击独立的 zoom/pan 语义。"]
];
sheets["Viewport"].mergeCells("B19:Q19");
sheets["Viewport"].mergeCells("B20:Q20");
sheets["Viewport"].mergeCells("B21:Q21");
styleBody(sheets["Viewport"].getRange("A19:Q21"));
sheets["Viewport"].getRange("A19:A21").format = { fill: colors.lightRed, font: { bold: true, color: colors.red }, borders: { preset: "all", style: "thin", color: colors.border } };

// Router/Detail
const routerRuns = [matchedRun, routerRun, detailRun];
addComparisonSheet(
  sheets["Router Detail"],
  "Global/Object/Detail Router 与 Detail Expert",
  "必须使用 Full matched 作为参考；旧 S3 因新增模块改变随机初始化序列，不用于严格增量比较。",
  routerRuns,
  0,
  "RouterDetailComparison"
);
const usageRows = [];
for (const [model, seeds] of Object.entries(data.routerUsage)) {
  for (const [seed, usage] of Object.entries(seeds)) {
    usageRows.push([
      model, Number(seed),
      usage.mean_probabilities[0], usage.mean_probabilities[1], usage.mean_probabilities[2],
      usage.argmax_fractions[0], usage.argmax_fractions[1], usage.argmax_fractions[2]
    ]);
  }
}
sheets["Router Detail"].getRange("A13:H13").values = [["模型", "seed", "P(global)", "P(object)", "P(detail)", "Argmax global", "Argmax object", "Argmax detail"]];
sheets["Router Detail"].getRange("A14:H" + (13 + usageRows.length)).values = usageRows;
styleHeader(sheets["Router Detail"].getRange("A13:H13"));
styleBody(sheets["Router Detail"].getRange("A14:H" + (13 + usageRows.length)));
sheets["Router Detail"].getRange("C14:H" + (13 + usageRows.length)).setNumberFormat("0.0000");
const routerNoteStart = 15 + usageRows.length;
sheets["Router Detail"].getRange("A" + routerNoteStart + ":Q" + (routerNoteStart + 2)).values = [
  ["Router 结论", "相对 Full matched：mIoU@5 +0.0202，std 0.0424→0.0032；但 Regression 与 Late Regression 略升。"],
  ["Detail 结论", "相对 Router：mIoU@5 -0.0019，早期 mIoU 和 late-stage stability 更差，当前没有额外收益。"],
  ["路由诊断", "Router+Detail 三个 seed 的 argmax 全落在 Detail，出现路由塌缩；Router-only 也没有稳定的语义分工。"]
];
sheets["Router Detail"].mergeCells("B" + routerNoteStart + ":Q" + routerNoteStart);
sheets["Router Detail"].mergeCells("B" + (routerNoteStart + 1) + ":Q" + (routerNoteStart + 1));
sheets["Router Detail"].mergeCells("B" + (routerNoteStart + 2) + ":Q" + (routerNoteStart + 2));
styleBody(sheets["Router Detail"].getRange("A" + routerNoteStart + ":Q" + (routerNoteStart + 2)));
sheets["Router Detail"].getRange("A" + routerNoteStart + ":A" + (routerNoteStart + 2)).format = { fill: colors.lightOrange, font: { bold: true, color: colors.navy }, borders: { preset: "all", style: "thin", color: colors.border } };

// Seed 明细
const seedSheet = sheets["Seed 明细"];
titleBlock(seedSheet, "正式实验 Seed 明细", "保存每个正式运行在最佳 epoch 的全部指标，用于复核均值、方差与配对差异。", "J");
const seedHeaders = ["状态", "实验族", "模型", "seed", "最佳 epoch", "参数量", "指标", "值", "可比组", "来源"];
const seedRows = [];
for (const run of uniqueRuns) {
  for (const seed of run.seeds) {
    for (const [metric, value] of Object.entries(seed.metrics)) {
      seedRows.push([run.status, run.family, run.model, seed.seed, seed.bestEpoch, seed.parameters, metric, value, run.comparisonGroup, run.source]);
    }
  }
}
const seedTable = writeTable(seedSheet, 6, seedHeaders, seedRows, "SeedMetrics");
seedSheet.getRange("F7:F" + seedTable.endRow).setNumberFormat("#,##0");
seedSheet.getRange("H7:H" + seedTable.endRow).setNumberFormat("0.000000");
seedSheet.freezePanes.freezeRows(6);
const seedWidths = [12, 17, 29, 9, 12, 14, 36, 14, 22, 52];
seedWidths.forEach((w, i) => seedSheet.getRange(colLetter(i) + "1:" + colLetter(i) + seedTable.endRow).format.columnWidth = w);

// Epoch 明细（精简但覆盖每个 epoch 的核心训练/验证轨迹）
const epochSheet = sheets["Epoch 明细"];
titleBlock(epochSheet, "每 Epoch 验证轨迹", "记录每个正式训练 run 的每个 epoch：训练 IoU/梯度、mIoU@5、NoC 与回退指标。", "N");
const epochHeaders = ["状态", "实验族", "模型", "seed", "epoch", "是否最佳", "Train IoU@max", "Gradient norm", "mIoU@1", "mIoU@5", "NoC@85", "NoC@90", "Regression", "Late Regression"];
const epochRows = [];
for (const run of uniqueRuns) {
  for (const seed of run.seeds) {
    for (const item of seed.epochHistory) {
      epochRows.push([
        run.status, run.family, run.model, seed.seed, item.epoch, item.epoch === seed.bestEpoch,
        item.mean_train_iou_at_max_clicks ?? null, item.mean_gradient_norm ?? null,
        item["mIoU@1"] ?? null, item["mIoU@5"] ?? null,
        item["NoC@85"] ?? null, item["NoC@90"] ?? null,
        item.RegressionRate ?? null, item["LateRegressionRate@4-5"] ?? null
      ]);
    }
  }
}
const epochTable = writeTable(epochSheet, 6, epochHeaders, epochRows, "EpochMetrics");
epochSheet.getRange("G7:N" + epochTable.endRow).setNumberFormat("0.000000");
epochSheet.freezePanes.freezeRows(6);
const epochWidths = [12, 17, 29, 9, 9, 11, 15, 15, 13, 13, 11, 11, 13, 15];
epochWidths.forEach((w, i) => epochSheet.getRange(colLetter(i) + "1:" + colLetter(i) + epochTable.endRow).format.columnWidth = w);

// 非正式检查
const nonformalSheet = sheets["非正式检查"];
titleBlock(nonformalSheet, "Quick / Smoke 工程检查", "这些结果只用于证明训练、评估和四模式/Router 管线可运行；不进入科研结论。", "K");
const nonformalHeaders = ["类型", "模型", "训练实例", "验证实例", "epochs", "点击", "seed", "指标", "值", "来源", "备注"];
const nonformalRows = [];
for (const [model, metrics] of Object.entries(data.quick.metrics)) {
  for (const [metric, value] of Object.entries(metrics)) {
    nonformalRows.push(["Quick", model, 64, 32, 1, 3, 42, metric, value, sources.quick, "不进入正式比较"]);
  }
}
for (const [metric, entry] of Object.entries(data.smokeRouter.aggregate)) {
  nonformalRows.push(["Smoke Router", "Router+Detail", 2, 2, 1, 3, 7, metric, entry.vimr_mean, sources.smokeRouter, "仅工程冒烟"]);
}
for (const [mode, aggregate] of Object.entries(data.smokeViewport.aggregates)) {
  for (const [metric, entry] of Object.entries(aggregate)) {
    nonformalRows.push(["Smoke Viewport", mode, 2, 2, 1, 5, 7, metric, entry.mean, sources.smokeViewport, "仅工程冒烟"]);
  }
}
const nonformalTable = writeTable(nonformalSheet, 6, nonformalHeaders, nonformalRows, "NonformalChecks");
nonformalSheet.getRange("I7:I" + nonformalTable.endRow).setNumberFormat("0.000000");
nonformalSheet.freezePanes.freezeRows(6);
const nonformalWidths = [18, 23, 11, 11, 9, 9, 9, 36, 14, 52, 20];
nonformalWidths.forEach((w, i) => nonformalSheet.getRange(colLetter(i) + "1:" + colLetter(i) + nonformalTable.endRow).format.columnWidth = w);

// Common fonts and tab colors.
for (const [name, sheet] of Object.entries(sheets)) {
  const used = sheet.getUsedRange();
  if (used) used.format.font = { name: "Arial", color: colors.text, size: 10 };
  sheet.tabColor = ["总览", "Safe Update", "组件消融", "Viewport", "Router Detail"].includes(name) ? colors.blue : colors.teal;
}
// Restore title/header typography after applying the common font.
for (const sheet of Object.values(sheets)) {
  sheet.getRange("A2").format.font = { name: "Arial", bold: true, color: "#FFFFFF", size: 18 };
}

workbook.recalculate();

const formulaChecks = [];
for (const name of ["Safe Update", "组件消融", "Viewport", "Router Detail"]) {
  const sheet = sheets[name];
  const used = sheet.getUsedRange();
  const values = used?.values || [];
  const errorCells = [];
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < (values[r] || []).length; c++) {
      const value = values[r][c];
      if (typeof value === "string" && /^#(REF|DIV\/0|VALUE|NAME|N\/A|NUM|NULL)!?/.test(value)) {
        errorCells.push(colLetter(c) + String(r + 1) + "=" + value);
      }
    }
  }
  formulaChecks.push({ sheet: name, errorCells });
}

const inspectSummary = await workbook.inspect({
  kind: "workbook,sheet,table,drawing",
  maxChars: 12000,
  tableMaxRows: 4,
  tableMaxCols: 8,
  tableMaxCellChars: 80
});
const keyRanges = {};
for (const [name, range] of Object.entries({
  "总览": "A2:N25",
  "Safe Update": "A2:Q16",
  "组件消融": "A2:Q16",
  "Viewport": "A2:Q21",
  "Router Detail": "A2:Q23"
})) {
  keyRanges[name] = (await workbook.inspect({ kind: "region", sheetId: name, range, maxChars: 10000 })).ndjson;
}

for (const [name, sheet] of Object.entries(sheets)) {
  const scale = ["Seed 明细", "Epoch 明细", "关键指标", "非正式检查"].includes(name) ? 0.22 : 0.65;
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale, format: "png" });
  const safeName = name.replace(/[ /]/g, "_");
  await fs.writeFile(path.join(previewDir, safeName + ".png"), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);

const reloaded = await SpreadsheetFile.importXlsx(await FileBlob.load(xlsxPath));
const reloadInspect = await reloaded.inspect({ kind: "sheet,table,drawing", maxChars: 8000, tableMaxRows: 2, tableMaxCols: 6 });

const fmt = value => value == null ? "—" : Number(value).toFixed(4);
const pm = run => fmt(meanOf(run, "mIoU@5")) + " ± " + fmt(stdOf(run, "mIoU@5"));
const safeLine = run => [
  run.model,
  run.protocol.lambda_safe ?? "—",
  run.protocol.lambda_monotonic ?? "—",
  run.protocol.gate_init_probability ?? "≈0.5",
  pm(run),
  fmt(meanOf(run, "NoC@85")),
  fmt(meanOf(run, "RegressionRate")),
  fmt(meanOf(run, "LateRegressionRate@4-5")),
  fmt(meanOf(run, "LateSevereRegressionRate@4-5>0.05"))
].join(" | ");
const componentLine = run => [
  run.model,
  run.parameters == null ? "—" : run.parameters.toLocaleString("en-US"),
  pm(run),
  fmt(meanOf(run, "RegressionRate")),
  fmt(meanOf(run, "LateRegressionRate@4-5"))
].join(" | ");
const viewportLine = run => [
  run.model,
  pm(run),
  fmt(meanOf(run, "NoC@85")),
  fmt(meanOf(run, "RegressionRate")),
  fmt(meanOf(run, "LateRegressionRate@4-5"))
].join(" | ");
const routerLine = run => [
  run.model,
  run.parameters.toLocaleString("en-US"),
  pm(run),
  fmt(meanOf(run, "NoC@85")),
  fmt(meanOf(run, "RegressionRate")),
  fmt(meanOf(run, "LateRegressionRate@4-5"))
].join(" | ");

const report = [
  "# VIMR-Click 全部实验整理与分析",
  "",
  "生成日期：2026-09-09",
  "",
  "## 1. 结论摘要",
  "",
  "1. Fusion-only 已确认训练适配是总提升的主要来源：mIoU@5=" + fmt(meanOf(fusionOnlyRun, "mIoU@5")) + "，相对其运行内 Frozen baseline 提升 +" + fmt(fusionOnlyRun.aggregate["mIoU@5"].delta) + "。+Viewport 相对 Fusion-only 的净增益只有 +" + fmt(meanOf(viewportComponentRun, "mIoU@5") - meanOf(fusionOnlyRun, "mIoU@5")) + "，且一个 seed 为负。",
  "2. 当前可归因的模块链条是：Viewport 提供小幅、不完全一致的增益；History 提供持续增益；Residual 进一步提高精度；Safe Update 主要降低更新破坏与严重回退。",
  "3. Safe Update 权重不存在单调最优：S1 的 mIoU@5 最高，S0 的普通回退频率最低，S3 的 NoC@85、Success@85 与严重 late-stage regression 更好。S3 是偏向交互效率与安全性的多目标选择。",
  "4. Router 在严格 matched 对照上把 mIoU@5 从 " + fmt(meanOf(matchedRun, "mIoU@5")) + " 提高到 " + fmt(meanOf(routerRun, "mIoU@5")) + "（+" + fmt(meanOf(routerRun, "mIoU@5") - meanOf(matchedRun, "mIoU@5")) + "），同时把 seed 标准差从 " + fmt(stdOf(matchedRun, "mIoU@5")) + " 降到 " + fmt(stdOf(routerRun, "mIoU@5")) + "；但普通和 late 回退率略有上升。",
  "5. Detail Expert 没有在 Router 上继续增益：mIoU@5 下降 " + fmt(meanOf(routerRun, "mIoU@5") - meanOf(detailRun, "mIoU@5")) + "，早期 IoU 与 late-stage stability 也更差；其路由 argmax 在三个 seed 上全部塌缩到 Detail。",
  "6. Viewport 的独立语义证据没有成立：Synthetic−Default 为 +" + fmt(criterion["synthetic_minus_default_mIoU@5"]) + "，但 Synthetic−Shuffled 只有 +" + fmt(criterion["synthetic_minus_shuffled_mIoU@5"]) + "，且不是每个 seed 都为正。当前不能声称模型真正理解了用户 zoom/pan 意图。",
  "",
  "## 2. 统一实验口径",
  "",
  "- 数据：COCO 2017 train split 的固定 image-disjoint holdout；每图取最大合格非 crowd 实例，面积比例 0.01–0.60。",
  "- 正式协议：500 个训练实例、200 个验证实例、3 epochs、最多 5 次点击、seeds 42/43/44、split seed 2026。",
  "- 点击：首次点击 GT 距离变换最大点；后续点击预测误差最大点，正负标签取自 GT。",
  "- 冻结：encoder、prompt encoder、mask decoder 均冻结，只训练 VIMR 新增模块。",
  "- NoC 未达到阈值时记为 6；Success 表示 5 次内曾达到阈值；FinalSuccess 表示第 5 次后仍保持。",
  "- n=3 只支持描述性结论；本报告不把 quick/smoke 结果用于排名或显著性声称。",
  "",
  "## 3. Safe Update 权重扫描",
  "",
  "模型 | λ_safe | λ_monotonic | gate init | mIoU@5 | NoC@85 | Regression | Late Regression | Late Severe>0.05",
  "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
  ...safeRuns.map(safeLine),
  "",
  "分析：S1 的 mIoU@5 最高（" + fmt(meanOf(safeRuns[1], "mIoU@5")) + "），S0 的 Regression 最低（" + fmt(meanOf(safeRuns[0], "RegressionRate")) + "），S3 的 NoC@85 最低（" + fmt(meanOf(s3Run, "NoC@85")) + "）且严重 late regression 为 " + fmt(meanOf(s3Run, "LateSevereRegressionRate@4-5>0.05")) + "。因此应把 S3 表述为安全/效率偏好的折中，而不是整体最优。R0 采用旧 gate 初始化，不能与 S0–S3 做严格因果归因。",
  "",
  "## 4. 累计组件消融",
  "",
  "模型 | 训练参数 | mIoU@5 | Regression | Late Regression",
  "--- | ---: | ---: | ---: | ---:",
  ...componentRuns.map(componentLine),
  "",
  "分析：Fusion-only 已达到 " + fmt(meanOf(componentRuns[1], "mIoU@5")) + "，相对其运行内 Frozen baseline 提升 +" + fmt(componentRuns[1].aggregate["mIoU@5"].delta) + "。+Viewport 相对 Fusion-only 只增加 " + fmt(meanOf(componentRuns[2], "mIoU@5") - meanOf(componentRuns[1], "mIoU@5")) + "，逐 seed 差值为 +0.0326/-0.0058/+0.0164；+History 再增加 " + fmt(meanOf(componentRuns[3], "mIoU@5") - meanOf(componentRuns[2], "mIoU@5")) + "；+Residual 再增加 " + fmt(meanOf(componentRuns[4], "mIoU@5") - meanOf(componentRuns[3], "mIoU@5")) + "，但回退变多；加入 Safe Update 后 mIoU@5 基本不变（" + fmt(meanOf(componentRuns[5], "mIoU@5") - meanOf(componentRuns[4], "mIoU@5")) + "），Regression 从 " + fmt(meanOf(componentRuns[4], "RegressionRate")) + " 降至 " + fmt(meanOf(componentRuns[5], "RegressionRate")) + "。Fusion-only 是本次新运行，+Viewport 复用旧结果；两次 Frozen baseline 相差约 0.0021，说明净差值仍含少量 GPU 非确定性。",
  "",
  "## 5. Viewport 信号消融",
  "",
  "模式 | mIoU@5 | NoC@85 | Regression | Late Regression",
  "--- | ---: | ---: | ---: | ---:",
  ...viewportRuns.map(viewportLine),
  "",
  "预注册判据要求 Synthetic−Default 与 Synthetic−Shuffled 都至少 +0.01 且每个 seed 为正。实际第二项只有 +" + fmt(criterion["synthetic_minus_shuffled_mIoU@5"]) + "，判据失败。Synthetic 和 Shuffled 几乎持平，说明主要收益可能来自额外 viewport 条件/训练分布，而不是位置正确性。Correct oracle 比 Synthetic 高 " + fmt(paired.correct_minus_synthetic["mIoU@5"].mean) + "，但 seed 一致性仍不足。",
  "",
  "## 6. Router 与 Detail Expert",
  "",
  "模型 | 训练参数 | mIoU@5 | NoC@85 | Regression | Late Regression",
  "--- | ---: | ---: | ---: | ---: | ---:",
  ...routerRuns.map(routerLine),
  "",
  "分析：Router 的 matched 精度收益明确且方差更低，但稳定性指标略差。Detail Expert 相对 Router 没有收益，并且三个 seed 的 argmax 都选择 Detail Expert，属于路由塌缩，不能据此声称 global/object/detail 已形成可解释分工。",
  "",
  "## 7. 论文可用结论与暂不可用结论",
  "",
  "可用：",
  "",
  "- Fusion-only 控制证明监督训练的融合适配器解释了总提升的大部分；模块贡献必须相对该控制组报告。",
  "- History/Residual 在 Fusion-only 与 Viewport 之后仍提供递进增益。",
  "- Safe Update 显著降低回退幅度、LostAfterSuccess 和严重 late-stage regression；其权重体现精度与安全性的权衡。",
  "- Router 在 matched 设置下改善 mIoU@5 与跨 seed 稳定性。",
  "",
  "暂不可用：",
  "",
  "- 不能把 Frozen baseline 到 Full VIMR 的全部差值归因于 VIMR 创新模块。",
  "- 不能说 synthetic viewport 提供了独立于点击位置的用户意图，因为 shuffled 没有明显下降。",
  "- 不能说 Detail Expert 有效或路由具备清晰语义，因为性能未提升且路由塌缩。",
  "- 不能把旧 S3 与新 Router 直接对比；新增模块改变了初始化随机数序列，必须使用 Full matched。",
  "",
  "## 8. 下一步优先级",
  "",
  "1. 先做 Router 的 route balancing/entropy 或弱监督，使三路均被使用，再重训 Detail Expert。",
  "2. 为 viewport 构造与 click 去相关的真实或半真实 zoom/pan 轨迹，并加入 viewport-only、click-only、遮蔽/扰动对照。",
  "3. 将正式验证扩大到至少 5 seeds 和更大实例集，报告 paired bootstrap CI；必要时再上 COCO val2017。",
  "4. 主论文表建议同时报告 S3（安全性主模型）和 Router（精度主模型），不要挑单一指标决定唯一冠军。",
  "",
  "## 9. 数据文件",
  "",
  "- Excel 工作簿包含：实验索引、全部汇总指标、每 seed 指标、每 epoch 轨迹、四组消融、路由使用率和非正式检查。",
  "- 所有来源路径均在工作簿中逐行保留。"
].join("\n");
await fs.writeFile(reportPath, report, "utf8");

await fs.writeFile(qaPath, JSON.stringify({
  xlsxPath,
  reportPath,
  formulaChecks,
  inspectSummary: inspectSummary.ndjson,
  keyRanges,
  reloadInspect: reloadInspect.ndjson,
  renderedSheets: sheetNames.map(name => path.join(previewDir, name.replace(/[ /]/g, "_") + ".png"))
}, null, 2), "utf8");

console.log(JSON.stringify({
  xlsxPath,
  reportPath,
  qaPath,
  sheets: sheetNames.length,
  formalRuns: uniqueRuns.length,
  aggregateRows: metricRows.length,
  seedRows: seedRows.length,
  epochRows: epochRows.length,
  formulaChecks
}, null, 2));
