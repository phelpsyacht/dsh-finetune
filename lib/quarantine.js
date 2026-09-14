/**
 * Quarantine review artifacts for the dataset-cleaning loop.
 *
 * When `quarantine` is enabled, boundary samples (records dropped by a numeric
 * threshold rule within `quarantineMargin` of the threshold) are written to
 * <path>.quarantine.jsonl, and this module builds the companion CSV + HTML
 * report so a human can Reject/Accept them before re-tuning thresholds:
 *
 *   1. open <path>.quarantine.html (or .csv) and review the boundary rows
 *   2. decide, per rule, whether the current threshold is right
 *   3. re-run finetune_dataset_clean with adjusted thresholds
 *
 * The HTML report is fully self-contained (embedded CSS/JS, sortable table,
 * rule filter, per-rule summary with a suggested threshold based on the
 * boundary samples). No external assets.
 */

const RULE_DIRECTION = {
  min_chars: "under",
  max_chars: "over",
  repeat_ratio: "over",
  meaningful_ratio: "under",
  ppl: "over",
};

const RULE_LABEL_ZH = {
  min_chars: "min_chars（过短）",
  max_chars: "max_chars（过长）",
  repeat_ratio: "repeat_ratio（单字重复）",
  meaningful_ratio: "meaningful_ratio（符号垃圾）",
  ppl: "ppl（困惑度）",
};

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function previewJson(value, max = 240) {
  const text = JSON.stringify(value ?? "");
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Per-rule aggregates over the boundary records: count, metric range and a
 * suggested threshold (for "over" rules the loosest boundary metric, for
 * "under" rules the tightest — a reference point, not a recommendation).
 */
function ruleStats(records) {
  const stats = new Map();
  for (const entry of records) {
    const rule = entry.rule;
    const stat = stats.get(rule) ?? { count: 0, min: Infinity, max: -Infinity };
    stat.count += 1;
    if (typeof entry.metric === "number") {
      stat.min = Math.min(stat.min, entry.metric);
      stat.max = Math.max(stat.max, entry.metric);
    }
    stats.set(rule, stat);
  }
  for (const [rule, stat] of stats) {
    const direction = RULE_DIRECTION[rule];
    if (direction === "under") stat.suggested = stat.min;
    else if (direction === "over") stat.suggested = stat.max;
    else stat.suggested = null;
  }
  return stats;
}

/** CSV report: one row per boundary sample, decision column for the reviewer. */
export function quarantineCsv(records) {
  const header = ["line", "rule", "source", "metric", "threshold", "deviation", "decision", "record_preview"];
  const rows = records.map((entry) =>
    [
      entry.line,
      entry.rule,
      entry.source ?? "",
      entry.metric,
      entry.threshold,
      entry.deviation,
      entry.decision ?? "",
      previewJson(entry.record, 400),
    ]
      .map(escapeCsv)
      .join(","),
  );
  return `${header.join(",")}\n${rows.join("\n")}\n`;
}

/** Self-contained HTML review report (sortable table + per-rule summary). */
export function quarantineHtml(records, summary, options = {}) {
  const stats = ruleStats(records);
  const total = summary?.inputLines ?? 0;
  const kept = summary?.keptLines ?? 0;
  const removed = summary?.removedLines ?? 0;

  const ruleCards = [...stats.entries()]
    .map(
      ([rule, stat]) => `
      <div class="card">
        <div class="card-title">${escapeHtml(RULE_LABEL_ZH[rule] ?? rule)}</div>
        <div class="card-body">
          <div>边界样本 <b>${stat.count}</b> 条</div>
          <div>metric 范围 <b>${stat.min === Infinity ? "—" : stat.min}</b> ~ <b>${stat.max === -Infinity ? "—" : stat.max}</b></div>
          ${
            stat.suggested !== null
              ? `<div>建议阈值（参考）<b>${Math.round(stat.suggested * 10000) / 10000}</b></div>`
              : ""
          }
        </div>
      </div>`,
    )
    .join("\n");

  const rows = records
    .map(
      (entry, index) => `
      <tr>
        <td data-sort="${entry.line}">${entry.line}</td>
        <td>${escapeHtml(entry.rule)}</td>
        <td>${escapeHtml(entry.source ?? "")}</td>
        <td class="num">${entry.metric}</td>
        <td class="num">${entry.threshold}</td>
        <td class="num">${entry.deviation}</td>
        <td class="decision">${escapeHtml(entry.decision ?? "")}</td>
        <td class="preview" title="${escapeHtml(previewJson(entry.record, 1000))}">${escapeHtml(previewJson(entry.record))}</td>
      </tr>`,
    )
    .join("\n");

  const ruleOptions = [...new Set(records.map((entry) => entry.rule))];
  const ruleFilter = `<option value="">全部规则</option>${ruleOptions
    .map((rule) => `<option value="${escapeHtml(rule)}">${escapeHtml(rule)}</option>`)
    .join("\n")}`;

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据清洗异常样本审查报告</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; line-height: 1.5; color: #1f2328; background: #fff; }
  @media (prefers-color-scheme: dark) { body { background: #0d1117; color: #e6edf3; } }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .meta { color: #57606a; font-size: 13px; margin-bottom: 16px; }
  .summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .summary .pill { border: 1px solid #d0d7de; border-radius: 999px; padding: 4px 14px; font-size: 13px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 14px; font-size: 13px; }
  .card-title { font-weight: 600; margin-bottom: 6px; }
  .card-body div { color: #57606a; }
  .card-body b { color: #1f2328; }
  @media (prefers-color-scheme: dark) { .card-body div { color: #8b949e; } .card-body b { color: #e6edf3; } }
  .instructions { border: 1px solid #d4a72c; background: #fff8e1; border-radius: 8px; padding: 12px 16px; font-size: 13px; margin-bottom: 20px; }
  @media (prefers-color-scheme: dark) { .instructions { background: #2a2313; border-color: #d4a72c; } }
  .toolbar { display: flex; gap: 12px; margin-bottom: 12px; }
  .toolbar select { padding: 4px 8px; border-radius: 6px; border: 1px solid #d0d7de; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; vertical-align: top; }
  th { background: #f6f8fa; cursor: pointer; user-select: none; white-space: nowrap; }
  @media (prefers-color-scheme: dark) { th { background: #161b22; } }
  th:hover { text-decoration: underline; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  td.preview { max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, monospace; font-size: 12px; color: #57606a; }
  .empty { padding: 24px; text-align: center; color: #57606a; }
  .badge { display: inline-block; border-radius: 6px; padding: 0 8px; font-size: 12px; background: #eaeef2; }
</style>
</head>
<body>
<h1>数据清洗异常样本审查报告</h1>
<div class="meta">quarantine 边界样本（Reject / Accept 后再调阈值重跑） · 生成时间 ${new Date().toISOString()}${options.path ? ` · 源文件 ${escapeHtml(options.path)}` : ""}</div>
<div class="summary">
  <span class="pill">输入 ${total} 条</span>
  <span class="pill">保留 ${kept} 条</span>
  <span class="pill">删除 ${removed} 条</span>
  <span class="pill">边界样本 ${records.length} 条</span>
</div>
<div class="instructions">
  <b>工作流：</b>1) 逐行审查下方边界样本，判断其是否该被删（例如 repeat_ratio 阈值 0.5 附近的记录是正常文本还是重复垃圾）；2) 在 decision 列标注 reject / accept；3) 依据「建议阈值」和边界样本的实际分布调整对应阈值；4) 用新阈值重新执行 finetune_dataset_clean。<br>
  数据同时保存在 <code>&lt;path&gt;.quarantine.jsonl</code>（机器可读，含完整原始记录）和 <code>&lt;path&gt;.quarantine.csv</code>。PPL 相关建议值见 <code>&lt;path&gt;.clean.python-report.json</code>。
</div>
${
  records.length === 0
    ? `<div class="empty">没有边界样本：本次运行的被删记录都离阈值较远，阈值无需调整。</div>`
    : `<div class="cards">${ruleCards}</div>
<div class="toolbar">
  <select id="rule-filter" onchange="filterRows()">${ruleFilter}</select>
  <span class="badge">点击表头排序</span>
</div>
<table id="report">
  <thead>
    <tr>
      <th data-key="line">行号</th>
      <th data-key="rule">规则</th>
      <th data-key="source">source</th>
      <th data-key="metric">metric</th>
      <th data-key="threshold">阈值</th>
      <th data-key="deviation">偏离</th>
      <th data-key="decision">decision</th>
      <th>记录预览</th>
    </tr>
  </thead>
  <tbody>${rows}</tbody>
</table>
<script>
  let sortKey = null, sortAsc = true;
  function filterRows() {
    const rule = document.getElementById("rule-filter").value;
    document.querySelectorAll("#report tbody tr").forEach((tr) => {
      tr.style.display = !rule || tr.children[1].textContent === rule ? "" : "none";
    });
  }
  document.querySelectorAll("#report th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (!key) return;
      if (sortKey === key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
      const tbody = document.querySelector("#report tbody");
      const rows = [...tbody.querySelectorAll("tr")];
      rows.sort((a, b) => {
        const va = a.dataset.sort !== undefined ? Number(a.dataset.sort) : a.children[th.cellIndex].textContent;
        const vb = b.dataset.sort !== undefined ? Number(b.dataset.sort) : b.children[th.cellIndex].textContent;
        const na = Number(va), nb = Number(vb);
        const cmp = Number.isFinite(na) && Number.isFinite(nb) ? na - nb : String(va).localeCompare(String(vb), "zh");
        return sortAsc ? cmp : -cmp;
      });
      rows.forEach((tr) => tbody.appendChild(tr));
    });
  });
</script>`
}
</body>
</html>
`;
}
