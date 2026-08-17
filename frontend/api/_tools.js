// Tool definitions + execution for the chat agent. Reads the same
// precomputed data the dashboard itself renders from - no separate
// data pipeline, no raw seismic volumes (those are too large to hand
// to the model; tools only return scalar/summary values).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { wellsData, wellsConfig, blindWellName } from "../src/data.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const WELL_NAMES = Object.keys(wellsConfig);
const PROPERTIES = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT", "AI", "VPVS", "POIS", "LMRHO", "MURHO"];

let _perfCache = null;
function loadModelPerformance() {
  if (_perfCache) return _perfCache;
  const csvPath = path.join(__dirname, "..", "..", "ml_outputs_v11", "model_performance.csv");
  const raw = fs.readFileSync(csvPath, "utf-8");
  _perfCache = parseCsv(raw);
  return _perfCache;
}

function parseCsv(raw) {
  const lines = raw.trim().split("\n");
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const row = {};
    headers.forEach((h, i) => { row[h] = cells[i]; });
    return row;
  });
}

function splitCsvLine(line) {
  const cells = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') { inQuotes = !inQuotes; continue; }
    if (c === "," && !inQuotes) { cells.push(cur); cur = ""; continue; }
    cur += c;
  }
  cells.push(cur);
  return cells;
}

export const toolDefinitions = [
  {
    name: "list_wells",
    description:
      "List all wells in the field study with their coordinates, inline/crossline position, kelly bushing elevation, well-seismic tie quality (correlation), and whether each is the held-out blind test well. Use for questions like 'what wells are there' or 'which well is the blind test well'.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "get_well_log",
    description:
      "Get a well's log curve (actual, where available, and V11-predicted) for one property across depth/time. Returns an array of {time, actual, predicted} samples. Use for questions like 'what is the VSH at Z-04' or 'show me the GR log for Z-07'.",
    input_schema: {
      type: "object",
      properties: {
        well: { type: "string", enum: WELL_NAMES, description: "Well name" },
        property: { type: "string", enum: PROPERTIES, description: "Petrophysical/elastic property code" },
      },
      required: ["well", "property"],
    },
  },
  {
    name: "get_model_performance",
    description:
      "Get the V11 ML model's cross-validation R2/MAE and blind-well R2/MAE for one or all prediction targets. Use for questions like 'how accurate is the VSH prediction' or 'which properties are reliable on the blind well'.",
    input_schema: {
      type: "object",
      properties: {
        target: { type: "string", enum: PROPERTIES, description: "Optional: restrict to one target. Omit to get all targets." },
      },
    },
  },
  {
    name: "compare_wells",
    description:
      "Compare a property's V11-predicted values between two wells (mean/min/max over the logged interval). Use for questions like 'compare VSH between Z-04 and Z-07'.",
    input_schema: {
      type: "object",
      properties: {
        wellA: { type: "string", enum: WELL_NAMES },
        wellB: { type: "string", enum: WELL_NAMES },
        property: { type: "string", enum: PROPERTIES },
      },
      required: ["wellA", "wellB", "property"],
    },
  },
];

export async function runTool(name, input) {
  switch (name) {
    case "list_wells":
      return listWells();
    case "get_well_log":
      return getWellLog(input.well, input.property);
    case "get_model_performance":
      return getModelPerformance(input.target);
    case "compare_wells":
      return compareWells(input.wellA, input.wellB, input.property);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function listWells() {
  return WELL_NAMES.map((name) => {
    const w = wellsData[name] || {};
    return {
      name,
      is_blind_test_well: name === blindWellName,
      x: w.x, y: w.y, kb: w.kb,
      inline: w.inline, crossline: w.crossline,
      tie_correlation: w.tie?.correlation ?? null,
      tie_polarity: w.tie?.polarity ?? null,
    };
  });
}

function getWellLog(well, property) {
  if (!wellsData[well]) throw new Error(`Unknown well: ${well}`);
  const samples = wellsData[well].samples || [];
  const points = samples
    .map((s) => ({
      time: s.time,
      actual: s[`${property} (Act)`] ?? null,
      predicted: s[`${property} (Pred)`] ?? null,
    }))
    .filter((p) => p.actual !== null || p.predicted !== null);
  return { well, property, sample_count: points.length, points };
}

function getModelPerformance(target) {
  const rows = loadModelPerformance();
  const filtered = target ? rows.filter((r) => r.target === target) : rows;
  return filtered.map((r) => ({
    target: r.target,
    is_sand_only_variant: r.is_sand_only,
    best_model: r.best_model,
    cv_r2: Number(r.best_cv_r2),
    blind_r2: Number(r.blind_r2),
  }));
}

function summarizeValues(vals) {
  if (vals.length === 0) return null;
  const sum = vals.reduce((a, b) => a + b, 0);
  return { count: vals.length, mean: sum / vals.length, min: Math.min(...vals), max: Math.max(...vals) };
}

function compareWells(wellA, wellB, property) {
  const a = getWellLog(wellA, property).points.map((p) => p.predicted).filter((v) => v != null);
  const b = getWellLog(wellB, property).points.map((p) => p.predicted).filter((v) => v != null);
  return { property, [wellA]: summarizeValues(a), [wellB]: summarizeValues(b) };
}
