/**
 * Shared JSONL dataset validation used by the standalone validate tool and by
 * pre-flight checks inside the remote create / local train tools.
 *
 * Supported per-line record formats (all records in one file must share one):
 * - chat:              { "messages": [{ "role", "content" }, ...] }
 * - prompt-completion: { "prompt": str, "completion": str }
 * - alpaca:            { "instruction": str, "input"?: str, "output": str }
 * - sharegpt:          { "conversations": [{ "from", "value" }, ...] }
 */

const CHAT_ROLES = new Set(["system", "user", "assistant", "tool"]);
const ASSISTANT_FROM = new Set(["gpt", "assistant"]);
const SHAREGPT_FROM = new Set(["system", "human", "gpt", "user", "assistant"]);

function isNonEmptyString(value) {
  return typeof value === "string" && value.length > 0;
}

function isChatRecord(record) {
  if (typeof record !== "object" || record === null || !Array.isArray(record.messages)) return false;
  if (record.messages.length === 0) return false;
  return record.messages.every(
    (message) =>
      typeof message === "object" &&
      message !== null &&
      CHAT_ROLES.has(message.role) &&
      isNonEmptyString(message.content),
  );
}

function isPromptCompletionRecord(record) {
  return (
    typeof record === "object" &&
    record !== null &&
    isNonEmptyString(record.prompt) &&
    isNonEmptyString(record.completion)
  );
}

function isAlpacaRecord(record) {
  return (
    typeof record === "object" &&
    record !== null &&
    isNonEmptyString(record.instruction) &&
    isNonEmptyString(record.output) &&
    (record.input === undefined || record.input === null || typeof record.input === "string")
  );
}

function isSharegptRecord(record) {
  if (typeof record !== "object" || record === null || !Array.isArray(record.conversations)) return false;
  const turns = record.conversations;
  if (turns.length === 0) return false;
  // Every turn must be { from, value } with a non-empty string value...
  for (const turn of turns) {
    if (
      typeof turn !== "object" ||
      turn === null ||
      !SHAREGPT_FROM.has(turn.from) ||
      !isNonEmptyString(turn.value)
    ) {
      return false;
    }
  }
  // ...and the file must end on an assistant answer so SFT has a target.
  const last = turns[turns.length - 1];
  return ASSISTANT_FROM.has(last.from);
}

const FORMAT_LABELS = {
  chat: "chat",
  "prompt-completion": "prompt-completion",
  alpaca: "alpaca",
  sharegpt: "sharegpt",
};

export function detectFormat(record) {
  if (isChatRecord(record)) return "chat";
  if (isPromptCompletionRecord(record)) return "prompt-completion";
  if (isAlpacaRecord(record)) return "alpaca";
  if (isSharegptRecord(record)) return "sharegpt";
  return null;
}

const FORMAT_EXAMPLE_HINT =
  'record must be {"messages":[{"role","content"},...]}, {"prompt","completion"}, ' +
  '{"instruction","output"} (alpaca), or {"conversations":[{"from","value"},...]} (sharegpt)';

/** Shared per-line validator core: handles one line at a time, O(1) memory. */
function makeValidator() {
  const errors = [];
  let format = null;
  let chars = 0;
  let preview = "";
  let rows = 0;

  const handleLine = (rawLine) => {
    const trimmed = rawLine.trim();
    if (trimmed === "") return;
    rows += 1;
    const lineNo = rows;

    let record;
    try {
      record = JSON.parse(trimmed);
    } catch (error) {
      errors.push({ line: lineNo, message: `invalid JSON: ${error.message}` });
      return;
    }
    chars += rawLine.length;

    const detected = detectFormat(record);
    if (detected === null) {
      errors.push({ line: lineNo, message: FORMAT_EXAMPLE_HINT });
      return;
    }
    if (format === null) format = detected;
    else if (format !== detected) {
      errors.push({ line: lineNo, message: `record is ${FORMAT_LABELS[detected]} but the file also contains ${FORMAT_LABELS[format]} records` });
      return;
    }

    if (preview === "") {
      preview = JSON.stringify(record).slice(0, 400);
    }
  };

  const finish = () => {
    if (rows === 0) {
      return {
        valid: false,
        lines: 0,
        format: "empty",
        errors: [],
        approxTokens: 0,
        preview: "",
      };
    }
    return {
      valid: errors.length === 0,
      lines: rows,
      format: format === null ? "invalid" : format,
      errors: errors.slice(0, 10),
      approxTokens: Math.round(chars / 4),
      preview,
    };
  };

  return { handleLine, finish };
}

/**
 * Validate JSONL fine-tuning content: every non-empty line must parse as JSON
 * in one of the supported formats, and all records must share that format.
 *
 * @param {string} text raw file content
 * @returns {{
 *   valid: boolean,
 *   lines: number,
 *   format: "chat" | "prompt-completion" | "alpaca" | "sharegpt" | "invalid" | "empty",
 *   errors: { line: number, message: string }[],
 *   approxTokens: number,
 *   preview: string,
 * }}
 */
export function validateJsonlDataset(text) {
  const validator = makeValidator();
  for (const line of text.split(/\r?\n/)) validator.handleLine(line);
  return validator.finish();
}

/**
 * Streaming variant of validateJsonlDataset for files that do not fit in
 * memory: same rules, one line at a time.
 *
 * @param {AsyncIterable<string>} lines raw lines from the file (e.g. a
 *        node:readline interface); empty lines are skipped
 * @returns {Promise<{
 *   valid: boolean,
 *   lines: number,
 *   format: "chat" | "prompt-completion" | "alpaca" | "sharegpt" | "invalid" | "empty",
 *   errors: { line: number, message: string }[],
 *   approxTokens: number,
 *   preview: string,
 * }>}
 */
export async function validateJsonlLines(lines) {
  const validator = makeValidator();
  for await (const line of lines) validator.handleLine(line);
  return validator.finish();
}
