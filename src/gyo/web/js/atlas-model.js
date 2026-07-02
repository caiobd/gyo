const SAMPLE_MODES = new Set(["representative", "outliers", "parent"]);

function copyPrefix(prefix) {
  if (!Array.isArray(prefix) || prefix.some(value => !Number.isInteger(value) || value < 0)) {
    throw new TypeError("prefix must be an array of non-negative integers");
  }
  return prefix.slice();
}

export function prefixKey(prefix) {
  const copy = copyPrefix(prefix);
  return copy.length === 0 ? "root" : copy.join(",");
}

export function createState(payload) {
  if (!payload || typeof payload !== "object") throw new TypeError("payload must be an object");
  return {
    payload,
    focus: copyPrefix(payload.focus ?? []),
    selected: null,
    sampleMode: "representative",
  };
}

export function selectNode(state, prefix) {
  return { ...state, selected: copyPrefix(prefix), sampleMode: "representative" };
}

export function enterNode(state) {
  if (state.selected === null) return state;
  return { ...state, focus: copyPrefix(state.selected), selected: null, sampleMode: "representative" };
}

export function setSampleMode(state, mode) {
  if (!SAMPLE_MODES.has(mode)) throw new RangeError(`unsupported sample mode: ${mode}`);
  return { ...state, sampleMode: mode };
}

export function parentPrefix(prefix) {
  const copy = copyPrefix(prefix);
  copy.pop();
  return copy;
}
