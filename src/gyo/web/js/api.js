const cache = new Map();

export async function fetchTree() {
  return (await fetch("/api/tree?level=99")).json();
}

export async function fetchNodeItems(prefixKey) {
  if (cache.has(prefixKey)) return cache.get(prefixKey);
  const p = fetch(`/api/node/${prefixKey}`).then(r => r.json());
  cache.set(prefixKey, p);
  return p;
}

export const thumbUrl = (idx) => `/thumb/${idx}`;

export async function fetchAtlas(prefix = "root", options) {
  const response = await fetch(`/api/atlas/${encodeURIComponent(prefix)}`, options);
  if (response.ok) return response.json();
  let detail;
  try { detail = (await response.json())?.detail; } catch { /* non-JSON response */ }
  throw new Error(`${detail || response.statusText || "Atlas request failed"} (HTTP ${response.status})`);
}
