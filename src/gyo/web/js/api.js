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
