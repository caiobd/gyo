const KEY = "gyo-vi-prefs";

export const loadPrefs = () => {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
  catch { return {}; }
};

export const savePrefs = (p) => localStorage.setItem(KEY, JSON.stringify({ ...loadPrefs(), ...p }));
