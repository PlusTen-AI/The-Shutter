// Per-model job defaults (design §11 모델 선택, §2 UI 요구). Field values here are the engine defaults; the boss's own
// defaults are saved in the Mac settings (settings.defaults[model]) from the 설정 › 기본값 page and win over these.
import { S, t, html } from './app.js';

export const MODELS = {
  qi21: { name: 'Qwen-Image 2.1', kind: 'image', fields: [
    { k: 'steps', type: 'num', def: 25 },
    { k: 'cfg', type: 'num', def: 1, note: 'cfgNote' },   // > 1 uses the negative prompt, two DiT calls a step (design §11)
    { k: 'neg', type: 'text', def: '' },
    { k: 'mp', type: 'sel', opts: [4, 1], def: 4 },
    { k: 'cands', type: 'seg', opts: [1, 2, 4], def: 2 },
    { k: 'seedFixed', type: 'bool', def: true },
    { k: 'autoRetry', type: 'bool', def: false },
  ] },
  ltx25: { name: 'LTX-2.5', kind: 'video', fields: [
    { k: 'sec', type: 'sel', opts: [5, 7.3, 10, 15, 20], def: 5 },
    { k: 'cands', type: 'seg', opts: [1, 2, 4], def: 2 },
    { k: 'face', type: 'bool', def: true },
    { k: 'upscale', type: 'bool', def: false },
  ] },
  pih3: { name: 'pi-H3', kind: 'video', fields: [
    { k: 'steps', type: 'seg', opts: [2, 4], def: 2 },
    { k: 'vsa', type: 'bool', def: false, note: 'vsaNote' },
    { k: 'sec', type: 'sel', opts: [5, 7.3, 10, 15], def: 5 },
    { k: 'cands', type: 'seg', opts: [1, 2, 4], def: 2 },
    { k: 'face', type: 'bool', def: true },
    { k: 'upscale', type: 'bool', def: false },
  ] },
  h3: { name: 'H3', kind: 'video', fields: [
    { k: 'ref_size', type: 'sel', opts: ['max', 'match'], def: 'max' },
    { k: 'sec', type: 'sel', opts: [5, 7.3, 10, 15], def: 5 },
    { k: 'cands', type: 'seg', opts: [1, 2, 4], def: 2 },
    { k: 'refine', type: 'bool', def: true },
    { k: 'upscale', type: 'bool', def: false },
  ] },
};

// engine defaults overlaid with the saved ones
export function defaults(model) {
  const m = MODELS[model];
  const saved = (S.status && S.status.settings && S.status.settings.defaults && S.status.settings.defaults[model]) || {};
  const out = {};
  for (const f of m.fields) out[f.k] = saved[f.k] !== undefined ? saved[f.k] : f.def;
  return out;
}

// Qwen 고급 (design §11): cfg, and the negative prompt only while cfg > 1 (two DiT calls a step, about 2× the image
// time). The fields carry data-cfg; a screen's input handler hands its events to cfgInput first.
const cfgOn = (st) => Number(st.cfg) > 1;
export const cfgX = (st) => cfgOn(st) ? 2 : 1;
export const cfgOpts = (st) => cfgOn(st) ? { cfg: Number(st.cfg), neg: (st.neg || '').trim() } : {};
export function cfgFields(st) {
  return html`<label class="field">${t('def.cfg')}<input class="in mono" data-cfg="cfg" inputmode="decimal" value="${st.cfg}" title="${t('def.cfgNote')}" style="width: 120px"></label>
    <label class="field" data-negf style="grid-column: 1 / -1" ${cfgOn(st) ? '' : 'hidden'}>${t('def.neg')}<textarea class="in" rows="2" data-cfg="neg" placeholder="${t('def.negNote')}">${st.neg || ''}</textarea></label>`;
}
export function cfgInput(e, st) {
  const k = e.target.dataset.cfg;
  if (!k) return false;
  st[k] = e.target.value;
  const f = k === 'cfg' && e.target.closest('label').parentElement.querySelector('[data-negf]');
  if (f) f.hidden = !cfgOn(st);
  return true;
}
