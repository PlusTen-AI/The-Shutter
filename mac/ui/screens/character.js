// 이미지 › 워크플로 › 캐릭터: card (or photo) → 원판 candidates → ● base → sheet (8 panels) + expressions → 등록.
// An outfit (wardrobe, boss session #13) is the same flow for a registered character: new clothes (text and / or an
// outfit picture) on its own master (char.derive kind outfit) → ● → sheet → 등록 as <code>-<style> (FL01-suit, FL01-2).
// Two panes (design §11): settings left (all chosen before 시작, never folds), results right (stages shown as 대기 from
// the start, filled in place). The session lives on the Mac in characters/<code>/session.json until 등록 / 닫기.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, enter, isStopped, unitStopped, failWord, failColor, tOr } from '../app.js';
import { defaults, cfgFields, cfgInput, cfgOpts, cfgX } from '../models.js';
import { sideMenu, IMAGE_MENU } from './wip.js';

const EXPR_IDS = ['calm', 'curious', 'tense', 'surprise', 'fear', 'sad', 'resolve', 'relax', 'talk'];
const SEC = { 1: 19, 4: 111, panel: 20 };   // seconds per image on one node (production, session #10)
const PANELS = ['sh_head_front', 'sh_head_34L', 'sh_head_34R', 'sh_head_side', 'sh_full_front', 'sh_full_34', 'sh_full_side', 'sh_full_back'];
const CARD_KEYS = ['name', 'age', 'extra', 'face', 'hair', 'body', 'clothing'];
const NOFACE = ['sh_head_side', 'sh_full_side', 'sh_full_back'];

let root = null, st = null, codePreview = '', open = [], meta = {}, detail = {}, autoBusy = false;
let baseInfo = null, outfits = [];   // an outfit's character (card, master) · a character's outfits (wardrobe)
let ui = { adv: false, en: false, enText: '', redo: false, redoText: '', askClose: false, askStop: false, upload: false, voiceUp: false };

function fresh() {
  const d = defaults('qi21');
  return { code: null, gender: 'F', rank: 'L', start: 'card', photo: null, card: {}, mode: 'all',
    cands: d.cands, seed: 1, seedFixed: d.seedFixed, steps: d.steps, mp: d.mp, autoRetry: d.autoRetry, cfg: d.cfg, neg: d.neg,
    expr: { on: true, ids: ['calm', 'surprise', 'sad', 'talk'], strength: 1, custom: [] },   // custom: [{id, text}] (기능 12)
    masterJobs: [], base: null, saved: [], sheetJob: null, sheetBase: null, sheetExpr: null, oldSheetJobs: [],
    cardAt: null, prompt: null, registered: null, of: null, outfitStyle: '', outfitImage: null };
}
const outfitCode = () => st.code || (st.outfitStyle ? st.of + '-' + st.outfitStyle : codePreview);

// ---- derived state ----
const job = (id) => S.jobs.find((j) => j.id === id) || detail[id] || null;
const active = (j) => j && (j.state === 'pending' || j.state === 'running');
const sessionJobs = () => [...st.masterJobs, st.sheetJob].filter(Boolean);
const running = () => sessionJobs().some((id) => active(job(id)));
const started = () => !!st.code && (st.masterJobs.length > 0 || !!st.registered);
const cardKey = () => JSON.stringify([st.start, st.photo, CARD_KEYS.map((k) => (st.card[k] || '').trim())]
  .concat(st.of ? [st.outfitImage && st.outfitImage.rel] : []));
const dirty = () => started() && st.cardAt !== null && st.cardAt !== cardKey();
const customs = () => (st.expr.custom || []).filter((x) => /^[a-z0-9]+(_[a-z0-9]+)*$/.test(x.id || '') && (x.text || '').trim() && !EXPR_IDS.includes(x.id));
const exprKey = () => JSON.stringify(st.expr.on ? [st.expr.ids.slice().sort(), st.expr.strength, customs().map((x) => [x.id, x.text.trim()])] : []);
const sameRef = (a, b) => !!a && !!b && a.job === b.job && a.name === b.name && a.registered === b.registered;

function candidates() {
  const out = [];
  if (st.registered) out.push({ registered: st.registered, url: filesUrl(`characters/${st.code}/${st.registered}`), seed: '', check: null });
  for (const id of st.masterJobs) {
    const j = job(id);
    if (!j) continue;
    const m = meta[id] || {};
    for (const f of j.files.filter((x) => x.name.endsWith('.png') && x.synced)) {
      const mm = m[f.name] || {};
      out.push({ job: id, name: f.name, url: jobFileUrl(j, f.name), seed: mm.seed, check: checkOf(mm) });
    }
    const left = j.count.pending + j.count.running;
    for (let k = 0; k < left; k++) out.push({ pending: true, job: id });
    for (let k = 0; k < j.count.failed; k++) out.push({ failed: true, stopped: isStopped(j), job: id });
  }
  return out;
}

// check dot: face (photo start / panels) or body (card master: ankles must not show; full panels: leg ratio)
function checkOf(mm) {
  if (!mm || mm.seed === undefined) return null;
  if (mm.passed === true) return { cls: 'ok', label: t('ch.pass'), tip: t('ch.faceTip', { sim: last(mm.face_sim) }) };
  if (mm.passed === false) return { cls: 'bad', label: t('ch.below'), tip: t('ch.faceTip', { sim: last(mm.face_sim) }) };
  const b = mm.body && mm.body[mm.body.length - 1];
  if (b) {
    const tip = mm.body_check === 'framing' ? t('ch.framingTip', { ok: b.ok ? t('ch.ok') : t('ch.ankles') }) : t('ch.legsTip', { v: b.legs == null ? '—' : b.legs.toFixed(2) });
    return b.ok ? { cls: 'ok', label: t('ch.pass'), tip } : { cls: 'warn', label: t('ch.check'), tip };
  }
  return null;
}
const last = (a) => Array.isArray(a) && a.length && a[a.length - 1] != null ? a[a.length - 1].toFixed(2) : '—';

function sheetUnits() {
  const d = st.sheetJob && detail[st.sheetJob];
  return d && d.units ? d.units : [];
}
function stageOf(kind) {   // 'm' | 's' | 'e' → { state: wait|run|done|failed|pick, pct }
  if (kind === 'm') {
    const js = st.masterJobs.map(job).filter(Boolean);
    if (js.some(active)) return { state: 'run', pct: Math.round(100 * js.filter(active).reduce((a, j) => a + j.progress, 0) / js.filter(active).length) };
    if (candidates().some((c) => c.url)) return { state: st.base ? 'done' : 'pick' };
    if (js.some((j) => j.state === 'failed')) return { state: 'failed', stopped: js.filter((j) => j.state === 'failed').every(isStopped) };
    return { state: 'wait' };
  }
  if (!st.sheetJob && (kind === 's' ? st.regSheet : (st.regExpr || []).length)) return { state: 'reg' };   // already registered
  const us = sheetUnits().filter((u) => kind === 's' ? !String(u.spec.kind || '').startsWith('ex_') : String(u.spec.kind || '').startsWith('ex_'));
  if (!st.sheetJob || !us.length) return { state: st.sheetJob && active(job(st.sheetJob)) ? 'run' : 'wait', pct: 0 };
  if (us.some((u) => u.state === 'running' || u.state === 'pending')) {
    return { state: 'run', pct: Math.round(100 * us.reduce((a, u) => a + (u.state === 'done' ? 1 : u.progress || 0), 0) / us.length) };
  }
  if (us.some((u) => u.state === 'failed')) return { state: 'failed', stopped: us.filter((u) => u.state === 'failed').every(unitStopped) };
  return { state: 'done' };
}

// ---- persistence ----
async function save() {
  if (st.code) await act(() => api('/api/session', { code: st.code, session: st }));
}

// ---- actions ----
function masterOptions(more) {
  const o = { count: st.cands, steps: st.steps, mp: st.mp, retries: st.autoRetry ? 3 : 0, body_retries: st.autoRetry ? 2 : 0, ...cfgOpts(st) };
  if (st.seedFixed || more) o.seed = st.seed + candidates().filter((c) => !c.registered).length;
  if (st.prompt) o.prompt = st.prompt;
  return o;
}
async function saveCard() {
  const c = Object.fromEntries(CARD_KEYS.map((k) => [k, (st.card[k] || '').trim()]).filter(([, v]) => v));
  return api('/api/codes', { code: st.code, card: c });
}
async function submitMaster(prompt) {
  const o = masterOptions(!!prompt);
  if (prompt) o.prompt = prompt;
  const r = st.of ?
    await api('/api/codejob', { type: 'char.derive', code: st.code, file_kind: 'ref', outfit_image: st.outfitImage ? st.outfitImage.rel : undefined,
      options: Object.assign(o, { kind: 'outfit', clothing: (st.card.clothing || '').trim(), aspect: '2:3', body_check: 'framing' }) }) :
    await api('/api/codejob', { type: st.start === 'photo' ? 'char.from_photo' : 'char.ref', code: st.code, options: o,
      inputs: st.start === 'photo' ? { image_1: st.photo } : {} });
  st.masterJobs.push(r.id);
}
async function submitSheet() {
  const o = { seed: st.seed, auto_retry: st.autoRetry, expr: st.expr.on ? st.expr.ids : [], strength: st.expr.strength,
    expr_custom: st.expr.on ? customs().map((x) => ({ id: x.id, text: x.text.trim() })) : [] };
  const r = await api('/api/codejob', { type: 'char.sheet', code: st.code, options: o, master: st.base && st.base.job ? st.base : undefined });
  if (st.sheetJob) st.oldSheetJobs.push(st.sheetJob);
  st.sheetJob = r.id;
  st.sheetBase = st.base;
  st.sheetExpr = exprKey();
}

async function start() {
  if (st.of && !(st.card.clothing || '').trim() && !st.outfitImage && !st.prompt) return toast(t('ch.needOutfit'), true);
  if (!st.of && st.start === 'photo' && !st.photo) return toast(t('ch.needPhoto'), true);
  if (!st.of && st.start === 'card' && !(st.card.face || st.card.hair || st.card.body || st.prompt)) return toast(t('ch.needCard'), true);
  const ok = await act(async () => {
    if (!st.code) st.code = (await api(st.of ? `/api/nextcode?of=${st.of}&style=${encodeURIComponent(st.outfitStyle)}` :
      `/api/nextcode?g=${st.gender}&r=${st.rank}`)).code;
    await saveCard();
    st.cardAt = cardKey();
    st.base = null;
    await submitMaster();
    await save();
    return true;
  });
  if (ok) { await poll(); renderAll(); }
}
async function rerunCard() {   // card changed after start: new candidates from the new card; base and sheet go stale
  await act(async () => {
    await saveCard();
    st.cardAt = cardKey();
    st.base = null;
    await submitMaster();
    await save();
  });
  await poll();
  renderAll();
}

// all mode: once the candidates are in, the first passing one becomes ● and the sheet starts by itself
async function auto() {
  if (autoBusy || !started() || st.mode !== 'all' || running()) return;
  if (!st.base && !st.sheetJob) {
    const cs = candidates().filter((c) => c.url && !c.registered);
    if (!cs.length) return;
    autoBusy = true;
    st.base = (cs.find((c) => c.check && c.check.cls === 'ok') || cs[0]);
    st.base = { job: st.base.job, name: st.base.name };
    await act(submitSheet);
    await save();
    autoBusy = false;
    await poll();
  }
}

// ---- rendering: left (settings) ----
function seg(name, opts, cur, label, disabled, cls) {
  return html`<div class="seg ${cls || ''}" role="group" aria-label="${label}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${name}" data-v="${v}" aria-pressed="${String(cur) === String(v)}" ${disabled ? 'disabled' : ''}>${l}</button>`)}</div>`;
}
function field(k, label, ph, full) {
  return html`<label class="field" style="${full ? 'grid-column: 1 / -1' : ''}">${label}
    <input class="in" type="text" data-card="${k}" value="${st.card[k] || ''}" placeholder="${ph}"></label>`;
}
function renderSet() {
  const el = root.querySelector('[data-set]');
  const locked = !!st.code;   // gender and rank make the code: fixed after 시작
  const resumable = open.filter((c) => c !== st.code);
  if (st.of) return renderOutfitSet(el, resumable);
  el.innerHTML = html`
    <div>
      <div class="lbl">${t('ch.crumb')}</div>
      <h1 class="h1">${st.card.name || (st.code ? '' : t('ch.new'))} <span class="mono" style="font-weight: ${st.card.name || !st.code ? 400 : 600}; color: ${st.card.name || !st.code ? 'var(--fg3)' : 'var(--fg)'}">${st.code || codePreview}${st.code ? '' : ' · ' + t('ch.auto')}</span></h1><p class="lead">${t('lead.character')}</p>
    </div>
    ${!started() && resumable.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${resumable.map((c) => html`<a class="chip" href="#/image/character?code=${c}" style="display: inline-flex; align-items: center; text-decoration: none">${c}</a>`)}</div>` : ''}
    <div class="col" style="gap: 6px">
      ${seg('mode', [['all', t('ch.modeAll')], ['step', t('ch.modeStep')]], st.mode, t('ch.mode'))}
      <span class="sub">${st.mode === 'all' ? t('ch.modeAllHint') : t('ch.modeStepHint')}</span></div>

    ${st.registered ? wardrobe() : ''}
    ${st.registered ? voiceBlock() : ''}
    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('ch.card')}</h2>
        ${seg('start', [['card', t('ch.fromCard')], ['photo', t('ch.fromPhoto')]], st.start, t('ch.startHow'), !!st.registered)}
        <div class="grow"></div>
        <button type="button" class="btn ghost" data-act="en" aria-expanded="${ui.en}" style="min-height: 30px; padding: 0 8px; font-size: 12px; color: var(--blue)">${t('ch.english')}</button></div>
      ${ui.en ? html`<div class="col" style="gap: 6px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
        <textarea class="in mono" rows="8" data-en style="font-size: 12px">${st.prompt || ui.enText}</textarea>
        <div class="row" style="justify-content: space-between"><span class="sub">${st.prompt ? t('ch.enEdited') : t('ch.enFromCard')}</span>
          ${st.prompt ? html`<button type="button" class="ib m" data-act="enReset" aria-label="${t('ch.enReset')}" title="${t('ch.enReset')}">${icon('redo')}</button>` : ''}</div></div>` : ''}
      ${st.start === 'photo' ? html`<div class="row" style="gap: 10px">
        <label class="ph" style="width: 96px; height: 128px; display: flex; align-items: center; justify-content: center; cursor: pointer" title="${t('ch.photo')}">
          ${st.photo ? html`<span class="sub" style="text-align: center">${t('ch.photoIn')}<br>${st.photo.slice(0, 8)}</span>` : html`<span class="sub">${ui.upload ? t('common.loading') : t('ch.photo')}</span>`}
          <input type="file" accept="image/*" data-photo hidden></label>
        <span class="sub grow">${t('ch.photoNote')}</span></div>` : ''}
      <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px">
        ${field('name', t('card.name'), t('ch.ph.name'))}
        <div class="field">${t('ch.code')}<span class="mono" style="min-height: 40px; display: flex; align-items: center; color: var(--fg)">${st.code || codePreview + ' · ' + t('ch.auto')}</span></div>
        <div class="row" style="grid-column: 1 / -1; gap: 12px; flex-wrap: nowrap; align-items: flex-start">
          <div class="field">${t('ch.gender')}${seg('gender', [['F', t('ch.female')], ['M', t('ch.male')]], st.gender, t('ch.gender'), locked)}</div>
          <div class="field grow">${t('ch.rank')}${seg('rank', [['L', t('ch.lead')], ['S', t('ch.support')], ['E', t('ch.extraRank')]], st.rank, t('ch.rank'), locked, 'fill')}</div></div>
        ${field('age', t('card.age'), '24')}
        ${field('extra', t('ch.extra'), t('ch.ph.extra'))}
        ${st.start === 'card' ? [field('face', t('card.face'), t('ch.ph.face'), 1), field('hair', t('card.hair'), t('ch.ph.hair'), 1), field('body', t('card.body'), t('ch.ph.body'), 1)] : ''}
        <label class="field" style="grid-column: 1 / -1"><span>${t('card.clothing')} <span class="sub">${st.start === 'photo' ? t('ch.clothingPhoto') : t('ch.clothingNote')}</span></span>
          <input class="in" type="text" data-card="clothing" value="${st.card.clothing || ''}" placeholder="${t('ch.ph.clothing')}"></label>
      </div></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('ch.master')}</h2><span class="sub mono">Qwen-Image 2.1 · ${st.mp}MP</span><div class="grow"></div>
        <button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      <div class="row" style="gap: 10px"><span class="lbl">${t('ch.cands')}</span>${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}<span class="sub">${t('ch.candsNote')}</span></div>
      ${ui.adv ? html`<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('def.mp')}<select class="in" data-num="mp"><option value="4" ${st.mp == 4 ? 'selected' : ''}>${t('def.mp.4')}</option><option value="1" ${st.mp == 1 ? 'selected' : ''}>${t('def.mp.1')}</option></select></label>
        <label class="field">${t('ch.seed')}<input class="in mono" data-num="seed" inputmode="numeric" value="${st.seed}"></label>
        <label class="field">${t('def.steps')}<input class="in mono" data-num="steps" inputmode="numeric" value="${st.steps}"></label>
        ${cfgFields(st)}
        <label class="tog" style="grid-column: 1 / -1"><input type="checkbox" data-flag="seedFixed" ${st.seedFixed ? 'checked' : ''}>${t('def.seedFixed')}</label>
        <label class="tog" style="grid-column: 1 / -1"><input type="checkbox" data-flag="autoRetry" ${st.autoRetry ? 'checked' : ''}>${t('def.autoRetry')}</label></div>` : ''}</div>

    <div class="sec"><div class="row" style="gap: 10px"><h2 class="h3">${t('ch.sheet')}</h2><span class="sub">${t('ch.sheetNote')}</span></div></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><label class="tog" style="font-weight: 600; min-height: 30px"><input type="checkbox" data-flag="expr" ${st.expr.on ? 'checked' : ''}>${t('ch.exprOn')}</label>
        <span class="sub">${t('ch.exprNote')}</span></div>
      ${st.expr.on ? html`<div class="row" role="group" aria-label="${t('ch.exprPick')}" style="gap: 6px">${EXPR_IDS.map((id) => html`
        <button type="button" class="chip" data-act="expr" data-v="${id}" aria-pressed="${st.expr.ids.includes(id)}">${t('ex.' + id)}</button>`)}</div>
        <div class="row" style="gap: 10px"><span class="lbl">${t('ch.strength')}</span>${seg('strength', [[0, t('ch.s0')], [1, t('ch.s1')], [2, t('ch.s2')]], st.expr.strength, t('ch.strength'))}</div>
        ${customRows()}` : ''}</div>
`.__raw;
  renderActions();
}

// a registered character's outfits: open one, or + a new one
function wardrobe() {
  const tile = (code, label, master, cur) => html`<a href="#/image/character?code=${code}" class="col" title="${code}"
    style="gap: 4px; width: 72px; padding: 4px; text-decoration: none; border: 1px solid ${cur ? 'var(--blue-line)' : '#2A2A30'}; border-radius: 8px; color: var(--fg3)">
    <div class="ph" style="width: 100%; aspect-ratio: 3 / 4">${master ? html`<img src="${filesUrl(`characters/${code}/${master}`)}" alt="" loading="lazy">` : ''}</div>
    <span style="font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${label}</span></a>`;
  return html`<div class="sec">
    <div class="row" style="gap: 4px 10px"><h2 class="h3">${t('ch.wardrobe')}</h2><span class="sub">${t('ch.wardrobeNote')}</span></div>
    <div class="row" style="gap: 8px; align-items: flex-start">
      ${tile(st.code, t('ch.outfitBase'), st.registered, true)}
      ${outfits.map((o) => tile(o.code, o.code.slice(st.code.length + 1), o.master, false))}
      <a class="slot" href="#/image/character?code=${st.code}&outfit=new" title="${t('ch.addOutfit')}" style="width: 72px; aspect-ratio: 3 / 4; text-decoration: none">${icon('plus', 16)}<span>${t('ch.addOutfitShort')}</span></a></div></div>`;
}

// 목소리 (기능 17): the character's reference voice — the voice in words (card) + a line → Qwen3-TTS candidates → ●
// (characters/<code>/<code>_voice_vN.wav), or the boss's own recording uploaded as it is (_voice_vN.m4a …).
// Dialogue cuts are converted to the newest one.
const VOICE_LINE = '안녕하세요. 오늘도 잘 부탁드립니다. 정말 그렇게 생각하세요?';
function voiceBlock() {
  return html`<div class="sec" data-voiceblk>${voiceInner()}</div>`;
}
function renderVoice() {   // the block alone, on every poll (candidates come in) — not while typing in it
  const el = root && root.querySelector('[data-voiceblk]');
  if (el && !el.contains(document.activeElement)) el.innerHTML = voiceInner().__raw;
}
function voiceInner() {
  const cands = (st.voiceJobs || []).flatMap((id) => {
    const j = job(id);
    if (!j) return [];
    const fs = j.files.filter((f) => /^voice_\d+\.wav$/.test(f.name) && f.synced).map((f) => ({ j, name: f.name }));
    return fs.concat(active(j) ? [{ j, pending: true }] : []);
  });
  const reg = (st.voices || []).slice(-1)[0];
  return html`
    <div class="row" style="gap: 4px 10px"><h2 class="h3">${t('ch.voice')}</h2><span class="sub">${t('ch.voiceNote')}</span></div>
    ${reg ? html`<div class="row" style="gap: 8px"><span class="chip mono">${reg}</span><audio controls preload="none" src="${filesUrl(`characters/${st.code}/${reg}`)}" style="height: 32px"></audio></div>` :
      html`<span class="sub">${t('ch.voiceNone')}</span>`}
    <label class="field">${t('ch.voiceText')}<input class="in" type="text" data-voice="text" value="${st.voiceText || ''}" placeholder="${t('ch.ph.voice')}"></label>
    <label class="field">${t('ch.voiceLine')}<input class="in" type="text" data-voice="line" value="${st.voiceLine || VOICE_LINE}"></label>
    <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="voiceGo" ${(st.voiceText || '').trim() ? '' : 'disabled'}>${t('ch.voiceGo')}</button>
      <span class="sub">${t('ch.voiceGoNote')}</span></div>
    <div class="row" style="gap: 8px"><label class="btn ghost" style="cursor: pointer">${icon('up', 14)} ${ui.voiceUp ? t('common.loading') : t('ch.voiceUp')}
      <input type="file" accept="audio/*,.m4a,.mp3,.wav,.flac,.ogg,.aac" data-voicefile hidden></label>
      <span class="sub">${t('ch.voiceUpNote')}</span></div>
    ${cands.length ? html`<div class="col" style="gap: 6px">${cands.map((c) => c.pending ? html`<span class="sub">${t('common.loading')}</span>` :
      html`<div class="row" style="gap: 8px; flex-wrap: nowrap"><audio controls preload="none" src="${jobFileUrl(c.j, c.name)}" style="height: 32px; width: 230px"></audio>
        <button type="button" class="btn ghost" data-act="voicePick" data-job="${c.j.id}" data-name="${c.name}" style="min-height: 30px">● ${t('ch.voicePick')}</button></div>`)}</div>` : ''}`;
}

// an outfit's left pane: whose outfit, its style (file name), the new clothes and / or a picture of them
function renderOutfitSet(el, resumable) {
  const locked = !!st.code;
  const who = baseInfo && baseInfo.card ? (baseInfo.card.name || st.of) : st.of;
  el.innerHTML = html`
    <div>
      <div class="lbl">${t('ch.crumb')} · ${t('ch.wardrobe')}</div>
      <h1 class="h1">${who} · ${st.code ? st.code.slice(st.of.length + 1) : t('ch.newOutfit')} <span class="mono" style="font-weight: 400; color: var(--fg3)" data-outfit-code>${outfitCode()}</span></h1>
      <p class="lead">${t('lead.outfit')}</p></div>
    ${!started() && resumable.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${resumable.map((c) => html`<a class="chip" href="#/image/character?code=${c}" style="display: inline-flex; align-items: center; text-decoration: none">${c}</a>`)}</div>` : ''}
    <div class="col" style="gap: 6px">
      ${seg('mode', [['all', t('ch.modeAll')], ['step', t('ch.modeStep')]], st.mode, t('ch.mode'))}
      <span class="sub">${st.mode === 'all' ? t('ch.modeAllHint') : t('ch.modeStepHint')}</span></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('ch.outfit')}</h2><div class="grow"></div>
        <button type="button" class="btn ghost" data-act="en" aria-expanded="${ui.en}" style="min-height: 30px; padding: 0 8px; font-size: 12px; color: var(--blue)">${t('ch.english')}</button></div>
      ${ui.en ? html`<div class="col" style="gap: 6px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
        <textarea class="in mono" rows="8" data-en style="font-size: 12px">${st.prompt || ui.enText}</textarea>
        <div class="row" style="justify-content: space-between"><span class="sub">${st.prompt ? t('ch.enEdited') : t('ch.enFromCard')}</span>
          ${st.prompt ? html`<button type="button" class="ib m" data-act="enReset" aria-label="${t('ch.enReset')}" title="${t('ch.enReset')}">${icon('redo')}</button>` : ''}</div></div>` : ''}
      <div class="row" style="gap: 10px; flex-wrap: nowrap; align-items: center">
        <div class="ph" style="width: 56px; height: 75px; flex: 0 0 auto">${baseInfo && baseInfo.master ? html`<img src="${filesUrl(`characters/${st.of}/${baseInfo.master}`)}" alt="">` : ''}</div>
        <span class="sub">${t('ch.outfitOf', { code: st.of })}</span></div>
      <label class="field">${t('ch.style')}<input class="in mono" data-style value="${st.outfitStyle}" placeholder="${t('ch.stylePh')}" ${locked ? 'disabled' : ''}></label>
      <label class="field"><span>${t('ch.outfitCloth')} <span class="sub">${t('ch.outfitClothNote')}</span></span>
        <textarea class="in" rows="3" data-card="clothing" placeholder="${t('ch.ph.clothing')}">${st.card.clothing || ''}</textarea></label>
      <div class="field"><span>${t('ch.outfitPic')} <span class="sub">${t('ch.outfitPicNote')}</span></span>
        ${st.outfitImage ? html`<div class="ph" style="width: 96px; height: 128px"><img src="${filesUrl(st.outfitImage.rel)}" alt="${st.outfitImage.name}" style="object-fit: contain">
            <button type="button" class="ib" data-act="clearOutfitPic" aria-label="${t('vg.remove')}" title="${t('vg.remove')}" style="position: absolute; right: 3px; top: 3px; width: 24px; height: 24px; background: #16161AE6">${icon('x', 12)}</button></div>` :
          html`<label class="slot" style="width: 96px; height: 128px; cursor: pointer">${icon('up', 16)}<span>${ui.upload ? t('common.loading') : t('vg.upload')}</span>
            <input type="file" accept="image/*" data-outfitpic hidden></label>`}</div></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('ch.master')}</h2><span class="sub mono">Qwen-Image 2.1 · ${st.mp}MP</span><div class="grow"></div>
        <button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      <div class="row" style="gap: 10px"><span class="lbl">${t('ch.cands')}</span>${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}<span class="sub">${t('ch.candsNote')}</span></div>
      ${ui.adv ? html`<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('ch.seed')}<input class="in mono" data-num="seed" inputmode="numeric" value="${st.seed}"></label>
        <label class="field">${t('def.steps')}<input class="in mono" data-num="steps" inputmode="numeric" value="${st.steps}"></label>
        ${cfgFields(st)}
        <label class="tog" style="grid-column: 1 / -1"><input type="checkbox" data-flag="autoRetry" ${st.autoRetry ? 'checked' : ''}>${t('def.autoRetry')}</label></div>` : ''}</div>

    <div class="sec"><div class="row" style="gap: 10px"><h2 class="h3">${t('ch.sheet')}</h2><span class="sub">${t('ch.sheetNote')}</span></div></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><label class="tog" style="font-weight: 600; min-height: 30px"><input type="checkbox" data-flag="expr" ${st.expr.on ? 'checked' : ''}>${t('ch.exprOn')}</label>
        <span class="sub">${t('ch.exprNote')}</span></div>
      ${st.expr.on ? html`<div class="row" role="group" aria-label="${t('ch.exprPick')}" style="gap: 6px">${EXPR_IDS.map((id) => html`
        <button type="button" class="chip" data-act="expr" data-v="${id}" aria-pressed="${st.expr.ids.includes(id)}">${t('ex.' + id)}</button>`)}</div>
        <div class="row" style="gap: 10px"><span class="lbl">${t('ch.strength')}</span>${seg('strength', [[0, t('ch.s0')], [1, t('ch.s1')], [2, t('ch.s2')]], st.expr.strength, t('ch.strength'))}</div>
        ${customRows()}` : ''}</div>
`.__raw;
  renderActions();
}

// why 시작 is off — the same checks start() makes
function problems() {
  if (st.of && !(st.card.clothing || '').trim() && !st.outfitImage && !st.prompt) return [t('ch.needOutfit')];
  if (!st.of && st.start === 'photo' && !st.photo) return [t('ch.needPhoto')];
  if (!st.of && st.start === 'card' && !(st.card.face || st.card.hair || st.card.body || st.prompt)) return [t('ch.needCard')];
  return [];
}
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const failed = sessionJobs().map(job).filter((j) => j && j.state === 'failed');
  let body;
  const est = Math.ceil(st.cands / 2) * SEC[Number(st.mp) > 2 ? 4 : 1] * cfgX(st) +
    (st.mode === 'all' ? Math.ceil((PANELS.length + (st.expr.on ? st.expr.ids.length + customs().length : 0)) / 2) * SEC.panel : 0);
  const pr = problems();   // top right of the results pane (boss, session #14): the note, then the button
  if (!started()) body = html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : (st.mode === 'all' ? t('ch.startAll') : t('ch.startStep')) + ' · ' + t('fn.estimate', { t: fmtDur(30 + est) })}</span>
    <button type="button" class="btn p" data-act="start" ${pr.length ? 'disabled' : ''}>${t('ch.start')}</button>`;
  else if (running()) body = html`<span class="note">${t('ch.runningNote')}</span><button type="button" class="btn" disabled>${t('ch.running')}</button>`;
  else if (dirty()) body = html`<span class="note">${t('ch.rerunNote')}</span><button type="button" class="btn p" data-act="rerun">${t('ch.rerun')}</button>`;
  else if (failed.length) body = html`<span class="note">${t('ch.resumeNote')}</span><button type="button" class="btn p" data-act="resume">${t('jobs.resume')}</button>`;
  else body = '';
  el.innerHTML = html`${body}`.__raw;
  el.hidden = !body;
}

// ---- rendering: right (results) ----
function stateText(s) {
  return { wait: t('ch.st.wait'), run: (s.pct || 0) + '%', done: t('ch.st.done'), failed: failWord(s.stopped), pick: t('ch.st.pick'), reg: t('ch.st.reg') }[s.state];
}
const stateDot = (s) => ({ wait: '', run: 'busy', done: 'ok', failed: s.stopped ? '' : 'bad', pick: 'warn', reg: 'ok' }[s.state]);

function tileMaster(c, i) {
  if (c.pending) return html`<div class="ph sk" style="aspect-ratio: 2 / 3"></div>`;
  if (c.failed) return html`<div class="ph" style="aspect-ratio: 2 / 3; display: flex; align-items: center; justify-content: center"><span class="sub" style="color: ${failColor(c.stopped)}">${failWord(c.stopped)}</span></div>`;
  const ref = c.registered ? { registered: c.registered } : { job: c.job, name: c.name };
  const isBase = sameRef(st.base, ref);
  const saved = !c.registered && st.saved.some((x) => sameRef(x, ref));
  return html`<div class="col ${enter('ch:' + (c.registered || c.job + '/' + c.name))}" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid ${isBase ? 'var(--blue-line)' : '#24242A'}; border-radius: 10px">
    <a class="ph" href="${c.url}" target="_blank" style="aspect-ratio: 2 / 3; display: block"><img src="${c.url}" alt="" loading="lazy">
      <span class="mono" style="position: absolute; left: 8px; top: 8px; padding: 1px 8px; border-radius: 6px; background: #0E0E10CC; font-size: 12px; color: var(--fg2)">${c.registered ? t('ch.registered') : 'seed ' + (c.seed ?? '')}</span></a>
    <div class="row" style="gap: 2px">
      ${c.registered ? '' : html`<button type="button" class="ib m" data-act="star" data-i="${i}" aria-pressed="${saved}" aria-label="${t('ch.saveCand')}" title="${t('ch.saveCand')}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="${saved ? '#FACC15' : 'none'}" stroke="${saved ? '#FACC15' : 'currentColor'}" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg></button>`}
      <button type="button" class="ib m" data-act="base" data-i="${i}" aria-pressed="${isBase}" aria-label="${t('ch.baseCand')}" title="${t('ch.baseCand')}">
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="${isBase ? '#8AB4F8' : 'none'}"/></svg></button>
      <div class="grow"></div>
      ${c.check ? html`<span class="sub row" style="gap: 5px" title="${c.check.tip}"><span class="dot ${c.check.cls}"></span>${c.check.label}</span>` : ''}</div></div>`;
}

// 기능 12: expression panels written as a sentence of their own — English id (the file kind ex_<id>, the script's
// `FL01 smirk`) + the expression sentence
function customRows() {
  return html`<div class="col" style="gap: 6px"><span class="lbl">${t('ch.exprCustom')}</span>
    ${(st.expr.custom || []).map((x, i) => html`<div class="row" style="gap: 6px; flex-wrap: nowrap">
      <input class="in mono" data-xid="${i}" value="${x.id || ''}" placeholder="smirk" title="${t('ch.exprIdTip')}" style="flex: 0 0 96px">
      <input class="in" data-xtext="${i}" value="${x.text || ''}" placeholder="${t('ch.exprTextPh')}" style="flex: 1 1 0; min-width: 0">
      <button type="button" class="ib m" data-act="delX" data-v="${i}" aria-label="${t('vg.remove')}" title="${t('vg.remove')}">${icon('x', 14)}</button></div>`)}
    <button type="button" class="btn add" data-act="addX">${icon('plus', 14)}${t('ch.exprAdd')}</button></div>`;
}
function panelTile(kind, files, j, units, ratio) {
  const f = j && j.files.find((x) => x.name === kind + '.png' && x.synced);
  const u = units.find((x) => x.spec.kind === kind);
  const busy = u && (u.state === 'pending' || u.state === 'running');
  const mm = (meta[st.sheetJob] || {})[kind + '.png'];
  // profile and back panels: the face check means nothing there (no retries either) — only the body check shows
  const chk = checkOf(mm && NOFACE.includes(kind) ? Object.assign({}, mm, { passed: null }) : mm);
  return html`<div class="col" style="gap: 4px; min-width: 0">
    <div class="ph ${busy ? 'sk' : ''}" style="aspect-ratio: ${ratio}">
      ${f && !busy ? html`<a class="${enter('chp:' + f.sha)}" href="${jobFileUrl(j, f.name)}" target="_blank"><img src="${jobFileUrl(j, f.name)}?v=${f.sha.slice(0, 8)}" alt="" loading="lazy"></a>` : ''}
      ${u && u.state === 'failed' ? html`<span class="sub" style="position: absolute; inset: 40% 0; text-align: center; color: ${failColor(unitStopped(u))}">${failWord(unitStopped(u))}</span>` : ''}
      ${chk && !busy ? html`<span class="dot ${chk.cls}" title="${chk.tip}" style="position: absolute; left: 6px; top: 6px"></span>` : ''}
      ${u && !busy && !running() ? html`<button type="button" class="ib" data-act="redoUnit" data-idx="${u.idx}" aria-label="${t('ch.redoOne')}" title="${t('ch.redoOne')}"
        style="position: absolute; right: 4px; bottom: 4px; width: 28px; height: 28px; background: #16161AE6">${icon('redo', 13)}</button>` : ''}</div>
    <div class="sub" style="text-align: center; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${tOr('pn.' + kind, kind.replace(/^ex_/, ''))}</div></div>`;
}

function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const m = stageOf('m'), s = stageOf('s'), e = stageOf('e');
  const steps = [['m', t('ch.master'), m], ['s', t('ch.sheet'), s]].concat(st.expr.on ? [['e', t('ch.expr'), e]] : []);
  const cands = candidates();
  const stale = st.sheetJob && (!sameRef(st.base, st.sheetBase) || st.sheetExpr !== exprKey());
  const sj = st.sheetJob && job(st.sheetJob);
  const units = sheetUnits();
  const canSheet = !running() && st.base && !st.sheetJob;
  const finished = st.base && sj && sj.state === 'done' && !stale;
  const unsaved = cands.filter((c) => c.url && !c.registered && !sameRef(st.base, c) && !st.saved.some((x) => sameRef(x, c))).length;
  const exprIds = st.sheetExpr ? EXPR_IDS.filter((id) => (JSON.parse(st.sheetExpr)[0] || []).includes(id)).concat((JSON.parse(st.sheetExpr)[2] || []).map((x) => x[0])) : [];
  const dim = (x) => dirty() || (stale && x) ? 'opacity: .45' : '';
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px" aria-label="${t('ch.progress')}">${steps.map(([, label, x]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
          <span class="dot ${stateDot(x)}"></span>${label}<span class="sub mono">${stateText(x)}</span></li>`)}</ol>
      <div class="toprow-end">${running() ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.askStop')}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${dirty() ? html`<div class="banner warn" role="status">${t('ch.dirty')}</div>` : ''}

    <section class="card" style="${dirty() ? 'opacity: .45' : ''}">
      <div class="row"><span class="num">1</span><h2 class="h2">${t('ch.master')}</h2>
        ${started() ? html`<span class="sub mono">${t('ch.candN', { n: cands.filter((c) => c.url).length })}</span>` : ''}<div class="grow"></div>
        <span class="sub mono">${stateText(m)}</span>
        ${m.state !== 'wait' && !running() ? html`<button type="button" class="ib m" data-act="more" aria-expanded="${ui.redo}" aria-label="${t('ch.more')}" title="${t('ch.more')}">${icon('redo')}</button>` : ''}</div>
      ${m.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${m.pct}%"></div></div>` : ''}
      ${m.state === 'wait' ? html`<p class="sub" style="margin: 6px 0 0 38px">${t('ch.masterWait', { n: st.cands })}</p>` :
        html`<p class="sub" style="margin: 6px 0 0 38px; color: ${m.state === 'pick' ? 'var(--warn-fg)' : 'var(--fg4)'}">${m.state === 'pick' ? t('ch.pickBase') : st.mode === 'all' ? t('ch.autoBase') : t('ch.baseRule')}</p>`}
      ${ui.redo ? html`<div class="col" style="margin-top: 12px; padding: 14px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px; gap: 10px">
        <label class="lbl" for="ch-redo">${t('ch.prompt')}</label>
        <textarea id="ch-redo" class="in mono" rows="6" data-redo style="font-size: 12px">${ui.redoText}</textarea>
        <div class="row" style="justify-content: space-between"><span class="lbl" data-redo-hint>${t(ui.redoText.trim() === ui.redoBase ? 'ch.moreSame' : 'ch.moreEdited', { n: st.cands })}</span>
          <div class="row" style="gap: 8px"><button type="button" class="ib" data-act="moreNo" aria-label="${t('common.cancel')}" title="${t('common.cancel')}" style="background: transparent">${icon('x')}</button>
          <button type="button" class="ib p" data-act="moreGo" aria-label="${t('ch.more')}" title="${t('ch.more')}">${icon('redo')}</button></div></div></div>` : ''}
      ${cands.length ? html`<div class="grid" style="margin-top: 14px; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr))">${cands.map(tileMaster)}</div>` : ''}
      ${canSheet ? html`<div class="row" style="margin-top: 14px; justify-content: flex-end"><button type="button" class="btn p" data-act="sheet">${st.expr.on ? t('ch.makeSheetExpr') : t('ch.makeSheet')}</button></div>` : ''}
    </section>

    ${stale && !running() ? html`<div class="row banner warn" role="status" style="justify-content: space-between">
      <span>${!sameRef(st.base, st.sheetBase) ? t('ch.staleBase') : t('ch.staleExpr')}</span>
      <button type="button" class="ib p" data-act="sheet" aria-label="${t('ch.rebuild')}" title="${t('ch.rebuild')}">${icon('redo')}</button></div>` : ''}

    <section class="card" style="${dim(1)}">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('ch.sheet')}</h2><span class="sub mono">${t('ch.sheetMeta')}</span><div class="grow"></div>
        <span class="sub mono">${stateText(s)}</span>
        ${sj && sj.files.some((f) => f.name === 'sheet.png' && f.synced) ? html`<a class="ib m" href="${jobFileUrl(sj, 'sheet.png')}" target="_blank" aria-label="${t('ch.sheetOpen')}" title="${t('ch.sheetOpen')}">${icon('open')}</a>` : ''}</div>
      ${s.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${s.pct}%"></div></div>` : ''}
      ${st.sheetJob ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 8px">
        ${PANELS.map((k) => panelTile(k, null, sj, units, '2 / 3'))}</div>` : st.regSheet ? html`<div class="col" style="margin-top: 14px; gap: 6px">
        <a class="ph" href="${filesUrl(`characters/${st.code}/${st.regSheet}`)}" target="_blank" style="display: block; aspect-ratio: 3440 / 1280"><img src="${filesUrl(`characters/${st.code}/${st.regSheet}`)}" alt="${st.regSheet}" loading="lazy"></a>
        <span class="sub">${t('ch.regSheet')} · <span class="mono">${st.regSheet}</span> · ${t('ch.regNote')}</span></div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${t('ch.sheetWait')}</p>`}
    </section>

    ${st.expr.on || exprIds.length || (!st.sheetJob && (st.regExpr || []).length) ? html`<section class="card" style="${dim(1)}">
      <div class="row"><span class="num">3</span><h2 class="h2">${t('ch.expr')}</h2>
        <span class="sub mono">${t('ch.exprMeta', { n: st.sheetJob ? exprIds.length : st.expr.ids.length + customs().length, s: t('ch.s' + st.expr.strength) })}</span><div class="grow"></div>
        <span class="sub mono">${stateText(e)}</span></div>
      ${e.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${e.pct}%"></div></div>` : ''}
      ${st.sheetJob && exprIds.length ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 10px">
        ${exprIds.map((id) => panelTile('ex_' + id, null, sj, units, '2 / 3'))}</div>` : !st.sheetJob && (st.regExpr || []).length ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 10px">
        ${st.regExpr.map((n) => html`<div class="col" style="gap: 4px; min-width: 0"><a class="ph" href="${filesUrl(`characters/${st.code}/${n}`)}" target="_blank" style="display: block; aspect-ratio: 2 / 3"><img src="${filesUrl(`characters/${st.code}/${n}`)}" alt="${n}" loading="lazy"></a>
          <div class="sub" style="text-align: center; font-size: 11px">${((k) => tOr('pn.' + k, k.replace(/^ex_/, '')))(n.replace(/_v\d+\.png$/, '').replace(/^[A-Z]+\d+(-[a-z0-9]+)?_/, ''))}</div></div>`)}</div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${t('ch.exprWait')}</p>`}
    </section>` : ''}

    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('ch.askClose', { n: unsaved })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button>
        <button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between; padding-top: 4px">
        <p class="sub" style="margin: 0">${t('ch.registerNote')}</p>
        <div class="row" style="gap: 8px">
          <button type="button" class="btn ghost" data-act="close" ${running() ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="register" ${finished && !running() ? '' : 'disabled'} title="${finished ? '' : t('ch.registerWait')}">${t(st.of ? 'ch.registerOutfit' : 'ch.register', { code: st.code })}</button></div></div>` : ''}`.__raw;
  renderActions();
}

function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(IMAGE_MENU(), '#/image/character')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- data refresh (every poll) ----
async function refresh() {
  if (!st || !st.code) return;
  for (const id of sessionJobs()) {
    const j = job(id);
    const want = id === st.sheetJob && (!detail[id] || active(j) || (j && detail[id].files.length !== j.files.length) || (j && detail[id].state !== j.state));
    if (want || !j) {
      try { detail[id] = await srv('/jobs/' + id); } catch (e) { /* server away */ }
    }
    const jj = job(id);
    const n = jj ? jj.files.filter((f) => f.synced && f.name.endsWith('.png')).map((f) => f.sha).join() : '';
    if (jj && n && (!meta[id] || meta[id]._n !== n)) {
      try { meta[id] = Object.assign(await api('/api/meta?job=' + id), { _n: n }); } catch (e) { /* not yet */ }
    }
  }
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const v = a.dataset.v;
  switch (a.dataset.act) {
    case 'set': {
      const k = a.dataset.k;
      if (k === 'strength') st.expr.strength = Number(v);
      else st[k] = ['cands'].includes(k) ? Number(v) : v;
      if ((k === 'gender' || k === 'rank') && !st.code) await previewCode();
      if (k === 'start') { st.photo = null; ui.enText = ''; }
      renderSet(); renderOut();
      if (st.code) save();
      return;
    }
    case 'addX': st.expr.custom = (st.expr.custom || []).concat([{ id: '', text: '' }]); renderSet(); return;
    case 'delX': st.expr.custom.splice(Number(v), 1); renderSet(); renderOut(); if (st.code) save(); return;
    case 'expr': {
      const ids = new Set(st.expr.ids);
      ids.has(v) ? ids.delete(v) : ids.add(v);
      st.expr.ids = EXPR_IDS.filter((x) => ids.has(x));
      renderSet(); renderOut();
      if (st.code) save();
      return;
    }
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'voiceGo': {   // the sentence goes on the card (dialogue cuts and later candidates use it), then 2 candidates
      const card = Object.fromEntries(CARD_KEYS.map((k) => [k, (st.card[k] || '').trim()]).filter(([, x]) => x));
      if (!(await act(() => api('/api/codes', { code: st.code, card: Object.assign(card, { voice: st.voiceText.trim() }) })))) return;
      const r = await act(() => api('/api/tts', { code: st.code, instruct: st.voiceText.trim(), text: (st.voiceLine || VOICE_LINE).trim(), count: 2,
        seed: 1 + 2 * (st.voiceJobs || []).length, label: (st.card.name || st.code) + ' · ' + t('ch.voice') }));
      if (!r) return;
      st.voiceJobs = (st.voiceJobs || []).concat([r.id]);
      await save(); await poll(); return renderSet();
    }
    case 'voicePick': {
      const r = await act(() => api('/api/voice/save', { code: st.code, job: a.dataset.job, name: a.dataset.name, instruct: st.voiceText, text: st.voiceLine || VOICE_LINE }));
      if (r) { st.voices = r.voices; toast(t('ch.voiceSaved', { name: r.saved })); }
      return renderSet();
    }
    case 'clearOutfitPic': st.outfitImage = null; renderSet(); renderActions(); renderOut(); if (st.code) save(); return;
    case 'en': ui.en = !ui.en; if (ui.en) await loadEnglish(); return renderSet();
    case 'enReset': st.prompt = null; await loadEnglish(); renderSet(); return renderOut();
    case 'start': return start();
    case 'rerun': return rerunCard();
    case 'resume':
      for (const j of sessionJobs().map(job).filter((x) => x && x.state === 'failed')) await act(() => api('/api/retry', { job: j.id }));
      await poll(); return;
    case 'more':
      ui.redo = !ui.redo;
      if (ui.redo) { ui.redoText = st.prompt || await englishPrompt(); ui.redoBase = ui.redoText.trim(); }
      return renderOut();
    case 'moreNo': ui.redo = false; return renderOut();
    case 'moreGo': {
      const edited = ui.redoText.trim() !== ui.redoBase;
      ui.redo = false;
      if (await act(() => submitMaster(edited ? ui.redoText.trim() : null))) { /* submitMaster returns undefined */ }
      await save(); await poll(); return renderOut();
    }
    case 'star': case 'base': {
      const c = candidates()[Number(a.dataset.i)];
      const ref = c.registered ? { registered: c.registered } : { job: c.job, name: c.name };
      if (a.dataset.act === 'base') st.base = ref;
      else st.saved = st.saved.some((x) => sameRef(x, ref)) ? st.saved.filter((x) => !sameRef(x, ref)) : st.saved.concat([ref]);
      renderOut(); renderActions();
      return save();
    }
    case 'sheet': await act(submitSheet); await save(); await poll(); return;
    case 'redoUnit': await act(() => srv(`/jobs/${st.sheetJob}/redo`, { idx: Number(a.dataset.idx) })); await poll(); return;
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop':
      ui.askStop = false;
      for (const id of sessionJobs().filter((x) => active(job(x)))) await act(() => srv(`/jobs/${id}/cancel`, {}));
      await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose':
      ui.askClose = false;
      if (await act(() => api('/api/discard', { code: st.code }))) return reset();
      return renderOut();
    case 'register': {
      const r = await act(() => api('/api/register', { code: st.code }));
      if (r) { toast(t('ch.registered2', { code: st.code, n: r.saved.length })); return reset(); }
      return;
    }
    default:
  }
}

function onInput(e) {
  const d = e.target.dataset;
  if (cfgInput(e, st)) return renderActions();
  if (d.xid !== undefined || d.xtext !== undefined) {   // 기능 12: a custom expression's id (lower-case English) / sentence
    const x = st.expr.custom[Number(d.xid ?? d.xtext)];
    if (d.xid !== undefined) { x.id = e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''); if (e.target.value !== x.id) e.target.value = x.id; } else x.text = e.target.value;
    renderActions();
    clearTimeout(onInput.timer);
    if (st.code) onInput.timer = setTimeout(save, 800);
    return;
  }
  if (d.voice !== undefined) {   // the voice block's sentence / line (kept in the session)
    st[d.voice === 'text' ? 'voiceText' : 'voiceLine'] = e.target.value;
    const b = root.querySelector('[data-act=voiceGo]');
    if (b) b.disabled = !(st.voiceText || '').trim();
    clearTimeout(onInput.timer);
    if (st.code) onInput.timer = setTimeout(save, 800);
    return;
  }
  if (d.card !== undefined) {
    st.card[d.card] = e.target.value;
    if (d.card === 'name') { const h = root.querySelector('[data-set] h1'); if (h) h.firstChild.textContent = (e.target.value || (st.code ? '' : t('ch.new'))) + ' '; }
    renderActions();
    if (started()) renderOut();
    clearTimeout(onInput.timer);
    if (st.code) onInput.timer = setTimeout(save, 800);
  } else if (d.en !== undefined) {
    st.prompt = e.target.value.trim() === ui.enText.trim() ? null : e.target.value;
  } else if (d.redo !== undefined) {
    ui.redoText = e.target.value;
    const h = root.querySelector('[data-redo-hint]');
    if (h) h.textContent = t(ui.redoText.trim() === ui.redoBase ? 'ch.moreSame' : 'ch.moreEdited', { n: st.cands });
  } else if (d.num !== undefined) {
    st[d.num] = Number(e.target.value) || 0;
  } else if (d.style !== undefined) {
    st.outfitStyle = e.target.value.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 20);
    if (e.target.value !== st.outfitStyle) e.target.value = st.outfitStyle;
    const c = root.querySelector('[data-outfit-code]');
    if (c) c.textContent = outfitCode();
  }
}

async function onChange(e) {
  const d = e.target.dataset;
  if (d.flag !== undefined) {
    if (d.flag === 'expr') st.expr.on = e.target.checked;
    else st[d.flag] = e.target.checked;
    renderSet(); renderOut();
    if (st.code) save();
  } else if (d.outfitpic !== undefined && e.target.files[0]) {
    ui.upload = true; renderSet();
    const f = e.target.files[0];
    const r = await act(async () => {
      const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.status);
      return j;
    });
    ui.upload = false;
    if (r) st.outfitImage = { rel: r.rel, name: f.name };
    renderSet(); renderActions(); renderOut();
    if (st.code) save();
  } else if (d.voicefile !== undefined && e.target.files[0]) {   // the boss's own recording → ● reference voice at once
    const f = e.target.files[0];
    ui.voiceUp = true; e.target.blur(); renderVoice();
    const r = await act(async () => {
      const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.status);
      return j;
    });
    const sv = r && await act(() => api('/api/voice/save', { code: st.code, asset: r.rel, file: f.name }));
    ui.voiceUp = false;
    if (sv) { st.voices = sv.voices; toast(t('ch.voiceSaved', { name: sv.saved })); }
    renderVoice();
  } else if (d.photo !== undefined && e.target.files[0]) {
    ui.upload = true; renderSet();
    const f = e.target.files[0];
    const r = await act(async () => {
      const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || res.status);
      return j;
    });
    ui.upload = false;
    if (r) st.photo = r.sha;
    renderSet(); renderActions(); renderOut();
  } else if (d.num !== undefined) {
    st[d.num] = Number(e.target.value) || 0;
    if (d.num === 'mp') renderSet();
  }
}

// ---- English prompt (what the template makes of the card; editing it overrides the card) ----
function cardOptions() {
  const c = st.card, f = st.gender === 'F', kid = Number(String(c.age || '').match(/\d+/)) > 0 && Number(String(c.age).match(/\d+/)) < 13;   // 기능 13
  const who = 'a Korean ' + (f ? (kid ? 'girl' : 'woman') : (kid ? 'boy' : 'man')) + ' who looks ' + (c.age ? c.age + ' years old' : 'in the twenties') + (c.extra ? ', ' + c.extra : '');
  return { who, gender: f ? 'female' : 'male', child: kid, face: c.face, hair: c.hair, body: c.body, clothing: c.clothing };
}
async function englishPrompt() {
  const r = await act(() => srv('/prompt', st.of ? { type: 'char.derive', options: { kind: 'outfit', gender: st.gender === 'F' ? 'female' : 'male',
    clothing: (st.card.clothing || '').trim(), outfit_pic: !!st.outfitImage } } :
    { type: st.start === 'photo' ? 'char.from_photo' : 'char.ref', options: cardOptions() }));
  return (r && r.prompt) || '';
}
async function loadEnglish() { ui.enText = await englishPrompt(); }
async function previewCode() {
  const r = await act(() => api(st.of ? `/api/nextcode?of=${st.of}` : `/api/nextcode?g=${st.gender}&r=${st.rank}`));
  codePreview = r ? r.code : '';
}

async function reset() {
  st = fresh();
  meta = {}; detail = {};
  ui = { adv: false, en: false, enText: '', redo: false, redoText: '', askClose: false, askStop: false, upload: false, voiceUp: false };
  history.replaceState(null, '', '#/image/character');
  await load(null);
}

async function load(code) {
  open = await act(() => api('/api/sessions')) || [];
  if (code) {
    const r = await act(() => api('/api/session?code=' + encodeURIComponent(code)));
    if (r && r.session && r.session.code) st = Object.assign(fresh(), r.session);
    else if (r) {   // a registered character opened for editing: its card, its master as the starting ●
      st = fresh();
      st.code = code; st.gender = code[0]; st.rank = code[1];
      for (const k of CARD_KEYS) if (r.card[k]) st.card[k] = r.card[k];
      if (r.card.of) { st.of = r.card.of; st.outfitStyle = code.slice(r.card.of.length + 1); }
      const refs = r.images.filter((i) => i.kind === 'ref').sort((a, b) => b.v - a.v);
      if (refs.length) { st.registered = refs[0].name; st.base = { registered: refs[0].name }; st.cardAt = cardKey(); }
    }
    if (r) { st.voices = r.voices || []; if (st.voiceText == null) st.voiceText = (r.card && r.card.voice) || ''; }
    if (r && r.images) {
      const top = (k) => r.images.filter((i) => i.kind === k).sort((a, b) => b.v - a.v)[0];
      st.regSheet = (top('sheet') || {}).name || null;
      st.regExpr = [...new Set(EXPR_IDS.map((id) => 'ex_' + id).concat(r.images.filter((i) => i.kind.startsWith('ex_')).map((i) => i.kind)))].map(top).filter(Boolean).map((i) => i.name);
    }
  }
  baseInfo = null; outfits = [];
  if (st.of) {
    const b = await act(() => api('/api/session?code=' + encodeURIComponent(st.of)));
    const m = b && b.images.filter((i) => i.kind === 'ref').sort((x, y) => y.v - x.v)[0];
    if (b) { baseInfo = { card: b.card, master: m ? m.name : null }; st.gender = st.of[0]; st.rank = st.of[1]; }
  } else if (st.registered) {
    const lib = await act(() => api('/api/library?scope=shared'));
    outfits = ((lib && lib.characters) || []).filter((c) => c.card.of === st.code);
  }
  if (!st.code) await previewCode();
  await refresh();
  renderAll();
}

export default {
  async mount(el, { query }) {
    root = el;
    const newOutfit = query.get('outfit') === 'new' && query.get('code');   // 옷장 › 의상 추가: a new outfit of that character
    if (newOutfit) { st = fresh(); st.of = newOutfit; history.replaceState(null, '', '#/image/character'); }
    else if (!st || query.get('code') && query.get('code') !== st.code) st = fresh();
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    await load(newOutfit ? null : query.get('code') || st.code);
  },
  async update() {
    if (!root || !st) return;
    await refresh();
    await auto();
    renderOut();
    renderActions();
    if (st.registered && !st.of) renderVoice();
  },
  unmount() { root = null; },
};
