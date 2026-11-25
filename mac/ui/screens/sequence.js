// 영상 생성 › 워크플로 시퀀스 (design §11): common settings left (인물 · 방향 · 모델 · 길이 · 후보 · 마무리), the cuts
// right — each row a cut (프롬프트, 첫 프레임, 상태), a row opens in the 한 컷 screen. Changing a common setting makes
// the finished cuts "옛 설정 기준". 이어 보기 plays the cut results in order; 내보내기 joins them (video.concat) into the
// project library. The sequence lives in projects/<p>/sequences/<id>.json, its cuts in cuts/<id>.json.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, codeLabel, codeLabelOf, isStopped, failWord, fetchJobs } from '../app.js';
import { MODELS } from '../models.js';
import { sideMenu, VIDEO_MENU } from './wip.js';
import * as C from './cut.js';
import * as P from '../prod.js';

let root = null, sq = null, cuts = [], chars = [], scenes = [], seqs = [], ui = {}, busy = false;
const freshUi = () => ({ adv: false, scenes: false, picked: new Set(), askClose: false, askStop: false, askDel: null, play: false, from: 0 });

function freshSeq() {
  const c = C.freshCut();
  return { id: null, name: '', common: Object.fromEntries(C.COMMON.map((k) => [k, c[k]])), cuts: [], export: null };
}

// ---- derived ----
const label = (c) => `${sq.name || t('sq.untitled')} · ${t('sq.cutN', { k: cuts.indexOf(c) + 1 })}`;
const allJobs = () => cuts.flatMap(C.cutJobs).concat(sq.export ? [sq.export.job] : []).concat(P.jobsOf(sq, []));
const prodOn = () => !!(sq.script && sq.prod && sq.prod.started);
let drive = true;   // this tab drives the open sequence's production (prod.js holdSeq); false: another tab does
const activeJobs = () => allJobs().map(C.jobOf).filter((j) => j && (j.state === 'pending' || j.state === 'running'));
const playing = () => [...root.querySelectorAll('[data-out] video')].some((v) => !v.paused && !v.ended);
const cutProblem = (c) => c.src ? null : !c.prompt.trim() ? t('vg.needPrompt') : C.whyNot(c, c.model, chars) ? t('vg.why.' + C.whyNot(c, c.model, chars)) : null;
const todo = () => cuts.filter((c) => !c.src && !c.cand.some((x) => x.key === C.key(c)) && !cutProblem(c));
const genCut = (c) => (c.src && cuts.find((x) => x.id === c.src)) || c;   // 기능 9: another cut's generation
const results = () => cuts.map((c) => { const g = genCut(c); return C.resultOf(g); });
// what the export depends on: the results and every editing choice (ranges, sound, breaks, screen text, fades)
const exportKey = () => JSON.stringify([results().map((r) => r && r.job + '/' + r.name),
  cuts.map((c) => [C.useOf(c), c.sound, c.src, c.shot || null, !!c.brk, c.texts || [], c.fade || null])]);

// the export's video files: part_NN.mp4 (video.edit), or seq.mp4 of an export made before (video.concat)
const exportFiles = (ex) => ex.files.filter((f) => /^part_\d+\.mp4$|^seq\.mp4$/.test(f.name)).sort((a, b) => a.name < b.name ? -1 : 1);

// ---- persistence ----
async function saveSeq() {
  const r = await act(() => api('/api/doc', { kind: 'sequences', id: sq.id, doc: sq }));
  if (r && !sq.id) { sq.id = r.id; history.replaceState(null, '', '#/video/seq?id=' + r.id); }
  return r;
}
async function saveCut(c) {
  const r = await act(() => api('/api/doc', { kind: 'cuts', id: c.id, doc: c }));
  if (r) c.id = r.id;
}
const timers = {};
const saveCutSoon = (c) => { clearTimeout(timers[c.id]); timers[c.id] = setTimeout(() => saveCut(c), 800); };

// a common setting → every cut that has not set it itself
async function setCommon(k, v) {
  sq.common[k] = v;
  if (k === 'model') {   // the model's defaults (설정 › 기본값) come with it
    const tmp = C.freshCut(sq.common);
    C.applyModel(tmp, v);
    for (const f of ['sec', 'cands', 'upOn', 'face', 'steps', 'vsa', 'refSize', 'refineOn']) sq.common[f] = tmp[f];
  }
  for (const c of cuts) {
    let ch = false;
    for (const f of C.COMMON) if (!c.over.includes(f) && JSON.stringify(c[f]) !== JSON.stringify(sq.common[f])) { c[f] = sq.common[f]; ch = true; }
    if (ch) await saveCut(c);
  }
  if (sq.id || cuts.length) await saveSeq();
  renderAll();
}

// ---- actions ----
async function addCuts(list) {   // list: [{prompt, first}]
  if (!(S.status && S.status.settings && S.status.settings.project)) return toast(t('vg.needProject'), true);
  if (!sq.id && !(await saveSeq())) return;
  for (const x of list) {
    const c = C.freshCut(Object.assign({ seq: sq.id }, sq.common, x));
    c.seed = 1;
    await saveCut(c);
    if (!c.id) return;
    cuts.push(c);
    sq.cuts.push(c.id);
  }
  await saveSeq();
  renderAll();
}
async function generate(list) {
  const skip = cuts.filter((c) => !c.cand.some((x) => x.key === C.key(c)) && cutProblem(c)).length;
  for (const c of list) {
    if (!(await act(() => C.submitCands(c, c.cands, label(c))))) { /* submitCands returns the made list */ }
    c.base = c.cand.some((x) => x.job === c.base && x.key === C.key(c)) ? c.base : null;
    await saveCut(c);
  }
  if (skip) toast(t('sq.skipped', { n: skip }), true);
  await poll();
  renderAll();
}
async function exportSeq() {
  const tl = C.timeline(cuts);
  if (!tl.cuts.length) return;
  const [w, h] = C.SIZE[sq.common.orient];
  const r = await act(() => api('/api/edit', { sources: tl.sources, speech: tl.speech, multi: tl.multi, cuts: tl.cuts, width: w, height: h,
    label: sq.name || t('sq.untitled'), codes: [...new Set(cuts.flatMap((c) => c.codes))].join(' ') }));
  if (!r) return;
  // parts: one file per episode stretch (a break inside the scene starts a new one); brk = an episode ends with it
  sq.export = { job: r.id, key: exportKey(), saved: null, parts: Array.from({ length: tl.parts }, (_, i) => ({ name: `part_${String(i).padStart(2, '0')}.mp4`, brk: i < tl.parts - 1 || tl.endBrk })) };
  await saveSeq();
  await poll();
  renderOut();
}
async function closeSeq() {
  const title = (sq.name || 'sequence').replace(/\s+/g, '_');
  const items = cuts.flatMap((c, k) => C.saveItems(c, `${title}_${k + 1}`, false));
  const r = await act(() => api('/api/finish', { items, jobs: allJobs(),
    docs: cuts.filter((c) => c.id).map((c) => ({ kind: 'cuts', id: c.id })).concat(sq.id ? [{ kind: 'sequences', id: sq.id }] : []) }));
  if (!r) return;
  toast(r.saved.length ? t('vg.savedN', { n: r.saved.length }) : t('vg.closed'));
  sq = freshSeq(); cuts = []; ui = freshUi();
  history.replaceState(null, '', '#/video/seq');
  seqs = await act(() => api('/api/docs?kind=sequences')) || [];
  await poll();
  renderAll();
}

// ---- left: common settings ----
function seg(k, opts, cur, labelText) {
  return html`<div class="seg" role="group" aria-label="${labelText}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${k}" data-v="${v}" aria-pressed="${String(cur) === String(v)}">${l}</button>`)}</div>`;
}
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const cm = sq.common;
  const addable = chars.filter((c) => c.master && !cm.codes.includes(c.code));
  const [w, h] = C.SIZE[cm.orient];
  el.innerHTML = html`
    <div><div class="lbl">${t('sq.crumb')}</div>
      <h1 class="h1">${sq.name || t('sq.new')} <span class="mono" style="font-weight: 400; color: var(--fg3)">${t('sq.meta', { n: cuts.filter((c) => !c.genOnly).length, s: cuts.filter((c) => !c.genOnly).reduce((a, c) => a + C.lenOf(c), 0).toFixed(1) })}</span></h1><p class="lead">${t('lead.seq')}</p></div>
    ${!sq.id && seqs.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${seqs.map((d) => html`<a class="chip" href="#/video/seq?id=${d.id}" style="display: inline-flex; align-items: center; text-decoration: none">${d.name || d.id}</a>`)}</div>` : ''}
    <label class="field">${t('sq.name')}<input class="in" data-k="name" value="${sq.name}" placeholder="${t('sq.namePh')}"></label>
    <div class="col" style="gap: 6px">${seg('mode', [['all', t('ch.modeAll')], ['step', t('ch.modeStep')]], cm.mode, t('ch.mode'))}
      <span class="sub">${cm.mode === 'all' ? t('sq.modeAllHint') : t('sq.modeStepHint')}</span></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('sq.common')}</h2><span class="sub">${t('sq.commonNote')}</span></div>
      <div class="field">${t('vg.people')}<div class="row" style="gap: 6px">
        ${cm.codes.map((c) => html`<span class="chip" style="display: inline-flex; align-items: center; gap: 6px">${codeLabelOf(chars, c)}
          <button type="button" class="ib m" data-act="delCode" data-code="${c}" aria-label="${t('vg.remove')} ${c}" title="${t('vg.remove')}" style="width: 22px; height: 22px">${icon('x', 12)}</button></span>`)}
        ${cm.codes.length < 2 ? html`<select class="in" data-add="code" aria-label="${t('vg.addPerson')}" style="width: 150px; min-height: 34px">
          <option value="">${t('vg.addPerson')}</option>${addable.map((c) => html`<option value="${c.code}">${codeLabel(c)}${C.hasSheet(chars, c.code) ? '' : ' · ' + t('sw.noSheet')}</option>`)}</select>` : ''}</div>
        <span class="sub">${t('sq.peopleNote')}</span></div>
      <div class="field">${t('sq.model')}${seg('model', C.MODEL_IDS.map((m) => [m, MODELS[m].name]), cm.model, t('sq.model'))}
        <span class="sub">${t('vg.note.' + cm.model)} · ${t('vg.speed.' + cm.model)}</span></div>
      <div class="row" style="gap: 10px 14px; align-items: flex-end">
        <div class="field">${t('vg.len')}${C.secInput(cm.model, cm.sec)}</div>
        <div class="field">${t('sw.orient')}${seg('orient', [['v', t('vg.portrait')], ['h', t('vg.landscape')]], cm.orient, t('sw.orient'))}</div>
        <div class="field">${t('ch.cands')}${seg('cands', [[1, '1'], [2, '2'], [4, '4']], cm.cands, t('ch.cands'))}</div></div>
      <span class="sub mono">${w}×${h}</span>
      ${cm.model === 'pih3' ? html`<div class="field">${t('def.steps')}${seg('steps', [[2, t('vg.steps2')], [4, t('vg.steps4')]], cm.steps, t('def.steps'))}</div>` : ''}
      ${cm.model === 'pih3' ? html`<div class="field" title="${t('vg.vsaNote')}">${t('def.vsa')}${seg('vsa', [[0, t('vg.st.off')], [1, t('vg.on')]], cm.vsa ? 1 : 0, t('def.vsa'))}</div>` : ''}
      ${cm.model === 'h3' ? html`<label class="field">${t('def.ref_size')}<select class="in" data-k="refSize">
        <option value="max" ${cm.refSize === 'max' ? 'selected' : ''}>${t('def.ref_size.max')}</option>
        <option value="match" ${cm.refSize === 'match' ? 'selected' : ''}>${t('def.ref_size.match')}</option></select></label>` : ''}</div>

    <div class="sec">
      <h2 class="h3">${t('vg.finish')}</h2>
      <label class="tog"><input type="checkbox" data-flag="face" ${cm.face ? 'checked' : ''}>${t('vg.face')} <span class="sub">${t('sq.faceNote')}</span></label>
      <label class="tog"><input type="checkbox" data-flag="refineOn" ${cm.refineOn ? 'checked' : ''}>${t('vg.refine')} <span class="sub">${t('sq.refineNote')}</span></label>
      <label class="tog"><input type="checkbox" data-flag="voiceOn" ${cm.voiceOn !== false ? 'checked' : ''}>${t('vg.voice')} <span class="sub">${t('vg.voiceNote')}</span></label>
      <label class="tog"><input type="checkbox" data-flag="upOn" ${cm.upOn ? 'checked' : ''}>${t('def.upscale')} <span class="sub">${cm.orient === 'h' ? '1920×1080' : '1080×1920'} · ${t('vg.upNote')}</span></label></div>`.__raw;
  renderActions();
}
// the main action, top right of the results pane (boss, session #14): the note (nothing left · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  if (sq.script && !prodOn()) {   // 기능 16: frames → ● → video, by itself from here (자동 진행) or stopping at each ● (단계별)
    el.innerHTML = html`<span class="note">${t('pr.startNote', { f: (sq.frames || []).length, c: cuts.length })}</span>
      <button type="button" class="btn p" data-act="prodStart">${t('pr.start')}</button>`.__raw;
    return;
  }
  if (prodOn() && !drive) {
    el.innerHTML = html`<span class="note">${t('pr.elsewhere')}</span>`.__raw;
    return;
  }
  const n = todo().length;
  const est = todo().reduce((a, c) => a + C.estimate(c, c.cands), 0);
  el.innerHTML = html`<span class="note">${n ? t('sq.estimate', { t: fmtDur(est) }) : cuts.length ? t('sq.nothingLeft') : t('sq.addFirst')}</span>
    <button type="button" class="btn p" data-act="genAll" ${n ? '' : 'disabled'}>${t('sq.genAll', { n })}</button>`.__raw;
}

// ---- right: cuts ----
const ROW = { wait: ['', 'ch.st.wait'], run: ['busy', ''], pick: ['warn', 'ch.st.pick'], done: ['ok', 'ch.st.done'], failed: ['bad', 'ch.st.failed'], stopped: ['', 'ch.st.stopped'], old: ['warn', 'vg.st.old'] };
function cutRow(c, k) {
  const s = C.cutState(genCut(c)), st = C.stages(genCut(c));   // a reuse cut shows its source's state
  const pct = s === 'run' ? (st.gen.state === 'run' ? st.gen.pct : st.ref.state === 'run' ? st.ref.pct : st.voice.state === 'run' ? st.voice.pct : st.up.pct || 0) : 0;
  const stepName = s !== 'run' ? '' : st.gen.state === 'run' ? t('vg.gen') : st.ref.state === 'run' ? t('vg.refineStage') : st.voice.state === 'run' ? t('vg.voice') : t('def.upscale');
  const prob = cutProblem(c);
  const g = genCut(c), r = C.resultOf(g);
  return html`<div class="item-row" style="align-items: flex-start; gap: 12px; ${s === 'old' ? 'border-color: var(--warn-line)' : ''}">
    <span class="mono lbl" style="min-width: 20px; padding-top: 8px">${k + 1}</span>
    <button type="button" class="ph" data-act="frameFor" data-k="${k}" title="${t('sq.firstFrame')}" style="flex: 0 0 ${c.orient === 'h' ? '96px' : '54px'}; aspect-ratio: ${c.orient === 'h' ? '16 / 9' : '9 / 16'}; padding: 0; cursor: pointer">
      ${c.first ? html`<img src="${filesUrl(c.first.rel)}" alt="${c.first.name}">` : html`<span class="sub" style="font-size: 10px">${t('sq.noFrame')}</span>`}</button>
    <div class="col" style="flex: 1 1 260px; min-width: 0; gap: 4px">
      <textarea class="in" rows="2" data-cut="${k}" data-f="prompt" placeholder="${t('vg.promptPh')}" aria-label="${t('sq.cutN', { k: k + 1 })} ${t('vg.prompt')}" style="font-size: 13px">${c.prompt}</textarea>
      <span class="mono sub">${c.src ? t('sq.srcOf', { k: cuts.indexOf(genCut(c)) + 1 }) : html`${t('vg.secN', { n: c.sec })} · ${MODELS[c.model].name}${c.over.length ? ' *' : ''} · ${t('ch.candN', { n: c.cands })}${c.codes.length ? ' · ' + c.codes.join(' ') : ''}`}${c.script ? ' · ' + c.script.id : ''}</span>
      ${c.script && (c.script.note || c.script.shot) ? html`<span class="sub" style="font-size: 12px">${[c.script.shot, c.script.note].filter(Boolean).join(' · ')}${c.script.gone ? ' · ' + t('pr.gone') : ''}</span>` : ''}
      ${c.genOnly ? html`<span class="sub" style="font-size: 12px">${t('sq.genOnly', { list: cuts.filter((x) => x.src === c.id).map((x) => cuts.indexOf(x) + 1).join(' · ') || '—' })}</span>` : editLine(c, k)}
      ${c.prodErr && c.prodErr.key === C.key(c) ? html`<span class="sub" style="color: var(--warn-fg)">${c.prodErr.msg}</span>` : ''}
      ${prob ? html`<span class="sub" style="color: var(--warn-fg)">${prob}</span>` : ''}</div>
    <div class="col" style="flex: 0 0 140px; gap: 6px; padding-top: 6px">
      <span class="row" style="gap: 6px; font-size: 12px; color: var(--fg3)"><span class="dot ${ROW[s][0]}"></span>${s === 'run' ? stepName + ' ' + pct + '%' : t(ROW[s][1])}</span>
      ${s === 'run' ? html`<div class="bar"><div style="width: ${pct}%"></div></div>` : ''}</div>
    <div class="row" style="gap: 2px">
      <button type="button" class="ib m" data-act="playFrom" data-k="${k}" aria-label="${t('sq.playCut', { k: k + 1 })}" title="${t('sq.playCut', { k: k + 1 })}" ${r ? '' : 'disabled'}>${icon('play', 13)}</button>
      <button type="button" class="ib m" data-act="redoCut" data-k="${k}" aria-label="${t('sq.redoCut', { k: k + 1 })}" title="${t('sq.redoCutTip')}" ${s === 'run' || prob ? 'disabled' : ''}>${icon('redo', 14)}</button>
      <a class="ib m" href="#/video/one?cut=${c.id}&seq=${sq.id}" aria-label="${t('sq.openCut', { k: k + 1 })}" title="${t('sq.openCut', { k: k + 1 })}">${icon('open', 14)}</a>
      <button type="button" class="ib m" data-act="upCut" data-k="${k}" aria-label="${t('sq.moveUp')}" title="${t('sq.moveUp')}" ${k ? '' : 'disabled'}>${icon('up', 14)}</button>
      <button type="button" class="ib m" data-act="delCut" data-k="${k}" aria-label="${t('sq.delCut')}" title="${t('sq.delCut')}" ${s === 'run' ? 'disabled' : ''}>${icon('trash', 14)}</button></div>
    ${ui.askDel === k ? html`<div class="confirm" role="alertdialog" style="flex-basis: 100%"><span>${t('sq.askDel', { k: k + 1 })}</span><div class="row" style="gap: 8px">
      <button type="button" class="btn ghost" data-act="noDel">${t('common.cancel')}</button><button type="button" class="btn p" data-act="yesDel" data-k="${k}">${t('lib.delete')}</button></div></div>` : ''}
    ${ui.frameFor === k ? framePicker(k) : ''}</div>`;
}
// editing (design §12 A): the used span of the generation, where the sound comes from, another cut's generation, break
function editLine(c, k) {
  const [a, b] = C.useOf(c), max = Number(genCut(c).sec) || 0;
  const own = cuts.map((x, i) => [x, i]).filter(([x, i]) => i !== k && !x.src && x.id);
  const sv = c.sound === 'none' ? 'none' : c.sound && c.sound.cut ? c.sound.cut : '';
  const sel = 'min-height: 28px; width: auto; font-size: 12px; padding: 2px 6px';
  return html`<div class="row" style="gap: 4px 8px; font-size: 12px; color: var(--fg3)">
    <span>${t('sq.use')}</span>
    <input class="in mono" type="number" step="0.1" min="0" max="${max}" data-cut="${k}" data-f="useA" value="${a}" aria-label="${t('sq.useA')}" style="width: 62px; min-height: 28px; font-size: 12px">
    <span>–</span><input class="in mono" type="number" step="0.1" min="0" max="${max}" data-cut="${k}" data-f="useB" value="${b}" aria-label="${t('sq.useB')}" style="width: 62px; min-height: 28px; font-size: 12px">
    <span class="mono">= ${(b - a).toFixed(1)}${t('vg.secUnit')}</span>
    <select class="in" data-cut="${k}" data-f="sound" aria-label="${t('sq.sound')}" title="${t('sq.sound')}" style="${sel}">
      <option value="">${t('sq.soundSelf')}</option><option value="none" ${sv === 'none' ? 'selected' : ''}>${t('sq.soundNone')}</option>
      ${own.map(([x, i]) => html`<option value="${x.id}" ${sv === x.id ? 'selected' : ''}>${t('sq.soundFrom', { k: i + 1 })}</option>`)}</select>
    ${c.sound && c.sound.cut ? html`<input class="in mono" type="number" step="0.1" min="0" data-cut="${k}" data-f="soundAt" value="${c.sound.at}" aria-label="${t('sq.soundAt')}" title="${t('sq.soundAt')}" style="width: 62px; min-height: 28px; font-size: 12px">` : ''}
    <select class="in" data-cut="${k}" data-f="src" aria-label="${t('sq.src')}" title="${t('sq.srcTip')}" style="${sel}">
      <option value="">${t('sq.srcSelf')}</option>
      ${own.filter(([x]) => x.id !== c.id).map(([x, i]) => html`<option value="${x.id}" ${c.src === x.id ? 'selected' : ''}>${t('sq.srcFrom', { k: i + 1 })}</option>`)}</select>
    ${c.src && genCut(c).genOnly ? html`<span class="row" style="gap: 4px; flex-wrap: nowrap" title="${t('sq.shotTip')}">${t('sq.shot')}<input class="in mono" type="number" step="1" min="1" data-cut="${k}" data-f="shot" value="${c.shot || ''}" aria-label="${t('sq.shot')}" style="width: 48px; min-height: 28px; font-size: 12px"></span>` : ''}
    <button type="button" class="chip" data-act="brk" data-k="${k}" aria-pressed="${!!c.brk}" title="${t('sq.brkTip')}" style="min-height: 28px; ${c.brk ? 'border-color: var(--warn-line); color: var(--warn-fg)' : ''}">▷ ${t('sq.brk')}</button>
    <button type="button" class="chip" data-act="textFor" data-k="${k}" aria-expanded="${ui.textFor === k}" title="${t('sq.textTip')}" style="min-height: 28px">${(c.texts || []).length ? t('sq.textN', { n: c.texts.length }) : t('sq.text')}</button>
    <button type="button" class="chip" data-act="fade" data-k="${k}" aria-pressed="${!!c.fade}" title="${t('sq.fadeTip')}" style="min-height: 28px">◐ ${t('sq.fade')}</button>
    ${c.fade ? html`<span class="row" style="gap: 4px; flex-wrap: nowrap">${t('sq.fadeIn')}<input class="in mono" type="number" step="0.1" min="0" data-cut="${k}" data-f="fadeIn" value="${c.fade.in}" aria-label="${t('sq.fadeIn')}" style="width: 56px; min-height: 28px; font-size: 12px"></span>
      <span class="row" style="gap: 4px; flex-wrap: nowrap">${t('sq.fadeOut')}<input class="in mono" type="number" step="0.1" min="0" data-cut="${k}" data-f="fadeOut" value="${c.fade.out}" aria-label="${t('sq.fadeOut')}" style="width: 56px; min-height: 28px; font-size: 12px"></span>` : ''}</div>
    ${ui.textFor === k ? html`<div class="col" style="gap: 4px"><textarea class="in mono" rows="${Math.max(2, (c.texts || []).length + 1)}" data-cut="${k}" data-f="texts" placeholder="${t('sq.textPh')}" style="font-size: 12px">${(c.texts || []).map((x) => x.src || '').join('\n')}</textarea>
      <span class="sub">${t('sq.textHelp')}</span></div>` : ''}`;
}
// 첫 프레임 (기능 16): the script's first-frame pictures — state, candidates (click = ●), again with a new seed
const FST = { done: ['ok', 'ch.st.done'], pick: ['warn', 'ch.st.pick'], run: ['busy', 'pr.st.run'], failed: ['bad', 'ch.st.failed'], wait: ['', 'pr.st.wait'], todo: ['', 'ch.st.wait'] };
const TYPE = { compose: 'pr.t.compose', plate: 'pr.t.plate', box: 'pr.t.box', t2i: 'pr.t.t2i', edit: 'pr.t.edit', file: 'pr.t.file' };
function framesCard() {
  const fs = sq.frames, n = fs.filter((f) => P.frameState(sq, f) === 'done').length;
  return html`<section class="card">
    <div class="row"><span class="num">0</span><h2 class="h2">${t('pr.frames')}</h2><span class="sub">${t('pr.framesMeta', { n, all: fs.length })} · ${sq.common.mode === 'step' ? t('pr.stepNote') : t('pr.autoNote')}</span></div>
    <div style="margin-top: 12px; display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px">${fs.map((f, i) => {
      const s = P.frameState(sq, f), users = cuts.filter((c) => c.script && (c.script.frame === f.id || c.script.last === f.id || (c.script.keys || []).some((x) => x.frame === f.id))).map((c) => c.script.id);
      const cands = (f.jobs || []).filter((x) => x.key === P.specKey(f)).flatMap((x) => { const j = C.jobOf(x.job); return j ? j.files.filter((y) => /^\d+_\d+\.png$/.test(y.name) && y.synced).map((y) => ({ j, name: y.name })) : []; });
      const portrait = (f.spec.orient || 'v') === 'v';
      return html`<div class="col" style="gap: 6px; padding: 8px; border: 1px solid ${s === 'pick' ? 'var(--warn-line)' : 'var(--line)'}; border-radius: 10px">
        <div class="row" style="justify-content: space-between; gap: 4px"><span class="mono" style="font-size: 12px">${f.id}</span><span class="sub" style="font-size: 11px">${t(TYPE[f.spec.type])}</span></div>
        ${f.rel ? html`<a class="ph" href="${filesUrl(f.rel)}" target="_blank" style="display: block; aspect-ratio: ${portrait ? '9 / 16' : '16 / 9'}"><img src="${filesUrl(f.rel)}" alt="${f.id}" loading="lazy"></a>` :
          html`<div class="ph ${s === 'run' ? 'sk' : ''}" style="aspect-ratio: ${portrait ? '9 / 16' : '16 / 9'}"></div>`}
        ${cands.length && (s === 'pick' || cands.length > 1) ? html`<div class="row" style="gap: 4px">${cands.map((c) => html`
          <button type="button" class="ph" data-act="pickFrame" data-i="${i}" data-job="${c.j.id}" data-name="${c.name}" aria-pressed="${!!(f.base && f.base.job === c.j.id && f.base.name === c.name)}" title="${t('pr.pick')}"
            style="padding: 0; width: 40px; aspect-ratio: ${portrait ? '9 / 16' : '16 / 9'}; cursor: pointer; ${f.base && f.base.job === c.j.id && f.base.name === c.name ? 'outline: 2px solid var(--blue)' : ''}"><img src="${jobFileUrl(c.j, c.name)}" alt="" loading="lazy"></button>`)}</div>` : ''}
        <div class="row" style="justify-content: space-between; gap: 4px">
          <span class="row" style="gap: 6px; font-size: 12px; color: var(--fg3)"><span class="dot ${FST[s][0]}"></span>${t(FST[s][1])}</span>
          <button type="button" class="ib m" data-act="redoFrame" data-i="${i}" aria-label="${t('pr.redo')}" title="${t('pr.redo')}" ${s === 'run' || s === 'wait' || !prodOn() || f.spec.type === 'file' ? 'disabled' : ''}>${icon('redo', 13)}</button></div>
        ${f.err && f.err.key === P.specKey(f) ? html`<span class="sub" style="color: var(--warn-fg); font-size: 11px">${f.err.msg}</span>` : ''}
        <span class="sub mono" style="font-size: 10px">${users.join(' ')}</span></div>`;
    })}</div></section>`;
}
function framePicker(k) {
  return html`<div class="col" style="flex-basis: 100%; gap: 8px; padding: 10px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
    <div class="row" style="justify-content: space-between"><span class="lbl">${t('vg.pickFor', { what: t('vg.first') })}</span>
      <div class="row" style="gap: 4px">${cuts[k].first ? html`<button type="button" class="btn ghost" data-act="clearFrame" data-k="${k}" style="min-height: 32px">${t('vg.remove')}</button>` : ''}
      <button type="button" class="ib m" data-act="frameFor" data-k="${k}" aria-label="${t('common.cancel')}">${icon('x', 14)}</button></div></div>
    ${scenes.length ? html`<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 8px">${scenes.map((i) => html`
      <button type="button" class="ph" data-act="setFrame" data-k="${k}" data-rel="${i.path}" data-name="${i.name}" title="${i.name}" style="padding: 0; aspect-ratio: 1; cursor: pointer">
        <img src="${filesUrl(i.path)}" alt="${i.name}" loading="lazy"></button>`)}</div>` : html`<span class="sub">${t('vg.noFrames')}</span>`}
    <span class="sub">${t('sq.frameMore')}</span></div>`;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const states = cuts.map((c) => C.cutState(genCut(c)));
  const cnt = (x) => states.filter((s) => s === x).length;
  const old = cnt('old');
  const rs = results();
  const ex = sq.export && C.jobOf(sq.export.job);
  const exFiles = ex ? exportFiles(ex).filter((f) => f.synced) : [];
  const exDone = ex && ex.state === 'done' && exFiles.length && exFiles.length === exportFiles(ex).length;
  const exOld = sq.export && sq.export.key !== exportKey();
  const total = rs.reduce((a, r, k) => a + (r && !cuts[k].genOnly ? C.lenOf(cuts[k]) : 0), 0);
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px" aria-label="${t('ch.progress')}">
        ${[['ok', 'done', 'sq.doneN'], ['busy', 'run', 'sq.runN'], ['warn', 'pick', 'sq.pickN'], ['bad', 'failed', 'sq.failN'], ['', 'stopped', 'sq.stopN']]
          .filter(([, s], i) => i === 0 || cnt(s)).map(([d, s, key]) => cuts.length ? [d, t(key, { n: cnt(s), all: cuts.length })] : ['', t('sq.noCuts')]).map(([d, s]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px"><span class="dot ${d}"></span>${s}</li>`)}</ol>
      <div class="toprow-end">${activeJobs().length ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.askStop')}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${cnt('failed') || cnt('stopped') ? html`<div class="row banner ${cnt('failed') ? 'bad' : 'warn'}" role="status" style="justify-content: space-between"><span>${cnt('failed') ? t('sq.failedBanner', { n: cnt('failed') + cnt('stopped') }) : t('sq.stoppedBanner', { n: cnt('stopped') })}</span>
      <button type="button" class="btn p" data-act="resume" style="min-height: 34px">${icon('play', 12)}${t('jobs.resume')}</button></div>` : ''}
    ${old ? html`<div class="row banner warn" role="status" style="justify-content: space-between"><span>${t('sq.oldBanner', { n: old })}</span>
      <button type="button" class="ib p" data-act="regenOld" aria-label="${t('sq.regenOld')}" title="${t('sq.regenOld')}">${icon('redo')}</button></div>` : ''}

    ${(sq.frames || []).length ? framesCard() : ''}
    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${t('sq.cuts')}</h2><span class="sub">${t('sq.cutsNote')}</span><div class="grow"></div>
        <button type="button" class="btn ghost" data-act="scenes" aria-expanded="${ui.scenes}" style="min-height: 34px">${icon('down', 14)}${t('sq.fromScenes')}</button>
        <button type="button" class="ib" data-act="addCut" aria-label="${t('sq.addCut')}" title="${t('sq.addCut')}">${icon('plus')}</button></div>
      ${ui.scenes ? html`<div class="col" style="margin-top: 12px; gap: 8px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
        <span class="lbl">${t('sq.scenesNote')}</span>
        ${scenes.length ? html`<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px">${scenes.map((i) => html`
          <button type="button" class="ph" data-act="pickScene" data-rel="${i.path}" aria-pressed="${ui.picked.has(i.path)}" title="${i.name}"
            style="padding: 0; aspect-ratio: 1; cursor: pointer; ${ui.picked.has(i.path) ? 'outline: 2px solid var(--blue); outline-offset: -2px' : ''}"><img src="${filesUrl(i.path)}" alt="${i.name}" loading="lazy"></button>`)}</div>
          <div class="row" style="justify-content: flex-end"><button type="button" class="btn p" data-act="importScenes" ${ui.picked.size ? '' : 'disabled'}>${t('sq.importN', { n: ui.picked.size })}</button></div>` :
          html`<span class="sub">${t('sq.noScenes')}</span>`}</div>` : ''}
      ${cuts.length ? html`<div class="list" style="margin-top: 14px">${cuts.map(cutRow)}</div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${t('sq.empty')}</p>`}
    </section>

    <section class="card">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('sq.watch')}</h2><span class="sub mono">${t('sq.watchMeta', { n: rs.filter(Boolean).length, all: cuts.length, s: total.toFixed(1) })}</span><div class="grow"></div>
        <button type="button" class="ib" data-act="playAll" aria-expanded="${ui.play}" aria-label="${t('sq.playAll')}" title="${t('sq.playAll')}" ${rs.some(Boolean) ? '' : 'disabled'}>${icon('play', 14)}</button></div>
      <div style="margin-top: 12px; display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px">${cuts.map((c, k) => html`
        <div style="flex: ${C.lenOf(c)} 0 0; min-width: 56px; display: flex; flex-direction: column; gap: 4px">
          <div style="height: 40px; border: 1px solid var(--line); border-radius: 6px; background: ${rs[k] ? '#2C2C33' : '#1B1B1F'}"></div>
          <span class="mono" style="font-size: 11px; color: var(--fg4)">${k + 1} · ${t('vg.secN', { n: C.lenOf(c).toFixed(1) })}${c.fade ? ' ◐' : ''}${c.brk ? ' ▷' : ''}</span></div>`)}</div>
      ${ui.play ? html`<div class="ph" style="margin-top: 12px; aspect-ratio: ${sq.common.orient === 'h' ? '16 / 9' : '9 / 16'}; max-height: 460px"><video data-playlist controls autoplay style="object-fit: contain; background: #000"></video></div>
        <span class="sub">${t('sq.playNote')}</span>` : ''}
    </section>

    <section class="card">
      <div class="row"><span class="num">3</span><h2 class="h2">${t('sq.export')}</h2><span class="sub">${t('sq.exportNote')}</span><div class="grow"></div>
        <span class="sub mono">${ex ? (ex.state === 'failed' ? failWord(isStopped(ex)) : exDone ? t('ch.st.done') : Math.round(ex.progress * 100) + '%') : ''}</span></div>
      ${exDone && !exOld ? html`<div class="row" style="margin-top: 12px; gap: 12px; align-items: flex-start">${exFiles.map((f, i) => html`
        <div class="col" style="gap: 4px; flex: 0 1 ${sq.common.orient === 'h' ? '460px' : '260px'}">
          <div class="ph" style="aspect-ratio: ${sq.common.orient === 'h' ? '16 / 9' : '9 / 16'}"><video src="${jobFileUrl(ex, f.name)}?v=${f.sha.slice(0, 8)}" controls preload="metadata" style="object-fit: contain; background: #000"></video></div>
          <span class="sub" style="text-align: center">${exFiles.length > 1 ? t('sq.partN', { n: i + 1 }) + ' · ' : ''}${sq.export.saved ? t('sq.savedAs', { name: sq.export.saved[i] || sq.export.saved }) : t('jobs.arriving')}${(sq.export.parts || [])[i] && sq.export.parts[i].brk ? ' · ▷' : ''}</span></div>`)}</div>` : ''}
      <div class="row" style="margin-top: 12px; justify-content: space-between">
        <span class="sub" style="color: ${exOld ? 'var(--warn-fg)' : 'var(--fg4)'}">${exOld ? t('sq.exportOld') : rs.some((r) => !r) && rs.some(Boolean) ? t('sq.exportSkip', { n: rs.filter((r) => !r).length }) : ''}</span>
        <button type="button" class="btn p" data-act="export" ${rs.some(Boolean) && !(ex && (ex.state === 'pending' || ex.state === 'running')) && (!sq.export || exOld || ex && ex.state === 'failed') ? '' : 'disabled'}>${t('sq.exportGo')}</button></div>
    </section>

    ${cuts.length ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('sq.askClose')}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button><button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between; padding-top: 4px"><p class="sub" style="margin: 0">${t('sq.closeNote')}</p>
        <button type="button" class="btn ghost" data-act="close" ${activeJobs().length ? 'disabled' : ''}>${t('jobs.close')}</button></div>` : ''}`.__raw;
  renderActions();
  const pl = el.querySelector('[data-playlist]');
  if (pl) {   // the used span of each cut (its own picture; borrowed sound is heard only in the export)
    const list = rs.map((r, k) => k >= ui.from && r && !cuts[k].genOnly ? [C.urlOf(r), ...C.useOf(cuts[k])] : null).filter(Boolean);
    let i = 0;
    const go = () => { const [u, a, b] = list[i]; pl.src = `${u}#t=${a},${b}`; pl.dataset.end = b; pl.play().catch(() => {}); };
    const next = () => { i += 1; if (i < list.length) go(); };
    if (list.length) go();
    pl.ontimeupdate = () => { if (pl.currentTime >= Number(pl.dataset.end) - 0.02 && !pl.paused) next(); };
    pl.onended = next;
  }
}

function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(VIDEO_MENU(), '#/video/seq')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const d = a.dataset, k = Number(d.k);
  switch (d.act) {
    case 'set': return setCommon(d.k, d.k === 'vsa' ? d.v === '1' : ['cands', 'steps'].includes(d.k) ? Number(d.v) : d.v);
    case 'delCode': return setCommon('codes', sq.common.codes.filter((c) => c !== d.code));
    case 'addCut': return addCuts([{}]);
    case 'scenes': ui.scenes = !ui.scenes; return renderOut();
    case 'pickScene': ui.picked.has(d.rel) ? ui.picked.delete(d.rel) : ui.picked.add(d.rel); return renderOut();
    case 'importScenes': {
      // the scene image's own prompt is an image edit text, never a video prompt (session #14): cuts start empty
      const list = scenes.filter((i) => ui.picked.has(i.path)).map((i) => ({ first: { rel: i.path, name: i.name }, prompt: '' }));
      ui.picked = new Set(); ui.scenes = false;
      return addCuts(list);
    }
    case 'frameFor': ui.frameFor = ui.frameFor === k ? null : k; return renderOut();
    case 'setFrame': cuts[k].first = { rel: d.rel, name: d.name }; ui.frameFor = null; await saveCut(cuts[k]); return renderAll();
    case 'clearFrame': cuts[k].first = null; ui.frameFor = null; await saveCut(cuts[k]); return renderAll();
    case 'genAll': return generate(todo());
    case 'regenOld': return generate(cuts.filter((c) => C.cutState(c) === 'old' && !cutProblem(c)));
    case 'redoCut': cuts[k].base = null; return generate([cuts[k]]);
    case 'upCut': [cuts[k - 1], cuts[k]] = [cuts[k], cuts[k - 1]]; sq.cuts = cuts.map((c) => c.id); await saveSeq(); return renderAll();
    case 'delCut': ui.askDel = k; return renderOut();
    case 'noDel': ui.askDel = null; return renderOut();
    case 'yesDel': {
      const c = cuts[k];
      ui.askDel = null;
      if (!(await act(() => api('/api/finish', { items: C.saveItems(c, `${(sq.name || 'sequence').replace(/\s+/g, '_')}_${k + 1}`, false), jobs: C.cutJobs(c), docs: [{ kind: 'cuts', id: c.id }] })))) return;
      cuts.splice(k, 1); sq.cuts = cuts.map((x) => x.id); await saveSeq(); return renderAll();
    }
    case 'playFrom': ui.play = true; ui.from = k; return renderOut();
    case 'playAll': ui.play = !ui.play; ui.from = 0; return renderOut();
    case 'export': return exportSeq();
    case 'brk': cuts[k].brk = !cuts[k].brk; await saveCut(cuts[k]); return renderOut();
    case 'fade': cuts[k].fade = cuts[k].fade ? null : { in: 0, out: Math.min(1, C.lenOf(cuts[k])) }; await saveCut(cuts[k]); return renderOut();
    case 'textFor': ui.textFor = ui.textFor === k ? null : k; return renderOut();
    case 'prodStart': sq.prod = Object.assign({}, sq.prod, { started: true, done: false }); await saveSeq(); return prodRound();
    case 'pickFrame': {
      const f = sq.frames[Number(d.i)];
      if (!(await act(() => P.pickFrame(sq, f, d.job, d.name)))) return;
      await saveSeq();
      return prodRound();
    }
    case 'redoFrame': {   // another candidate with the next seed; ● stays until a new one is picked
      const f = sq.frames[Number(d.i)];
      f.err = null;
      if (!(await act(() => P.submitFrame(sq, f)))) return;
      await saveSeq();
      await poll();
      return renderAll();
    }
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop': ui.askStop = false; for (const j of activeJobs()) await act(() => srv(`/jobs/${j.id}/cancel`, {})); await poll(); return;
    case 'resume': for (const j of allJobs().map(C.jobOf).filter((x) => x && x.state === 'failed')) await act(() => api('/api/retry', { job: j.id })); await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; return closeSeq();
    default:
  }
}
function onInput(e) {
  const dd = e.target.dataset;
  if (dd.cut !== undefined) {
    const c = cuts[Number(dd.cut)];
    c.prompt = e.target.value;
    if (c.id) saveCutSoon(c);
    renderSet();
    return;
  }
  if (dd.k === 'name') {
    sq.name = e.target.value;
    const h = root.querySelector('[data-set] h1');
    if (h) h.firstChild.textContent = (sq.name || t('sq.new')) + ' ';
    clearTimeout(onInput.t);
    onInput.t = setTimeout(saveSeq, 800);
  }
}
async function onChange(e) {
  const dd = e.target.dataset;
  if (dd.cut !== undefined && dd.f && dd.f !== 'prompt') {   // the editing line (design §12 A)
    const c = cuts[Number(dd.cut)], v = e.target.value;
    const max = Number(genCut(c).sec) || 0;
    if (dd.f === 'useA' || dd.f === 'useB') {
      const u = C.useOf(c), x = Math.min(max, Math.max(0, Math.round(Number(v) * 10) / 10));
      if (dd.f === 'useA') u[0] = Math.min(x, u[1] - 0.1); else u[1] = Math.max(x, u[0] + 0.1);
      c.use = u[0] === 0 && u[1] === max ? null : u;
    } else if (dd.f === 'sound') c.sound = v === '' ? null : v === 'none' ? 'none' : { cut: v, at: c.sound && c.sound.at || 0 };
    else if (dd.f === 'soundAt') c.sound = Object.assign({}, c.sound, { at: Math.max(0, Math.round(Number(v) * 10) / 10) });
    else if (dd.f === 'src') { c.src = v || null; c.use = null; c.shot = null; }
    else if (dd.f === 'shot') c.shot = Math.round(Number(v)) >= 1 ? Math.round(Number(v)) : null;
    else if (dd.f === 'fadeIn' || dd.f === 'fadeOut') {   // seconds from / to black; both 0 = no fade
      const x = Math.min(C.lenOf(c), Math.max(0, Math.round(Number(v) * 10) / 10) || 0);
      c.fade = Object.assign({ in: 0, out: 0 }, c.fade, { [dd.f === 'fadeIn' ? 'in' : 'out']: x });
      if (!c.fade.in && !c.fade.out) c.fade = null;
    }
    else if (dd.f === 'texts') {   // checked by the Mac with the script rules (docs/script.md §5)
      const r = await act(() => api('/api/texts', { lines: v.split('\n') }));
      if (!r) return;
      c.texts = r.texts;
    }
    saveCutSoon(c);
    renderOut();
    return;
  }
  if (dd.add === 'code' && e.target.value) return setCommon('codes', sq.common.codes.concat([e.target.value]).slice(0, 2));
  if (dd.flag) return setCommon(dd.flag, e.target.checked);
  if (dd.k === 'sec') { const v = C.secClamp(sq.common.model, e.target.value); return v != null ? setCommon('sec', v) : renderSet(); }
  if (dd.k === 'refSize') return setCommon('refSize', e.target.value);
}

// one production round on the open sequence (prod.js), saving what changed
async function prodRound(draw = true) {
  if (busy) return;
  if (drive) {
    busy = true;
    try {
      for (const c of cuts) await C.loadFaces(c);
      const r = await act(() => P.advance(sq, cuts));
      if (r) {
        for (const c of r.cuts) await saveCut(c);
        if (r.sq) await saveSeq();
      }
    } finally {
      busy = false;
    }
  }
  if (draw) { await poll(); renderAll(); }
}
async function load(id) {
  sq = freshSeq(); cuts = []; ui = freshUi();
  if (S.openSeq && S.openSeq !== id) P.releaseSeq(S.openSeq);
  await P.withRun(id, async () => {   // after a background round on it, then held by this screen (prod.js)
    if (id) {
      const d = await act(() => api('/api/doc?kind=sequences&id=' + encodeURIComponent(id)));
      if (d && d.id) sq = Object.assign(freshSeq(), d, { common: Object.assign(freshSeq().common, d.common) });
      const all = (sq.cuts || []).length ? await act(() => api('/api/docs?kind=cuts&ids=' + encodeURIComponent(sq.cuts.join(',')))) || [] : [];
      cuts = sq.cuts.map((cid) => all.find((c) => c && c.id === cid)).filter(Boolean).map((c) => C.freshCut(c));
    }
    S.openSeq = sq.id;
    drive = await P.holdSeq(sq.id);
  });
  S.watch = new Set(allJobs());
  await fetchJobs([...S.watch]);
}

export default {
  async mount(el, { query }) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    chars = ((await act(() => api('/api/library?scope=shared'))) || {}).characters || [];
    scenes = (((await act(() => api('/api/library?scope=project'))) || {}).items || []).filter((i) => i.kind === 'scenes');
    const id = query.get('id') || (sq && sq.id) || null;
    await load(id);
    if (id && !sq.id) history.replaceState(null, '', '#/video/seq');
    seqs = (await act(() => api('/api/docs?kind=sequences')) || []).filter((d) => d && d.id !== sq.id);
    for (const c of cuts) await C.loadFaces(c);
    renderAll();
  },
  async update() {
    if (!root || !sq) return;
    allJobs().forEach((id) => S.watch.add(id));
    if (prodOn()) await prodRound(false);
    for (const c of sq.script ? [] : cuts) {
      await C.loadFaces(c);
      let changed = C.autoBase(c);
      if (c.mode === 'all' && c.base && !busy) {
        busy = true;
        changed = !!(await act(() => C.nextStep(c, label(c)))) || changed;
        busy = false;
      }
      if (changed) await saveCut(c);
    }
    const ex = sq.export && C.jobOf(sq.export.job);
    if (ex && !sq.export.saved && exportFiles(ex).length && ex.state === 'done' && exportFiles(ex).every((f) => f.synced)) {   // the parts go to the library once
      const title = (sq.name || 'sequence').replace(/\s+/g, '_'), fs = exportFiles(ex), saved = [];
      for (let i = 0; i < fs.length; i++) {
        const r = await act(() => api('/api/save', { job: ex.id, name: fs[i].name, title: fs.length > 1 ? `${title}_p${i + 1}` : title }));
        if (!r) return;
        saved.push(r.saved);
      }
      sq.export.saved = saved;
      await saveSeq();
    }
    if (root.contains(document.activeElement) && document.activeElement.matches('textarea, input, select')) return;   // typing
    if (playing()) return;
    renderAll();
  },
  unmount() { root = null; P.releaseSeq(S.openSeq); S.openSeq = null; S.watch.clear(); },
};
