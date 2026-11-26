// 영상 생성 › 한 컷 (design §11): settings left (입력 · 인물 · 모델 · 길이 · 마무리, all chosen before 시작, never
// folds), results right (생성 → 다듬기 → 업스케일) filled in place. The cut lives on the Mac in
// projects/<p>/work/cuts/<id>.json until 보관함에 저장 / 닫기; #/video/one?cut=<id>[&seq=<id>] reopens it (a sequence's cut).
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, fmtDur, enter, codeLabel, codeLabelOf, failWord, failColor } from '../app.js';
import { MODELS } from '../models.js';
import { sideMenu, VIDEO_MENU } from './wip.js';
import * as C from './cut.js';

let root = null, st = null, seq = null, chars = [], locs = [], scenes = [], open = [], ui = {}, busy = false;
const freshUi = () => ({ adv: false, pick: null, askClose: false, askStop: false, uploading: false, autoErr: false });

// ---- derived ----
const started = () => st.cand.length > 0;
const ratio = () => st.orient === 'h' ? '16 / 9' : '9 / 16';
const label = () => seq ? `${seq.name || t('sq.untitled')} · ${t('sq.cutN', { k: seq.cuts.indexOf(st.id) + 1 })}` : st.title.trim() || t('vg.cut');
const anyActive = () => C.cutJobs(st).map(C.jobOf).some((j) => j && (j.state === 'pending' || j.state === 'running'));
const playing = () => [...root.querySelectorAll('[data-out] video')].some((v) => !v.paused && !v.ended);

function problems() {
  const out = [];
  if (!(S.status && S.status.settings && S.status.settings.project)) out.push(t('vg.needProject'));   // cuts and drafts live in the project
  if (!st.prompt.trim()) out.push(t('vg.needPrompt'));
  const why = C.whyNot(st, st.model, chars);
  if (why) out.push(t('vg.why.' + why));
  return out;
}

// ---- persistence ----
async function save() {
  if (!st.id && !started()) return;
  const r = await act(() => api('/api/doc', { kind: 'cuts', id: st.id, doc: st }));
  if (r && !st.id) {
    st.id = r.id;
    history.replaceState(null, '', '#/video/one?cut=' + r.id + (seq ? '&seq=' + seq.id : ''));
  }
}
let saveTimer = 0;
const saveSoon = () => { clearTimeout(saveTimer); if (st.id) saveTimer = setTimeout(save, 800); };
// in a sequence, a common setting changed here stays this cut's own
const own = (k) => { if (seq && C.COMMON.includes(k) && !st.over.includes(k)) st.over.push(k); };

// ---- actions ----
async function generate(n) {
  const pr = problems();
  if (pr.length) return toast(pr[0], true);
  ui.autoErr = false;
  if (await act(() => C.submitCands(st, n || st.cands, label()))) {
    await save();
    await poll();
  }
  renderAll();
}
// the cut follows its first frame's shape (or the last frame's when there is no first): a landscape scene → 가로
async function followFrame(slot) {
  const f = st[slot];
  if (!f || (slot === 'last' && st.first)) return;
  const img = new Image();
  img.src = filesUrl(f.rel);
  try { await img.decode(); } catch (e) { return; }
  if (img.naturalWidth === img.naturalHeight) return;
  const o = img.naturalWidth > img.naturalHeight ? 'h' : 'v';
  if (o === st.orient) return;
  st.orient = o; own('orient');
  toast(t('vg.orientAuto', { o: t(o === 'h' ? 'vg.landscape' : 'vg.portrait') }));
  renderSet(); renderOut(); saveSoon();
}
async function upload(slot, f) {
  ui.uploading = true; renderSet();
  const r = await act(async () => {
    const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.status);
    return j;
  });
  ui.uploading = false;
  if (r) { st[slot] = { rel: r.rel, name: f.name }; ui.pick = null; saveSoon(); }
  renderAll();
  if (r) followFrame(slot);
}
async function step() {   // 다듬기 / 업스케일 for the ● base (단계마다 확인, or again after a new ●)
  ui.autoErr = false;
  if (await act(() => C.nextStep(st, label()))) { await save(); await poll(); }
  renderOut();
}
async function finish(saveIt) {
  const title = (seq ? label() : st.title.trim() || st.prompt.trim().slice(0, 24) || 'cut').replace(/\s+/g, '_');
  const r = await act(() => api('/api/finish', { items: C.saveItems(st, title, saveIt), jobs: C.cutJobs(st),
    docs: st.id ? [{ kind: 'cuts', id: st.id }] : [] }));
  if (!r) return;
  toast(r.saved.length ? t('vg.savedN', { n: r.saved.length }) : t('vg.closed'));
  st = C.freshCut(); ui = freshUi();
  history.replaceState(null, '', '#/video/one');
  open = (await act(() => api('/api/docs?kind=cuts')) || []).filter((d) => !d.seq);
  await poll();
  renderAll();
}

// ---- left: settings ----
function seg(k, opts, cur, labelText, dis) {
  return html`<div class="seg" role="group" aria-label="${labelText}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${k}" data-v="${v}" aria-pressed="${String(cur) === String(v)}" ${dis ? 'disabled' : ''}>${l}</button>`)}</div>`;
}
function slot(which) {
  const f = st[which];
  return html`<div class="field"><span>${t('vg.' + which)} <span class="sub">· ${t('vg.optional')}</span></span>
    ${f ? html`<div class="ph" style="aspect-ratio: ${ratio()}; max-height: 180px">
        <img src="${filesUrl(f.rel)}" alt="${f.name}" style="object-fit: contain">
        <button type="button" class="ib" data-act="clearFrame" data-slot="${which}" aria-label="${t('vg.remove')}" title="${t('vg.remove')}"
          style="position: absolute; right: 4px; top: 4px; width: 28px; height: 28px; background: #16161AE6">${icon('x', 13)}</button></div>` :
      html`<button type="button" class="slot" data-act="pick" data-slot="${which}" aria-expanded="${ui.pick === which}" style="aspect-ratio: ${ratio()}; max-height: 180px">
        ${icon('plus', 18)}<span>${ui.uploading && ui.pick === which ? t('common.loading') : t('vg.fromLib')}</span></button>`}</div>`;
}
function picker() {
  if (!ui.pick) return '';
  const items = scenes.map((i) => ({ rel: i.path, name: i.name })).concat(locs.filter((l) => l.master).map((l) => ({ rel: `locations/${l.code}/${l.master}`, name: l.code })));
  return html`<div class="col" style="gap: 10px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
    <div class="row" style="justify-content: space-between"><span class="lbl">${t('vg.pickFor', { what: t('vg.' + ui.pick) })}</span>
      <button type="button" class="ib m" data-act="unpick" aria-label="${t('common.cancel')}" title="${t('common.cancel')}">${icon('x', 14)}</button></div>
    <label class="btn ghost" style="align-self: flex-start; cursor: pointer">${icon('up', 14)}${t('vg.upload')}<input type="file" accept="image/*" data-file="${ui.pick}" hidden></label>
    ${items.length ? html`<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 8px">${items.map((x) => html`
      <button type="button" class="ph" data-act="frame" data-rel="${x.rel}" data-name="${x.name}" title="${x.name}" style="padding: 0; aspect-ratio: 1; cursor: pointer">
        <img src="${filesUrl(x.rel)}" alt="${x.name}" loading="lazy"></button>`)}</div>` : html`<span class="sub">${t('vg.noFrames')}</span>`}</div>`;
}
function modelCard(m) {
  const why = C.whyNot(st, m, chars);
  const on = st.model === m;
  // always pickable (the model can come before its inputs); a misfit shows its reason here and blocks 시작 (problems)
  return html`<button type="button" class="mcard" role="radio" aria-checked="${on}" data-act="model" data-m="${m}">
    <span style="font-weight: 600">${MODELS[m].name}</span>
    <span class="mcard-line" style="font-size: 12px; color: ${why ? 'var(--warn-fg)' : 'var(--fg3)'}" title="${why ? t('vg.why.' + why) : t('vg.note.' + m)}">${why ? '! ' + t('vg.why.' + why) : t('vg.note.' + m)}</span>
    <span class="mono" style="font-size: 11px; color: var(--fg4)">${t('vg.speed.' + m)}</span></button>`;
}
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const addable = chars.filter((c) => c.master && !st.codes.includes(c.code));
  const [w, h] = C.SIZE[st.orient];
  const faceOk = st.model !== 'h3';
  el.innerHTML = html`
    <div>
      ${seq ? html`<a class="row lbl" href="#/video/seq?id=${seq.id}" style="gap: 6px; text-decoration: none; color: var(--blue)">← ${t('sq.back', { name: seq.name || t('sq.untitled') })}</a>` : ''}
      <div class="lbl">${seq ? t('sq.crumbCut') : t('vg.crumb')}</div>
      <h1 class="h1">${seq ? t('sq.cutN', { k: seq.cuts.indexOf(st.id) + 1 }) + (st.title ? ' · ' + st.title : '') : st.title || t('vg.newCut')}</h1><p class="lead">${t('lead.one')}</p></div>
    ${!seq && !started() && open.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${open.map((d) => html`<a class="chip" href="#/video/one?cut=${d.id}" style="display: inline-flex; align-items: center; text-decoration: none">${d.title || d.prompt.slice(0, 14) || d.id}</a>`)}</div>` : ''}
    <div class="col" style="gap: 6px">${seg('mode', [['all', t('ch.modeAll')], ['step', t('ch.modeStep')]], st.mode, t('ch.mode'))}
      <span class="sub">${st.mode === 'all' ? t('vg.modeAllHint') : t('vg.modeStepHint')}</span></div>

    <div class="sec">
      <h2 class="h3">${t('vg.input')}</h2>
      ${seq ? '' : html`<label class="field">${t('vg.title')}<input class="in" data-k="title" value="${st.title}" placeholder="${t('vg.titlePh')}"></label>`}
      <label class="field">${t('vg.prompt')}<textarea class="in" rows="4" data-k="prompt" placeholder="${t('vg.promptPh')}">${st.prompt}</textarea></label>
      <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px">${slot('first')}${slot('last')}</div>
      ${picker()}</div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('vg.people')}</h2><span class="sub">${t('vg.peopleNote.' + (st.model === 'h3' ? 'h3' : 'face'))}</span></div>
      <div class="row" style="gap: 6px">
        ${st.codes.map((c) => html`<span class="chip" style="display: inline-flex; align-items: center; gap: 6px">${codeLabelOf(chars, c)}${C.hasSheet(chars, c) ? '' : html`<span class="sub">· ${t('sw.noSheet')}</span>`}
          <button type="button" class="ib m" data-act="delCode" data-code="${c}" aria-label="${t('vg.remove')} ${c}" title="${t('vg.remove')}" style="width: 22px; height: 22px">${icon('x', 12)}</button></span>`)}
        ${st.codes.length < 2 ? html`<select class="in" data-add="code" aria-label="${t('vg.addPerson')}" style="width: 150px; min-height: 34px">
          <option value="">${t('vg.addPerson')}</option>${addable.map((c) => html`<option value="${c.code}">${codeLabel(c)}${C.hasSheet(chars, c.code) ? '' : ' · ' + t('sw.noSheet')}</option>`)}</select>` : ''}</div></div>

    <div class="sec">
      <h2 class="h3">${t('vg.model')} <span class="sub" style="font-weight: 400">· ${t('vg.modelNote')}</span></h2>
      <div role="radiogroup" aria-label="${t('vg.model')}" style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px">
        ${C.MODEL_IDS.map(modelCard)}</div>
      <div class="row" style="gap: 10px 14px; align-items: flex-end">
        <div class="field">${t('vg.len')}${C.secInput(st.model, st.sec)}</div>
        <div class="field">${t('sw.orient')}${seg('orient', [['v', t('vg.portrait')], ['h', t('vg.landscape')]], st.orient, t('sw.orient'))}</div>
        <div class="field">${t('ch.cands')}${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}</div>
        <button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      <span class="sub mono">${w}×${h}</span>
      ${ui.adv ? html`<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('ch.seed')}<input class="in mono" data-k="seed" inputmode="numeric" value="${st.seed}"></label>
        ${st.model === 'pih3' ? html`<div class="field">${t('def.steps')}${seg('steps', [[2, t('vg.steps2')], [4, t('vg.steps4')]], st.steps, t('def.steps'))}</div>` : ''}
        ${st.model === 'pih3' ? html`<div class="field" title="${t('vg.vsaNote')}">${t('def.vsa')}${seg('vsa', [[0, t('vg.st.off')], [1, t('vg.on')]], st.vsa ? 1 : 0, t('def.vsa'))}</div>` : ''}
        ${st.model === 'h3' ? html`<label class="field">${t('def.ref_size')}<select class="in" data-k="refSize">
          <option value="max" ${st.refSize === 'max' ? 'selected' : ''}>${t('def.ref_size.max')}</option>
          <option value="match" ${st.refSize === 'match' ? 'selected' : ''}>${t('def.ref_size.match')}</option></select></label>` : ''}</div>` : ''}</div>

    <div class="sec">
      <h2 class="h3">${t('vg.finish')}</h2>
      ${faceOk ? html`<label class="tog"><input type="checkbox" data-flag="face" ${st.face ? 'checked' : ''} ${st.codes.length ? '' : 'disabled'}>${t('vg.face')}
        <span class="sub">${st.codes.length ? t('vg.faceNote') : t('vg.faceNeed')}</span></label>` :
        html`<label class="tog"><input type="checkbox" data-flag="refineOn" ${st.refineOn ? 'checked' : ''}>${t('vg.refine')}
        <span class="sub">${t('vg.refineNote', { w, h })}</span></label>`}
      ${C.speakersOf(st).length ? html`<label class="tog"><input type="checkbox" data-flag="voiceOn" ${st.voiceOn !== false ? 'checked' : ''}>${t('vg.voice')}
        <span class="sub">${t('vg.voiceNote')}</span></label>` : ''}
      <label class="tog"><input type="checkbox" data-flag="upOn" ${st.upOn ? 'checked' : ''}>${t('def.upscale')}
        <span class="sub">${st.orient === 'h' ? '1920×1080' : '1080×1920'} · ${t('vg.upNote')}</span></label></div>`.__raw;
  renderActions();
}
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const pr = problems();
  const cur = st.cand.filter((x) => x.key === C.key(st));
  const n = Number(st.cands);
  const est = t('vg.estimate', { t: fmtDur(C.estimate(st, n)), n });
  // top right of the results pane (boss, session #14): what is missing / the estimate, then the button
  el.innerHTML = (!started() ? html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : est}</span>
      <button type="button" class="btn p" data-act="start" ${pr.length || ui.uploading ? 'disabled' : ''}>${t('ch.start')}</button>` :
    !cur.length ? html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : t('vg.regenNote') + ' · ' + est}</span>
      <button type="button" class="btn p" data-act="start" ${pr.length ? 'disabled' : ''}>${t('vg.regen', { n })}</button>` :
    html`<span class="note">${t('vg.moreHint')}</span>`).__raw;
}

// ---- right: results ----
const stText = (s) => ({ wait: t('ch.st.wait'), run: (s.pct || 0) + '%', done: t('ch.st.done'), failed: failWord(s.stopped),
  pick: t('ch.st.pick'), old: t('vg.st.old'), off: t('vg.st.off') }[s.state]);
const stDot = (s) => ({ run: 'busy', done: 'ok', failed: s.stopped ? '' : 'bad', pick: 'warn', old: 'warn' }[s.state] || '');

function tile(x, i) {
  const isBase = st.base === x.job, saved = st.saved.includes(x.job);
  const busyTile = x.state === 'pending' || x.state === 'running';
  const over = { running: x.pct + '%', pending: t('ch.st.wait'), failed: failWord(x.stopped), arriving: t('jobs.arriving'), gone: t('vg.gone') }[x.state];
  return html`<div class="col ${enter('vg:' + x.job + (x.url ? ':v' : ''))}" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid ${isBase ? 'var(--blue-line)' : x.state === 'failed' && !x.stopped ? 'var(--bad-line)' : '#24242A'}; border-radius: 10px; ${x.old && !isBase ? 'opacity: .45' : ''}">
    <div class="ph ${busyTile ? 'sk' : ''}" style="aspect-ratio: ${ratio()}">
      ${x.url ? html`<video src="${x.url}" preload="metadata" controls playsinline style="object-fit: contain; background: #000"></video>` : ''}
      ${over ? html`<span class="mono" style="position: absolute; inset: 44% 0; text-align: center; font-size: 13px; color: ${x.state === 'failed' ? failColor(x.stopped) : 'var(--blue-fg)'}">${over}</span>` : ''}
      <span class="mono" style="position: absolute; left: 6px; top: 6px; padding: 1px 7px; border-radius: 6px; background: #0E0E10CC; font-size: 11px; color: var(--fg2); pointer-events: none">seed ${x.seed}${x.old ? ' · ' + t('vg.st.old') : ''}</span></div>
    <div class="row" style="gap: 2px">
      <button type="button" class="ib m" data-act="star" data-i="${i}" aria-pressed="${saved}" aria-label="${t('ch.saveCand')}" title="${t('ch.saveCand')}" ${x.url ? '' : 'disabled'}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="${saved ? '#FACC15' : 'none'}" stroke="${saved ? '#FACC15' : 'currentColor'}" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg></button>
      <button type="button" class="ib m" data-act="base" data-i="${i}" aria-pressed="${isBase}" aria-label="${t('vg.baseCand')}" title="${t('vg.baseCand')}" ${x.url ? '' : 'disabled'}>
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="${isBase ? '#8AB4F8' : 'none'}"/></svg></button>
      <button type="button" class="ib m" data-act="redoCand" data-i="${i}" aria-label="${t('vg.redoCand')}" title="${t('vg.redoCand')}" ${busyTile ? 'disabled' : ''}>${icon('redo', 14)}</button>
      <div class="grow"></div>
      ${x.face != null ? html`<span class="sub row" style="gap: 5px" title="${t('vg.faceTip')}"><span class="dot ${x.face >= 0.5 ? 'ok' : 'warn'}"></span>${x.face.toFixed(2)}</span>` : ''}</div></div>`;
}

function videoBox(r, caption) {
  return html`<div class="col" style="gap: 4px; flex: 0 1 ${st.orient === 'h' ? '560px' : '320px'}; min-width: 200px">
    <div class="ph" style="aspect-ratio: ${ratio()}"><video src="${C.urlOf(r)}" controls playsinline style="object-fit: contain; background: #000"></video></div>
    <span class="sub" style="text-align: center">${caption}</span></div>`;
}

function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const s = C.stages(st);
  const all = st.cand.map((x) => C.cand(st, x));
  const bj = C.jobOf(st.base);
  const h3 = st.model === 'h3' || (bj && bj.type === 'video.r2v');
  const speaks = C.speakersOf(st).length > 0 && st.voiceOn !== false;   // 목소리 (기능 17): script cuts with dialogue
  const pills = [[t('vg.gen'), s.gen]].concat(h3 ? [[t('vg.refineStage'), s.ref]] : [], speaks ? [[t('vg.voice'), s.voice]] : [], st.upOn ? [[t('def.upscale'), s.up]] : []);
  const failed = all.filter((x) => x.state === 'failed').length + (s.ref.state === 'failed') + (s.voice.state === 'failed') + (s.up.state === 'failed');
  const stoppedOnly = all.filter((x) => x.state === 'failed').every((x) => x.stopped) && [s.ref, s.voice, s.up].every((x) => x.state !== 'failed' || x.stopped);
  const fin = C.finalOf(st), res = C.resultOf(st), voiced = C.voicedOf(st);
  const unsaved = all.filter((x) => x.url && x.job !== st.base && !st.saved.includes(x.job)).length;
  const refineDue = bj && bj.type === 'video.r2v' && st.refineOn && (!st.refineJob || st.refineJob.from !== st.base);
  const voiceDue = fin && speaks && (!st.voiceJob || st.voiceJob.from !== fin.job + '/' + fin.name);
  const upDue = !voiceDue && voiced && st.upOn && (!st.upJob || st.upJob.from !== voiced.job + '/' + voiced.name);
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px" aria-label="${t('ch.progress')}">${pills.map(([lb, x]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
          <span class="dot ${stDot(x)}"></span>${lb}<span class="sub mono">${stText(x)}</span></li>`)}</ol>
      <div class="toprow-end">${anyActive() ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.askStop')}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${failed ? html`<div class="row banner ${stoppedOnly ? 'warn' : 'bad'}" role="status" style="justify-content: space-between"><span>${t(stoppedOnly ? 'vg.stoppedN' : 'vg.failedN', { n: failed })}</span>
      <button type="button" class="btn p" data-act="resume" style="min-height: 34px">${icon('play', 12)}${t('jobs.resume')}</button></div>` : ''}
    ${started() && s.gen.state === 'old' ? html`<div class="row banner warn" role="status" style="justify-content: space-between"><span>${t('vg.oldBanner')}</span>
      <button type="button" class="ib p" data-act="start" aria-label="${t('vg.regen', { n: st.cands })}" title="${t('vg.regen', { n: st.cands })}" ${problems().length ? 'disabled' : ''}>${icon('redo')}</button></div>` : ''}

    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${t('vg.gen')}</h2>
        <span class="sub mono">${MODELS[st.model].name} · ${t('vg.secN', { n: st.sec })} · ${t('ch.candN', { n: all.filter((x) => !x.old).length })}</span><div class="grow"></div>
        <span class="sub mono">${started() ? stText(s.gen) : ''}</span>
        ${started() && s.gen.state !== 'run' && s.gen.state !== 'old' ? html`<button type="button" class="ib m" data-act="more" aria-label="${t('ch.more')}" title="${t('vg.moreTip', { n: st.cands })}" ${problems().length ? 'disabled' : ''}>${icon('plus')}</button>` : ''}</div>
      ${s.gen.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${s.gen.pct}%"></div></div>` : ''}
      <p class="sub" style="margin: 6px 0 0 38px; color: ${s.gen.state === 'pick' ? 'var(--warn-fg)' : 'var(--fg4)'}">${s.gen.state === 'pick' ? t('vg.pickBase') :
        started() ? (st.mode === 'all' ? t('vg.autoBase') : t('vg.baseRule')) : t('vg.genWait', { n: st.cands })}</p>
      ${all.length ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(${st.orient === 'h' ? '260px' : '150px'}, 1fr)); gap: 12px">${all.map(tile)}</div>` : ''}
    </section>

    ${h3 ? html`<section class="card" style="${st.refineOn ? '' : 'opacity: .55'}">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('vg.refineStage')}</h2>
        <span class="sub">${st.refineOn ? t('vg.refineMeta') : t('vg.refineOff')}</span><div class="grow"></div>
        <span class="sub mono">${st.refineOn && st.base ? stText(s.ref) : ''}</span></div>
      ${s.ref.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${s.ref.pct}%"></div></div>` : ''}
      ${refineDue && st.base && !anyActive() ? html`<div class="row" style="margin-top: 12px; justify-content: space-between">
        <span class="sub">${st.refineJob ? t('vg.refineOld') : t('vg.refineReady')}</span>
        <button type="button" class="btn p" data-act="step">${t('vg.refineGo')}</button></div>` :
        !st.base ? html`<p class="sub" style="margin: 6px 0 0 38px">${t('vg.needBase')}</p>` :
        fin && st.refineOn ? html`<div class="row" style="margin-top: 12px">${videoBox(fin, t('vg.refined'))}</div>` : ''}
    </section>` : ''}

    <section class="card">
      <div class="row"><span class="num">${h3 ? 3 : 2}</span><h2 class="h2">${t('vg.result')}</h2>
        <span class="sub">${st.upOn ? t('vg.upMeta', { size: st.orient === 'h' ? '1920×1080' : '1080×1920' }) : t('sw.upOff')}</span><div class="grow"></div>
        <span class="sub mono">${st.upOn && fin ? stText(s.up) : ''}</span>
        ${fin && !st.upOn ? html`<button type="button" class="ib m" data-act="upNow" aria-label="${t('def.upscale')}" title="${t('vg.upNow')}">${icon('expand', 14)}</button>` : ''}</div>
      ${s.up.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${s.up.pct}%"></div></div>` : ''}
      ${voiceDue && !anyActive() ? html`<div class="row" style="margin-top: 12px; justify-content: space-between">
        <span class="sub">${t('vg.voiceReady')}</span><button type="button" class="btn p" data-act="step">${t('vg.voiceGo')}</button></div>` : ''}
      ${upDue && !anyActive() ? html`<div class="row" style="margin-top: 12px; justify-content: space-between">
        <span class="sub">${st.upJob ? t('vg.upOld') : t('vg.upReady')}</span><button type="button" class="btn p" data-act="step">${t('def.upscale')}</button></div>` : ''}
      ${res || fin ? html`<div class="row" style="margin-top: 12px; align-items: flex-start">${videoBox(res || fin, res && res.name === 'up.mp4' ? t('vg.upscaled') : t('sw.result'))}</div>` :
        html`<p class="sub" style="margin: 6px 0 0 38px">${t('vg.resultWait')}</p>`}
    </section>

    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('vg.askClose', { n: unsaved })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button>
        <button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between; padding-top: 4px">
        <p class="sub" style="margin: 0">${seq ? t('vg.seqNote') : t('vg.saveNote')}</p>
        ${seq ? html`<a class="btn p lg" href="#/video/seq?id=${seq.id}">← ${t('sq.back', { name: seq.name || t('sq.untitled') })}</a>` : html`<div class="row" style="gap: 8px">
          <button type="button" class="btn ghost" data-act="close" ${anyActive() ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="save" ${(res || fin || st.saved.length) && !anyActive() ? '' : 'disabled'}>${icon('star', 16)}${t('sw.save')}</button></div>`}</div>` : ''}`.__raw;
  renderActions();
}

function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(VIDEO_MENU(), seq ? '#/video/seq' : '#/video/one')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const d = a.dataset;
  switch (d.act) {
    case 'set': {
      const k = d.k;
      st[k] = k === 'vsa' ? d.v === '1' : ['cands', 'steps'].includes(k) ? Number(d.v) : d.v;
      own(k); renderSet(); renderOut(); return saveSoon();
    }
    case 'model': if (st.model !== d.m) {
      C.applyModel(st, d.m);
      // a sequence cut owns its model and length; the rest stays the sequence's common setting unless the cut changed
      // it itself (session #14: switching one cut to pi-H3 turned the sequence's 1 candidate into the model's 2)
      if (seq) for (const k of ['cands', 'upOn', 'face', 'steps', 'vsa', 'refSize', 'refineOn']) if (!st.over.includes(k) && seq.common[k] !== undefined) st[k] = seq.common[k];
      own('model'); own('sec');
    } renderSet(); renderOut(); return saveSoon();
    case 'delCode': st.codes = st.codes.filter((c) => c !== d.code); own('codes'); renderSet(); renderOut(); return saveSoon();
    case 'pick': ui.pick = ui.pick === d.slot ? null : d.slot; return renderSet();
    case 'unpick': ui.pick = null; return renderSet();
    case 'frame': { const sl = ui.pick; st[sl] = { rel: d.rel, name: d.name }; ui.pick = null; renderSet(); renderOut(); saveSoon(); return followFrame(sl); }
    case 'clearFrame': st[d.slot] = null; renderSet(); renderOut(); return saveSoon();
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'start': return generate();
    case 'more': return generate();
    case 'redoCand': ui.autoErr = false; if (await act(() => C.redoCand(st, Number(d.i), label()))) { /* returns undefined */ } await save(); await poll(); return renderOut();
    case 'star': { const j = st.cand[Number(d.i)].job; st.saved = st.saved.includes(j) ? st.saved.filter((x) => x !== j) : st.saved.concat([j]); renderOut(); return save(); }
    case 'base': st.base = st.cand[Number(d.i)].job; ui.autoErr = false; renderOut(); return save();
    case 'step': return step();
    case 'upNow': st.upOn = true; own('upOn'); renderSet(); return step();
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop':
      ui.askStop = false;
      for (const j of C.cutJobs(st).map(C.jobOf).filter((x) => x && (x.state === 'pending' || x.state === 'running'))) await act(() => srv(`/jobs/${j.id}/cancel`, {}));
      await poll(); return;
    case 'resume':
      ui.autoErr = false;
      for (const j of C.cutJobs(st).map(C.jobOf).filter((x) => x && x.state === 'failed')) await act(() => api('/api/retry', { job: j.id }));
      await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; return finish(false);
    case 'save': return finish(true);
    default:
  }
}
function onInput(e) {
  const k = e.target.dataset.k;
  if (!k || k === 'sec' || e.target.tagName === 'SELECT') return;   // 길이 is taken on change (clamped)
  st[k] = k === 'seed' ? Number(e.target.value) || 0 : e.target.value;
  if (k === 'title') { const h = root.querySelector('[data-set] h1'); if (h && !seq) h.textContent = st.title || t('vg.newCut'); }
  renderActions();
  if (started()) renderOut();
  saveSoon();
}
function onChange(e) {
  const dd = e.target.dataset;
  if (dd.file && e.target.files[0]) return upload(dd.file, e.target.files[0]);
  if (dd.add === 'code' && e.target.value) { st.codes = st.codes.concat([e.target.value]).slice(0, 2); own('codes'); renderSet(); renderOut(); return saveSoon(); }
  if (dd.flag) { st[dd.flag] = e.target.checked; own(dd.flag); renderSet(); renderOut(); return saveSoon(); }
  if (dd.k === 'sec') { const v = C.secClamp(st.model, e.target.value); if (v != null) { st.sec = v; own('sec'); } renderSet(); renderOut(); return saveSoon(); }
  if (dd.k === 'refSize') { st.refSize = e.target.value; own('refSize'); renderSet(); renderOut(); return saveSoon(); }
}

export default {
  async mount(el, { query }) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    const lib = await act(() => api('/api/library?scope=shared')) || {};
    chars = lib.characters || []; locs = lib.locations || [];
    scenes = ((await act(() => api('/api/library?scope=project')) || {}).items || []).filter((i) => i.kind === 'images');
    const cid = query.get('cut'), sid = query.get('seq');
    seq = sid ? await act(() => api('/api/doc?kind=sequences&id=' + encodeURIComponent(sid))) : null;
    if (seq && !seq.id) seq = null;
    if (cid) {   // always from disk: the sequence screen may have changed this cut meanwhile
      const d = await act(() => api('/api/doc?kind=cuts&id=' + encodeURIComponent(cid)));
      st = C.freshCut(d && d.id ? d : {});
      ui = freshUi();
    } else if (!st || (!cid && st.seq)) {
      st = C.freshCut(); ui = freshUi();
    }
    if (query.get('code') && !started() && !st.codes.length) st.codes = [query.get('code')];
    open = (await act(() => api('/api/docs?kind=cuts')) || []).filter((d) => !d.seq && d.id !== st.id);
    await C.loadFaces(st);
    renderAll();
  },
  async update() {
    if (!root || !st) return;
    await C.loadFaces(st);
    let changed = C.autoBase(st);
    if (st.mode === 'all' && !busy && !ui.autoErr && st.base) {
      busy = true;
      const r = await act(() => C.nextStep(st, label()));
      busy = false;
      if (r === undefined) ui.autoErr = true;   // shown once; a click (●, 다시, 이어서 하기) tries again
      changed = changed || !!r;
    }
    if (changed) await save();
    renderActions();
    if (playing()) return;   // a video being watched is not torn down by the 3 s refresh
    renderOut();
  },
  unmount() { root = null; },
};
