// 영상 바꾸기: one screen (design §11) — settings left (원본 · 사람 · 배경 · 마무리), results right (나누기 · 경계 ·
// 구간 · 마무리) filled in place from the video.swap job's units. #/swap?job=<id> reopens a job.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, codeLabel, isStopped, unitStopped, failWord } from '../app.js';

const SEG_MIN = 25;       // one 4~5 s segment on one node, H3 8+4 incl. its boundary share (5 s test: 27 min, session #10;
                         // 29 s = 7 segments on C·D: 91 min, session #7)
const UP_MIN_PER_SEC = 0.9;   // SeedVR2: 263 s per 5 s on one node (estimate); C·D share the chunks

let root = null, st = null, chars = [], locs = [], detail = null, plan = null, segMeta = {}, ui = {};

function fresh() {
  return { video: null, probe: null, start: '', end: '', crop: 'orig', people: [{ face: null, desc: '', to: 'keep' }],
    bg: 'keep', bgCode: '', place: '', face: true, up: false, seed: 1, refSize: 'match', taller: '', job: null };
}
const freshUi = () => ({ zoom: null, askStop: false, play: false, side: false, uploading: false, probing: false, adv: false });

// ---- derived ----
const swapped = () => st.people.filter((p) => p.to !== 'keep');
const path = () => swapped().length === 2 ? 'two' : 'one';
const job = () => st.job && (S.jobs.find((j) => j.id === st.job) || detail);
const units = () => (detail && detail.units) || [];
const active = (j) => j && (j.state === 'pending' || j.state === 'running');
const srcSec = () => {
  const total = st.video && st.video.sec ? st.video.sec : 0;
  const a = Number(st.start) || 0, b = st.end !== '' ? Number(st.end) : total;
  return Math.max(0, b - a);
};
const hasSheet = (code) => { const c = chars.find((x) => x.code === code); return !!c && c.images.some((i) => i.kind === 'sheet'); };

function problems() {
  const out = [], sw = swapped();
  if (!(S.status && S.status.settings && S.status.settings.project)) out.push(t('vg.needProject'));   // the result goes to the project library
  if (!st.video) out.push(t('sw.needVideo'));
  if (!sw.length) out.push(t('sw.needOne'));
  if (sw.length > 2) out.push(t('sw.max2'));
  if (sw.length === 1 && st.people.length > 1 && !sw[0].desc.trim()) out.push(t('sw.needDescOne'));
  if (sw.length === 2) {
    if (sw.some((p) => !p.desc.trim())) out.push(t('sw.needDescTwo'));
    if (sw.some((p) => !hasSheet(p.to))) out.push(t('sw.needSheets'));
    if (st.bg !== 'keep') out.push(t('sw.noBgTwo'));
  }
  if (st.bg === 'image' && !st.bgCode) out.push(t('sw.needBg'));
  if (st.bg === 'text' && !st.place.trim()) out.push(t('sw.needPlace'));
  return out;
}
function pathNote() {
  const sw = swapped();
  if (!sw.length) return '';
  if (sw.length > 2) return t('sw.max2');
  let s = sw.length === 2 ? t('sw.pathTwo') : t('sw.pathOne') + (st.people.length > 1 ? ' · ' + t('sw.pathOneMany') : '');
  return s;
}
function estimate() {
  const sec = srcSec();
  if (!sec) return '';
  const segs = Math.max(1, Math.round(sec / 4.5));   // 4~5 s segments
  let m = (path() === 'two' ? segs : Math.ceil(segs / 2)) * SEG_MIN + 2;   // two people run in order on one node
  if (st.up) m += sec * UP_MIN_PER_SEC / 2;
  return t('sw.estimate', { m: Math.round(m), how: path() === 'two' ? t('sw.inOrder') : 'C·D' });
}

// ---- actions ----
async function pickVideo(f) {
  const url = URL.createObjectURL(f);
  st = Object.assign(fresh(), { video: { name: f.name, url }, people: st.people.map((p) => Object.assign({}, p, { face: null })) });
  ui.uploading = true; renderAll();
  const r = await act(async () => {
    const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.status);
    return j;
  });
  ui.uploading = false;
  if (!r) return renderAll();
  st.video.sha = r.sha;
  st.video.rel = r.rel;
  ui.probing = true; renderAll();
  const p = await act(() => api('/api/srv/jobs', { type: 'video.probe', inputs: { video: r.sha }, options: {} }));
  if (p) st.probe = { job: p.id };
  await poll();
}

// the probe job's info.json + face crops → source info and one 사람 row per face
async function readProbe() {
  const j = st.probe && S.jobs.find((x) => x.id === st.probe.job);
  if (!j || st.probe.info) return;
  if (j.state === 'failed') { ui.probing = false; st.probe.info = { faces: [] }; toast(t('sw.probeFailed'), true); return renderAll(); }
  const f = j.files.find((x) => x.name === 'info.json' && x.synced);
  if (j.state !== 'done' || !f || j.files.some((x) => !x.synced)) return;
  const info = await act(async () => (await fetch(jobFileUrl(j, 'info.json'))).json());
  if (!info) return;
  st.probe.info = info;
  st.probe.base = j;
  Object.assign(st.video, { sec: info.sec, w: info.width, h: info.height });
  const keep = st.people.filter((p) => p.to !== 'keep').map((p) => p.to);
  st.people = info.faces.length ? info.faces.map((fc, i) => ({ face: fc.k, desc: '', to: keep[i] || 'keep' })) : [{ face: null, desc: '', to: keep[0] || 'keep' }];
  ui.probing = false;
  srv(`/jobs/${j.id}/close`, {}).catch(() => {});   // the probe is not a work result
  renderAll();
}

async function start() {
  const pr = problems();
  if (pr.length) return toast(pr[0], true);
  const sw = swapped();
  const o = { face: st.face, upscale: st.up, seed: st.seed, ref_size: st.refSize, source_name: st.video.name };
  if (st.start !== '') o.start = Number(st.start);
  if (st.end !== '') o.end = Number(st.end);
  if (st.crop === '9:16') o.crop = '9:16';
  if (st.bg === 'text' || (st.bg === 'image' && st.place.trim())) o.place = st.place.trim();
  if (sw.length === 1 && st.people.length > 1) o.person = sw[0].desc.trim();
  if (sw.length === 2) Object.assign(o, { code_b: sw[1].to, person_a: sw[0].desc.trim(), person_b: sw[1].desc.trim() }, st.taller ? { taller: st.taller } : {});
  const body = { type: 'video.swap', code: sw[0].to, options: o, inputs: { video: st.video.sha } };
  if (st.bg === 'image') body.background_code = st.bgCode;
  const r = await act(() => api('/api/codejob', body));
  if (!r) return;
  st.job = r.id;
  history.replaceState(null, '', '#/swap?job=' + r.id);
  renderSet();   // settings are read-only from here (no second 시작)
  await poll();
}

// ---- left pane ----
function seg(k, opts, cur, label, dis) {
  return html`<div class="seg" role="group" aria-label="${label}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${k}" data-v="${v}" aria-pressed="${String(cur) === String(v)}" ${dis ? 'disabled' : ''}>${l}</button>`)}</div>`;
}
function renderSet() {
  const el = root.querySelector('[data-set]');
  const lock = !!st.job;
  const v = st.video;
  const ratio = v && v.w ? `${v.w} / ${v.h}` : '16 / 9';
  const cropBox = st.crop === '9:16' && v && v.w > v.h * 9 / 16;
  const faceUrl = (k) => st.probe && st.probe.base ? jobFileUrl(st.probe.base, `face_${k}.png`) : '';
  const opts = [['keep', t('sw.keep')]].concat(chars.filter((c) => c.master).map((c) => [c.code, codeLabel(c) + (hasSheet(c.code) ? '' : ' · ' + t('sw.noSheet'))]));
  el.innerHTML = html`
    <div><div class="lbl">${t('nav.swap')}</div>
      <h1 class="h1" style="word-break: break-all">${v ? v.name : t('sw.new')}${swapped().length ? html` <span class="mono" style="font-weight: 400; color: var(--fg3); word-break: normal">→ ${swapped().map((p) => p.to).join(' · ')}</span>` : ''}</h1><p class="lead">${t('lead.swap')}</p></div>
    ${lock ? html`<div class="row" style="gap: 8px"><span class="sub">${t('sw.locked')}</span><button type="button" class="btn ghost" data-act="new" style="min-height: 32px">${icon('plus', 14)}${t('sw.new')}</button></div>` : ''}

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('sw.source')}</h2>
        ${v && v.sec ? html`<span class="sub mono">${v.sec.toFixed(1)}s · ${v.w}×${v.h} ${v.w > v.h ? t('sw.landscape') : t('sw.portrait')}</span>` : ''}</div>
      <label class="ph" style="aspect-ratio: ${ratio}; display: flex; align-items: center; justify-content: center; cursor: ${lock ? 'default' : 'pointer'}">
        ${v && v.url ? html`<video src="${v.url}" muted preload="metadata" style="object-fit: contain"></video>` :
          v && v.rel ? html`<video src="${filesUrl(v.rel)}#t=0.5" muted preload="metadata" style="object-fit: contain"></video>` :
          html`<span class="sub">${ui.uploading ? t('common.loading') : lock ? t('sw.noSource') : t('sw.pickVideo')}</span>`}
        ${cropBox ? html`<div aria-hidden="true" style="position: absolute; top: 0; bottom: 0; left: 50%; width: ${(100 * v.h * 9 / 16 / v.w).toFixed(1)}%; transform: translateX(-50%); border: 2px solid var(--blue); box-shadow: 0 0 0 999px rgba(14,14,16,.6)"></div>` : ''}
        ${lock ? '' : html`<input type="file" accept="video/*" data-file="video" hidden>`}</label>
      <div class="row" style="gap: 12px">
        <label class="row lbl" style="gap: 8px">${t('sw.from')}<input class="in mono" data-k="start" inputmode="decimal" placeholder="0.0" value="${st.start}" style="width: 80px" ${lock ? 'disabled' : ''}></label>
        <label class="row lbl" style="gap: 8px">${t('sw.to')}<input class="in mono" data-k="end" inputmode="decimal" placeholder="${v && v.sec ? v.sec.toFixed(1) : ''}" value="${st.end}" style="width: 80px" ${lock ? 'disabled' : ''}></label>
        <span class="sub">${t('sw.seconds')}</span></div>
      <div class="col" style="gap: 6px"><span class="lbl">${t('sw.orient')}</span>
        ${seg('crop', [['orig', t('sw.orig')], ['9:16', t('sw.crop916')]], st.crop, t('sw.orient'), lock || !(v && v.w > v.h))}
        <span class="sub">${t('sw.canvasNote')}</span></div></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('sw.people')}</h2>
        <span class="sub">${ui.probing ? t('sw.probing') : st.probe && st.probe.info ? t('sw.found', { n: st.probe.info.faces.length }) : ''}</span></div>
      <div class="list">${st.people.map((p, i) => html`
        <div class="item-row" style="flex-direction: column; align-items: stretch; gap: 8px; padding: 8px 10px">
          <div class="row" style="gap: 8px; flex-wrap: nowrap">
            ${p.face != null && faceUrl(p.face) ? html`<img src="${faceUrl(p.face)}" alt="" style="width: 40px; height: 40px; border-radius: 8px; object-fit: cover; flex: 0 0 auto">` :
              html`<span aria-hidden="true" style="width: 40px; height: 40px; border-radius: 8px; background: #2C2C33; flex: 0 0 auto"></span>`}
            <span aria-hidden="true" class="sub">→</span>
            <select class="in" data-person="${i}" data-f="to" style="flex: 1 1 0; min-width: 0" aria-label="${t('sw.to2')}" ${lock ? 'disabled' : ''}>
              ${opts.map(([val, label]) => html`<option value="${val}" ${val === p.to ? 'selected' : ''}>${label}</option>`)}</select>
            ${lock || st.people.length < 2 ? '' : html`<button type="button" class="ib m" data-act="delPerson" data-i="${i}" aria-label="${t('sw.delPerson')}" title="${t('sw.delPerson')}">${icon('x', 14)}</button>`}</div>
          <input class="in" data-person="${i}" data-f="desc" value="${p.desc}" placeholder="${t('sw.descPh')}" ${lock ? 'disabled' : ''}></div>`)}</div>
      ${lock ? '' : html`<button type="button" class="btn add" data-act="addPerson">${icon('plus', 14)}${t('sw.addPerson')}</button>`}
      <span class="sub" style="color: ${swapped().length > 2 ? 'var(--bad-fg)' : 'var(--fg4)'}">${pathNote()}</span>
      ${swapped().length === 2 ? html`<div class="row" style="gap: 8px"><span class="lbl">${t('sw.taller')}</span>
        ${seg('taller', [['', t('sw.same')], ['a', swapped()[0].to], ['b', swapped()[1].to]], st.taller, t('sw.taller'), lock)}</div>` : ''}</div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('sw.bg')}</h2><span class="tag" title="${t('sw.bgTip')}">${t('sw.experimental')}</span></div>
      ${seg('bg', [['keep', t('sw.keep')], ['image', t('sw.bgImage')], ['text', t('sw.bgText')]], st.bg, t('sw.bg'), lock || swapped().length === 2)}
      ${st.bg === 'image' ? html`<select class="in" data-k="bgCode" ${lock ? 'disabled' : ''}><option value="">${t('sw.pickLoc')}</option>
        ${locs.filter((l) => l.master).map((l) => html`<option value="${l.code}" ${l.code === st.bgCode ? 'selected' : ''}>${l.code}</option>`)}</select>` : ''}
      ${st.bg !== 'keep' ? html`<input class="in" data-k="place" value="${st.place}" placeholder="${st.bg === 'text' ? t('sw.placePh') : t('sw.placeOpt')}" ${lock ? 'disabled' : ''}>` : ''}</div>

    <div class="sec">
      <label class="tog"><input type="checkbox" data-flag="face" ${st.face ? 'checked' : ''} ${lock ? 'disabled' : ''}>${t('sw.facePass')} <span class="sub">${t('sw.facePassNote')}</span></label>
      <div class="row" style="gap: 10px; flex-wrap: nowrap">
        <label class="tog grow"><input type="checkbox" data-flag="up" ${st.up ? 'checked' : ''} ${lock ? 'disabled' : ''}>${t('def.upscale')} <span class="sub">${st.crop === '9:16' || (v && v.h > v.w) ? '1080×1920' : '1920×1080'}</span></label>
        <button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      ${ui.adv ? html`<div style="display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 10px 12px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('ch.seed')}<input class="in mono" data-k="seed" inputmode="numeric" value="${st.seed}" ${lock ? 'disabled' : ''}></label>
        <label class="field">${t('def.ref_size')}<select class="in" data-k="refSize" ${lock ? 'disabled' : ''}>
          <option value="match" ${st.refSize === 'match' ? 'selected' : ''}>${t('def.ref_size.match')}</option>
          <option value="max" ${st.refSize === 'max' ? 'selected' : ''}>${t('def.ref_size.max')}</option></select></label></div>` : ''}</div>
`.__raw;
  renderActions();
}
// the main action under the progress pills, right (boss, session #14): the note (what is missing · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const pr = problems();
  el.innerHTML = st.job ? '' : html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length && !ui.probing ? pr.join(' · ') : estimate()}</span>
    <button type="button" class="btn p" data-act="start" ${pr.length || ui.uploading || ui.probing ? 'disabled' : ''}>${t('ch.start')}</button>`.__raw;
}

// ---- right pane ----
const unitsOf = (step) => units().filter((u) => u.spec.step === step);
function stageOf(steps) {
  const us = units().filter((u) => steps.includes(u.spec.step));
  if (!us.length) return { state: active(job()) ? 'wait' : 'wait', pct: 0 };
  if (us.some((u) => u.state === 'running' || u.state === 'pending')) {
    return us.some((u) => u.state === 'running') || us.some((u) => u.state === 'done') ?
      { state: 'run', pct: Math.round(100 * us.reduce((a, u) => a + (u.state === 'done' ? 1 : u.progress || 0), 0) / us.length) } : { state: 'wait' };
  }
  return us.some((u) => u.state === 'failed') ? { state: 'failed', stopped: us.filter((u) => u.state === 'failed').every(unitStopped) } : { state: 'done' };
}
const stText = (s) => ({ wait: t('ch.st.wait'), run: (s.pct || 0) + '%', done: t('ch.st.done'), failed: failWord(s.stopped) }[s.state]);
const stDot = (s) => ({ wait: '', run: 'busy', done: 'ok', failed: s.stopped ? '' : 'bad' }[s.state]);
const fileOf = (j, name) => j && j.files.find((f) => f.name === name && f.synced);
const fsec = (f) => ((f + (Number(st.start) || 0) * 24) / 24).toFixed(1);

function redoBtn(u, label, top) {   // top: on video tiles, clear of the player's control bar
  return u && u.state !== 'running' && u.state !== 'pending' && !active(job()) ? html`<button type="button" class="ib" data-act="redo" data-idx="${u.idx}" aria-label="${label}" title="${label}"
    style="position: absolute; right: 4px; ${top ? 'top' : 'bottom'}: 4px; width: 28px; height: 28px; background: #16161AE6">${icon('redo', 13)}</button>` : '';
}

function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const j = job();
  const two = path() === 'two' || (detail && detail.inputs && detail.inputs.sheet_b);
  const ratio = plan ? `${plan.canvas[0]} / ${plan.canvas[1]}` : '16 / 9';
  const sPlan = stageOf(['plan']), sEdit = stageOf(['edit']), sSeg = stageOf(['segment']), sPost = stageOf(['stitch', 'face', 'up', 'final']);
  const steps = [[t('sw.split'), sPlan]].concat(two ? [] : [[t('sw.bounds'), sEdit]], [[t('sw.segs'), sSeg], [t('sw.finish'), sPost]]);
  const failed = j && j.state === 'failed';
  const segUs = unitsOf('segment').sort((a, b) => a.spec.k - b.spec.k);
  const doneSegs = segUs.filter((u) => fileOf(j, `seg_${String(u.spec.k).padStart(2, '0')}.mp4`));
  const result = fileOf(j, 'swap.mp4');
  const bounds = plan ? plan.bounds : [];
  const editU = (f) => unitsOf('edit').find((u) => u.spec.frame === f);
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px">${steps.map(([label, x]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
          <span class="dot ${st.job ? stDot(x) : ''}"></span>${label}<span class="sub mono">${st.job ? stText(x) : t('ch.st.wait')}</span></li>`)}</ol>
      <div class="toprow-end">${active(j) ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('sw.askStop', { n: doneSegs.length })}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${!st.job ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${failed ? html`<div class="row banner ${isStopped(j) ? 'warn' : 'bad'}" role="status" style="justify-content: space-between"><span>${isStopped(j) ? t('sw.stopped') : t('sw.failed', { e: (j.errors.find((x) => x !== 'cancelled') || '').split('\n')[0].slice(0, 120) })}</span>
      <button type="button" class="btn p" data-act="resume" style="min-height: 34px">${icon('play', 12)}${t('jobs.resume')}</button></div>` : ''}

    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${t('sw.split')}</h2>
        <span class="sub mono">${plan ? t('sw.splitMeta', { n: plan.segments.length, w: plan.canvas[0], h: plan.canvas[1], k: (plan.kept || []).length }) : t('sw.splitWait')}</span>
        <div class="grow"></div><span class="sub mono">${st.job ? stText(sPlan) : ''}</span></div>
      ${two ? html`<p class="sub" style="margin: 8px 0 0 38px">${t('sw.chainNote')}</p>` : bounds.length ? html`
        <div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(${plan.canvas[0] > plan.canvas[1] ? '180px' : '96px'}, 1fr)); gap: 10px">
          ${bounds.map((f) => {
            const u = editU(f), busy = u && (u.state === 'running' || u.state === 'pending');
            const ed = fileOf(j, `bndc_${String(f).padStart(5, '0')}.png`);
            return html`<div class="col" style="gap: 4px">
              <div class="ph ${busy ? 'sk' : ''}" style="aspect-ratio: ${ratio}">
                ${ed ? html`<button type="button" data-act="zoom" data-f="${f}" style="all: unset; cursor: zoom-in; display: block; width: 100%; height: 100%" title="${t('sw.zoom')}"><img src="${jobFileUrl(j, ed.name)}?v=${ed.sha.slice(0, 8)}" alt=""></button>` :
                  fileOf(j, `bnd_${String(f).padStart(5, '0')}.png`) && !busy ? html`<img src="${jobFileUrl(j, `bnd_${String(f).padStart(5, '0')}.png`)}" alt="" style="opacity: .5">` : ''}
                ${redoBtn(u, t('sw.redoBound'))}</div>
              <span class="sub mono" style="text-align: center">${fsec(f)}s${u && u.state === 'done' && !ed ? ' · ' + t('sw.noGuide') : ''}</span></div>`;
          })}</div>` : ''}
      ${ui.zoom != null && j ? html`<div class="col" style="margin-top: 12px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px; gap: 8px">
        <div class="row" style="justify-content: space-between"><span class="lbl">${t('sw.boundAt', { s: fsec(ui.zoom) })}</span>
          <button type="button" class="ib m" data-act="unzoom" aria-label="${t('jobs.close')}">${icon('x', 14)}</button></div>
        <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px">
          <div class="col" style="gap: 4px"><div class="ph" style="aspect-ratio: ${ratio}"><img src="${jobFileUrl(j, `bnd_${String(ui.zoom).padStart(5, '0')}.png`)}" alt="" style="object-fit: contain"></div><span class="sub" style="text-align: center">${t('sw.srcFrame')}</span></div>
          <div class="col" style="gap: 4px"><div class="ph" style="aspect-ratio: ${ratio}"><img src="${jobFileUrl(j, `bndc_${String(ui.zoom).padStart(5, '0')}.png`)}" alt="" style="object-fit: contain"></div><span class="sub" style="text-align: center">${t('sw.editFrame')}</span></div></div></div>` : ''}
    </section>

    <section class="card">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('sw.segs')}</h2>
        <span class="sub mono">${segUs.length ? t('sw.segMeta', { d: doneSegs.length, n: segUs.length }) : ''}</span><div class="grow"></div>
        <span class="sub mono">${st.job ? stText(sSeg) : ''}</span>
        ${doneSegs.length ? html`<button type="button" class="ib m" data-act="playAll" aria-expanded="${ui.play}" aria-label="${t('sw.playAll')}" title="${t('sw.playAll')}">${icon('play', 14)}</button>` : ''}</div>
      ${sSeg.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${sSeg.pct}%"></div></div>` : ''}
      ${ui.play && doneSegs.length ? html`<div class="ph" style="margin-top: 12px; aspect-ratio: ${ratio}; max-height: 420px"><video data-playlist controls autoplay style="object-fit: contain; background: #000"></video></div>
        <span class="sub">${t('sw.playNote')}</span>` : ''}
      ${segUs.length ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(${plan && plan.canvas[0] > plan.canvas[1] ? '200px' : '110px'}, 1fr)); gap: 10px">
        ${segUs.map((u) => {
          const name = `seg_${String(u.spec.k).padStart(2, '0')}.mp4`, f = fileOf(j, name), busy = u.state === 'running' || u.state === 'pending';
          const sim = segMeta[u.spec.k];
          return html`<div class="col" style="gap: 4px; padding: 6px; background: var(--bg2); border: 1px solid ${u.state === 'failed' && !unitStopped(u) ? 'var(--bad-line)' : '#24242A'}; border-radius: 10px">
            <div class="ph ${u.state === 'running' ? 'sk' : ''}" style="aspect-ratio: ${ratio}">
              ${f && !busy ? html`<video src="${jobFileUrl(j, name)}?v=${f.sha.slice(0, 8)}" preload="metadata" controls muted style="object-fit: cover"></video>` : ''}
              ${u.state === 'running' ? html`<span class="mono" style="position: absolute; inset: 40% 0; text-align: center; font-size: 13px; color: var(--blue-fg)">${Math.round((u.progress || 0) * 100)}%</span>` : ''}
              ${redoBtn(u, t('sw.redoSeg'), true)}</div>
            <div class="row sub" style="justify-content: space-between; gap: 4px; font-size: 11px">
              <span>${t('sw.segN', { k: u.spec.k + 1 })} · ${fsec(u.spec.start)}–${fsec(u.spec.end)}s</span>
              <span class="row" style="gap: 4px"><span class="dot ${u.state === 'done' ? 'ok' : u.state === 'failed' && !unitStopped(u) ? 'bad' : u.state === 'running' ? 'busy' : ''}"></span>${u.worker || ''}${sim != null ? ' · ' + sim : ''}</span></div></div>`;
        })}</div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${two ? t('sw.segWaitTwo') : t('sw.segWait')}</p>`}
    </section>

    <section class="card">
      <div class="row"><span class="num">3</span><h2 class="h2">${t('sw.finish')}</h2>
        <span class="sub">${t('sw.finishMeta', { face: st.face ? t('sw.facePass') : t('sw.faceOff'), up: st.up ? t('def.upscale') : t('sw.upOff') })}</span><div class="grow"></div>
        <span class="sub mono">${st.job ? stText(sPost) : ''}</span>
        ${result ? html`<button type="button" class="ib m" data-act="side" aria-pressed="${ui.side}" aria-label="${t('sw.side')}" title="${t('sw.side')}">${icon('sliders', 14)}</button>
          <button type="button" class="ib m" data-act="upLater" aria-label="${t('sw.upLater')}" title="${t('sw.upLater')}">${icon('expand', 14)}</button>` : ''}</div>
      ${result ? html`<div class="row" style="margin-top: 12px; align-items: flex-start; --pw: ${plan && plan.canvas[0] < plan.canvas[1] ? '340px' : '560px'}">
          ${ui.side && st.video ? html`<div class="col" style="gap: 4px; flex: 0 1 var(--pw); min-width: 200px"><div class="ph" style="aspect-ratio: ${ratio}"><video data-sync="a" src="${st.video.url || filesUrl(st.video.rel)}" muted controls style="object-fit: contain; background: #000"></video></div><span class="sub" style="text-align: center">${t('sw.src')}</span></div>` : ''}
          <div class="col" style="gap: 4px; flex: 0 1 var(--pw); min-width: 200px"><div class="ph" style="aspect-ratio: ${ratio}"><video data-sync="b" src="${jobFileUrl(j, 'swap.mp4')}?v=${result.sha.slice(0, 8)}" controls style="object-fit: contain; background: #000"></video></div><span class="sub" style="text-align: center">${t('sw.result')}</span></div></div>
        <div class="row" style="margin-top: 14px; justify-content: space-between">
          <p class="sub" style="margin: 0">${t('sw.saveNote')}</p>
          <button type="button" class="btn p lg" data-act="save">${icon('star', 16)}${t('sw.save')}</button></div>` :
        html`<p class="sub" style="margin: 6px 0 0 38px">${t('sw.finishWait')}</p>`}
    </section>`.__raw;
  renderActions();
  const pl = el.querySelector('[data-playlist]');
  if (pl) playlist(pl, doneSegs.map((u) => jobFileUrl(j, `seg_${String(u.spec.k).padStart(2, '0')}.mp4`)));
  const a = el.querySelector('[data-sync="a"]'), b = el.querySelector('[data-sync="b"]');
  if (a && b) { b.addEventListener('play', () => { a.currentTime = b.currentTime; a.play(); }); b.addEventListener('pause', () => a.pause()); b.addEventListener('seeked', () => { a.currentTime = b.currentTime; }); }
}

// "지금까지 이어서 보기": the finished segments in order, one after another (no join on the node)
function playlist(v, urls) {
  let k = 0;
  v.src = urls[0];
  v.onended = () => { k += 1; if (k < urls.length) { v.src = urls[k]; v.play(); } };
}

function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">
    <section class="pane-set" data-set aria-label="${t('ch.settings')}" style="max-width: 460px; flex-basis: 380px"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- data ----
async function refresh() {
  if (st.probe && !st.probe.info) await readProbe();
  if (!st.job) return;
  const j = S.jobs.find((x) => x.id === st.job);
  if (!detail || detail.id !== st.job || active(j) || (j && (detail.state !== j.state || detail.files.length !== j.files.length))) {
    try { detail = await srv('/jobs/' + st.job); } catch (e) { return; }
  }
  const jj = job();
  if (!plan && fileOf(jj, 'plan.json')) plan = await act(async () => (await fetch(jobFileUrl(jj, 'plan.json'))).json());
  for (const f of jj.files.filter((x) => /^seg_\d+\.json$/.test(x.name) && x.synced)) {
    const k = Number(f.name.slice(4, 6));
    if (segMeta[k] === undefined || segMeta['_' + k] !== f.sha) {
      const m = await act(async () => (await fetch(jobFileUrl(jj, f.name))).json());
      const sim = m && m.face_sim;   // {median, ...} (one person) or {a: {...}, b: {...}} (two)
      const v = !sim ? null : sim.median != null ? sim.median : sim.a && sim.b ? Math.min(sim.a.median, sim.b.median) : null;
      segMeta[k] = v == null ? null : Number(v).toFixed(2);
      segMeta['_' + k] = f.sha;
    }
  }
}

// a job opened by link: the left pane shows what it was made with (read-only)
function fromJob(d) {
  const o = d.opts || {};
  st = Object.assign(fresh(), { job: d.id, start: o.start ?? '', end: o.end ?? '', crop: o.crop || 'orig', face: !!o.face, up: !!o.upscale,
    seed: o.seed, refSize: o.ref_size || 'match', taller: o.taller || '', place: o.place || '',
    bg: d.inputs.background ? 'image' : o.place ? 'text' : 'keep', video: { name: o.source_name || t('sw.sourceOf', { id: d.id }), sha: d.inputs.video } });
  const a = o.asset_code || '';
  st.people = d.inputs.sheet_b ? [{ face: null, desc: o.person_a || '', to: a }, { face: null, desc: o.person_b || '', to: o.code_b || '?' }] :
    [{ face: null, desc: o.person || '', to: a }];
}

async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  switch (a.dataset.act) {
    case 'set': st[a.dataset.k] = a.dataset.v; return renderAll();
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'addPerson': st.people.push({ face: null, desc: '', to: 'keep' }); return renderSet();
    case 'delPerson': st.people.splice(Number(a.dataset.i), 1); renderSet(); return renderOut();
    case 'start': return start();
    case 'new': st = fresh(); detail = null; plan = null; segMeta = {}; ui = freshUi(); history.replaceState(null, '', '#/swap'); return renderAll();
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop': ui.askStop = false; await act(() => srv(`/jobs/${st.job}/cancel`, {})); await poll(); return;
    case 'resume': await act(() => api('/api/retry', { job: st.job }), t('jobs.resumed')); await poll(); return;
    case 'redo': await act(() => srv(`/jobs/${st.job}/redo`, { idx: Number(a.dataset.idx) })); await poll(); return;
    case 'zoom': ui.zoom = Number(a.dataset.f); return renderOut();
    case 'unzoom': ui.zoom = null; return renderOut();
    case 'playAll': ui.play = !ui.play; return renderOut();
    case 'side': ui.side = !ui.side; return renderOut();
    case 'upLater': await act(() => api('/api/upscale', { job: st.job, name: 'swap.mp4' }), t('lib.upscaleQueued', { n: 1 })); return;
    case 'save': { const r = await act(() => api('/api/save', { job: st.job, name: 'swap.mp4' })); if (r) toast(t('sw.saved', { name: r.saved })); return; }
    default:
  }
}
function onInput(e) {
  const d = e.target.dataset;
  if (d.person !== undefined && d.f === 'desc') st.people[Number(d.person)].desc = e.target.value;
  else if (d.k) st[d.k] = d.k === 'seed' ? Number(e.target.value) || 0 : e.target.value;
  else return;
  renderActions();
}
function onChange(e) {
  const d = e.target.dataset;
  if (d.file === 'video' && e.target.files[0]) return pickVideo(e.target.files[0]);
  if (d.flag) { st[d.flag] = e.target.checked; return renderSet(); }
  if (d.person !== undefined && d.f === 'to') { st.people[Number(d.person)].to = e.target.value; if (swapped().length === 2) st.bg = 'keep'; return renderSet(); }
  if (d.k === 'bgCode' || d.k === 'refSize') { st[d.k] = e.target.value; return renderSet(); }
}

export default {
  async mount(el, { query }) {
    root = el;
    if (!st) { st = fresh(); ui = freshUi(); }
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    const lib = await act(() => api('/api/library?scope=shared')) || {};
    chars = lib.characters || []; locs = lib.locations || [];
    const jid = query.get('job');
    if (jid && jid !== st.job) {
      detail = null; plan = null; segMeta = {}; ui = freshUi();
      const d = await act(() => srv('/jobs/' + jid));
      if (d) {
        fromJob(d);
        detail = d;
        const a = await act(() => api('/api/asset?sha=' + d.inputs.video));
        if (a && a.rel) st.video.rel = a.rel;
      }
    }
    if (query.get('code') && !st.job) st.people[0].to = query.get('code');
    await refresh();
    renderAll();
  },
  async update() {
    if (!root || !st) return;
    await refresh();
    if (st.probe && !st.probe.info) return;
    // a video being watched is not torn down by the 3 s refresh (the next refresh after it pauses catches up)
    if ([...root.querySelectorAll('[data-out] video')].some((v) => !v.paused && !v.ended)) return;
    renderOut();
  },
  unmount() { root = null; },
};
