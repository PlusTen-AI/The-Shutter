// 영상 생성 › 마무리 — finishing from an uploaded file (boss, session #13: the library is not kept, a finished video is
// uploaded and finished here): face pass (video.face, the characters' master + sheet) → upscale (video.upscale), in
// that order, each optional. Upscale can be turned on after the face pass is done. The draft lives on the Mac in
// projects/<p>/drafts/<id>.json (screen 'finish') until 보관함에 저장 / 닫기; #/video/finish?d=<id> reopens it.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, enter, codeLabel, codeLabelOf, isStopped, failWord } from '../app.js';
import { sideMenu, VIDEO_MENU } from './wip.js';

const SEC = { face: 5, faceLoad: 10, up: 53, upLoad: 30 };   // seconds per second of video on one node (cut.js, progress.md)

let root = null, st = null, chars = [], open = [], ui = {}, busy = false;
const freshUi = () => ({ uploading: false, askClose: false, askStop: false, drag: false });
const fresh = () => ({ id: null, screen: 'finish', title: '', video: null, codes: [], face: true, up: false, faceJob: null, upJob: null });

// ---- derived ----
const jobOf = (id) => (id && S.jobs.find((j) => j.id === id)) || null;
const live = (j) => j && (j.state === 'pending' || j.state === 'running');
const synced = (j, name) => j && j.files.find((f) => f.name === name && f.synced);
const jobs = () => [st.faceJob, st.upJob && st.upJob.job].filter(Boolean);
const started = () => !!st.faceJob || !!st.upJob;
const anyActive = () => jobs().some((id) => live(jobOf(id)));
const portrait = () => !st.video || !st.video.w || st.video.h >= st.video.w;
const ratio = () => st.video && st.video.w ? `${st.video.w} / ${st.video.h}` : '9 / 16';
const playing = () => [...root.querySelectorAll('video')].some((v) => !v.paused && !v.ended);
const faceOut = () => { const j = jobOf(st.faceJob); return synced(j, 'face.mp4') ? { job: j, name: 'face.mp4' } : null; };
const upOut = () => { const j = st.upJob && jobOf(st.upJob.job); return synced(j, 'up.mp4') ? { job: j, name: 'up.mp4' } : null; };
const result = () => (st.up ? upOut() : null) || (!st.up && st.face ? faceOut() : null);

function stage(on, id, name) {   // wait · run · done · failed · off
  if (!on) return { state: 'off' };
  const j = jobOf(id);
  if (!j) return { state: 'wait' };
  if (j.state === 'failed') return { state: 'failed', stopped: isStopped(j) };
  if (synced(j, name)) return { state: 'done' };
  return { state: 'run', pct: Math.round(j.progress * 100) };
}
function problems() {
  const p = [];
  if (!(S.status && S.status.settings && S.status.settings.project)) p.push(t('vg.needProject'));   // the result goes to the project library
  if (!st.video || !st.video.rel) p.push(t('fn.needVideo'));
  if (!st.face && !st.up) p.push(t('fn.needTodo'));
  if (st.face && !st.codes.length) p.push(t('fn.needPerson'));
  return p;
}
function estimate() {
  const sec = (st.video && st.video.sec) || 5;
  return (st.face ? SEC.faceLoad + SEC.face * sec : 0) + (st.up ? SEC.upLoad + SEC.up * sec / (sec >= 2 ? 2 : 1) : 0);   // from 2 s C·D share the upscale chunks
}
const label = () => st.title.trim() || (st.video && st.video.name) || t('side.finish');

// ---- persistence ----
async function save() {
  if (!st.id && !started()) return;
  const r = await act(() => api('/api/doc', { kind: 'drafts', id: st.id, doc: st }));
  if (r && !st.id) { st.id = r.id; history.replaceState(null, '', '#/video/finish?d=' + r.id); }
}

// ---- actions ----
async function upload(f) {
  if (!f || !f.type.startsWith('video/')) return;
  const local = URL.createObjectURL(f);
  st.video = { name: f.name, local };
  ui.uploading = true; renderAll();
  const meta = await new Promise((ok) => {   // size and length from the browser, before the upload ends
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.onloadedmetadata = () => ok({ w: v.videoWidth, h: v.videoHeight, sec: v.duration });
    v.onerror = () => ok({});
    v.src = local;
  });
  const r = await act(async () => {
    const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.status);
    return j;
  });
  ui.uploading = false;
  if (r) Object.assign(st.video, meta, { sha: r.sha, rel: 'assets/' + r.saved.split('/assets/').pop() });
  else st.video = null;
  renderAll();
}
async function start() {
  const pr = problems();
  if (pr.length) return toast(pr.join(' · '), true);
  if (st.face) {
    const r = await act(() => api('/api/face', { video: st.video.rel, codes: st.codes, label: label() }));
    if (!r) return;
    st.faceJob = r.id;
  } else {
    const r = await act(() => api('/api/upscale', { path: st.video.rel, options: { asset_kind: label() } }));
    if (!r) return;
    st.upJob = { job: r.id, from: 'source' };
  }
  await save();
  await poll();
  renderAll();
}
// upscale after the face pass (or when turned on later): from the face result, else the upload
async function next() {
  if (!st.up || st.upJob || busy || !started()) return false;
  const f = st.face ? faceOut() : null;
  if (st.face && !f) return false;
  busy = true;
  const r = await act(() => api('/api/upscale', f ? { job: f.job.id, name: f.name, options: { asset_kind: label() } } :
    { path: st.video.rel, options: { asset_kind: label() } }));
  busy = false;
  if (!r) return false;
  st.upJob = { job: r.id, from: f ? f.job.id + '/' + f.name : 'source' };
  return true;
}
async function finish(saveIt) {
  const res = result();
  const items = saveIt && res ? [{ job: res.job.id, name: res.name, title: label().replace(/\.[^.]+$/, '').replace(/\s+/g, '_') }] : [];
  const r = await act(() => api('/api/finish', { items, jobs: jobs(), docs: st.id ? [{ kind: 'drafts', id: st.id }] : [] }));
  if (!r) return;
  toast(r.saved.length ? t('vg.savedN', { n: r.saved.length }) : t('vg.closed'));
  st = fresh(); ui = freshUi();
  history.replaceState(null, '', '#/video/finish');
  await loadOpen();
  await poll();
  renderAll();
}

// ---- left ----
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const lock = started();
  const v = st.video;
  const src = v && (v.local || (v.rel && filesUrl(v.rel)));
  const addable = chars.filter((c) => c.master && !st.codes.includes(c.code));
  const pr = problems();
  el.innerHTML = html`
    <div><div class="lbl">${t('fn.crumb')}</div><h1 class="h1" style="word-break: break-all">${st.title || (v && v.name) || t('fn.new')}</h1><p class="lead">${t('lead.finish')}</p></div>
    ${!started() && open.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${open.map((d) => html`<a class="chip" href="#/video/finish?d=${d.id}" style="display: inline-flex; align-items: center; text-decoration: none">${d.title || (d.video && d.video.name) || d.id}</a>`)}</div>` : ''}
    ${lock ? html`<p class="sub" style="margin: 0">${t('sw.locked')}</p>` : ''}

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('fn.source')}</h2>
        ${v && v.w ? html`<span class="sub mono">${v.sec ? v.sec.toFixed(1) + 's · ' : ''}${v.w}×${v.h} ${portrait() ? t('sw.portrait') : t('sw.landscape')}</span>` : ''}</div>
      ${src ? html`<div class="ph" style="aspect-ratio: ${ratio()}; width: ${v.w ? `min(100%, ${Math.round(360 * v.w / v.h)}px)` : '100%'}"><video src="${src}" controls playsinline preload="metadata" style="object-fit: contain; background: #000"></video></div>
        ${lock ? '' : html`<label class="btn ghost" style="align-self: flex-start; cursor: pointer">${icon('up', 14)}${ui.uploading ? t('common.loading') : t('fn.change')}<input type="file" accept="video/*" data-file hidden></label>`}` :
        html`<label class="slot" data-drop aria-expanded="${ui.drag}" style="min-height: 180px; cursor: pointer">${icon('up', 20)}<span>${ui.uploading ? t('common.loading') : t('fn.drop')}</span>
          <input type="file" accept="video/*" data-file hidden></label>`}
      <label class="field">${t('vg.title')}<input class="in" data-k="title" value="${st.title}" placeholder="${t('vg.titlePh')}"></label></div>

    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('fn.people')}</h2><span class="sub">${t('fn.peopleNote')}</span></div>
      <div class="row" style="gap: 6px">
        ${st.codes.map((c) => html`<span class="chip" style="display: inline-flex; align-items: center; gap: 6px">${codeLabelOf(chars, c)}
          ${lock ? '' : html`<button type="button" class="ib m" data-act="delCode" data-code="${c}" aria-label="${t('vg.remove')} ${c}" title="${t('vg.remove')}" style="width: 22px; height: 22px">${icon('x', 12)}</button>`}</span>`)}
        ${st.codes.length < 2 && !lock ? html`<select class="in" data-add="code" aria-label="${t('vg.addPerson')}" style="width: 150px; min-height: 34px">
          <option value="">${t('vg.addPerson')}</option>${addable.map((c) => html`<option value="${c.code}">${codeLabel(c)}</option>`)}</select>` : ''}</div></div>

    <div class="sec">
      <h2 class="h3">${t('fn.todo')}</h2>
      <label class="tog"><input type="checkbox" data-flag="face" ${st.face ? 'checked' : ''} ${lock ? 'disabled' : ''}>${t('fn.face')}
        <span class="sub">${st.codes.length ? t('fn.faceNote') : t('vg.faceNeed')}</span></label>
      <label class="tog"><input type="checkbox" data-flag="up" ${st.up ? 'checked' : ''} ${st.upJob ? 'disabled' : ''}>${t('def.upscale')}
        <span class="sub">${portrait() ? '1080×1920' : '1920×1080'} · ${t('vg.upNote')}</span></label></div>
`.__raw;
  renderActions();
}
// the main action under the progress pills, right (boss, session #14): the note (what is missing · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const pr = problems();
  el.innerHTML = started() ? '' : html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : t('fn.estimate', { t: fmtDur(estimate()) })}</span>
    <button type="button" class="btn p" data-act="start" ${pr.length || ui.uploading ? 'disabled' : ''}>${t('ch.start')}</button>`.__raw;
}

// ---- right ----
const stText = (s) => ({ wait: t('ch.st.wait'), run: (s.pct || 0) + '%', done: t('ch.st.done'), failed: failWord(s.stopped), off: t('fn.off') }[s.state]);
const stDot = (s) => ({ run: 'busy', done: 'ok', failed: s.stopped ? '' : 'bad' }[s.state] || '');

function videoBox(r, caption) {
  return html`<div class="col ${enter('fn:' + r.job.id + '/' + r.name)}" style="gap: 4px; flex: 0 1 ${portrait() ? '320px' : '560px'}; min-width: 200px">
    <div class="ph" style="aspect-ratio: ${ratio()}"><video src="${jobFileUrl(r.job, r.name)}" controls playsinline preload="metadata" style="object-fit: contain; background: #000"></video></div>
    <span class="sub" style="text-align: center">${caption}</span></div>`;
}
function card(n, title, s, meta, body) {
  return html`<section class="card" style="${s.state === 'off' ? 'opacity: .55' : ''}">
    <div class="row"><span class="num">${n}</span><h2 class="h2">${title}</h2><span class="sub">${meta}</span><div class="grow"></div>
      <span class="sub mono">${started() || s.state === 'off' ? stText(s) : ''}</span></div>
    ${s.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${s.pct}%"></div></div>` : ''}
    ${body}</section>`;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const fs = stage(st.face, st.faceJob, 'face.mp4');
  const us = stage(st.up, st.upJob && st.upJob.job, 'up.mp4');
  const pills = [[t('fn.face'), fs], [t('def.upscale'), us]];
  const failed = [fs, us].filter((x) => x.state === 'failed').length;
  const fo = faceOut(), uo = upOut(), res = result();
  const sc = ui.score;
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px" aria-label="${t('ch.progress')}">${pills.map(([lb, x]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px; ${x.state === 'off' ? 'opacity: .55' : ''}">
          <span class="dot ${stDot(x)}"></span>${lb}<span class="sub mono">${stText(x)}</span></li>`)}</ol>
      <div class="toprow-end">${anyActive() ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.askStop')}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${failed ? html`<div class="row banner ${[fs, us].filter((x) => x.state === 'failed').every((x) => x.stopped) ? 'warn' : 'bad'}" role="status" style="justify-content: space-between"><span>${t([fs, us].filter((x) => x.state === 'failed').every((x) => x.stopped) ? 'vg.stoppedN' : 'vg.failedN', { n: failed })}</span>
      <button type="button" class="btn p" data-act="resume" style="min-height: 34px">${icon('play', 12)}${t('jobs.resume')}</button></div>` : ''}

    ${card(1, t('fn.face'), fs, st.face ? st.codes.join(' · ') : '', fo ? html`<div class="row" style="margin-top: 12px; align-items: flex-start">
      ${videoBox(fo, sc ? t('fn.score', { a: sc.a, b: sc.b }) : t('fn.face'))}</div>` : '')}
    ${card(2, t('def.upscale'), us, st.up ? (portrait() ? '1080×1920' : '1920×1080') : '', uo ? html`<div class="row" style="margin-top: 12px; align-items: flex-start">
      ${videoBox(uo, t('vg.upscaled'))}</div>` : st.up && st.face && !fo && started() ? html`<p class="sub" style="margin: 6px 0 0 38px">${t('fn.wait')}</p>` : '')}

    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('fn.askClose')}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button>
        <button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between; padding-top: 4px">
        <p class="sub" style="margin: 0">${t('fn.saveNote')}</p>
        <div class="row" style="gap: 8px">
          <button type="button" class="btn ghost" data-act="close" ${anyActive() ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="save" ${res && !anyActive() ? '' : 'disabled'}>${icon('star', 16)}${t('sw.save')}</button></div></div>` : ''}`.__raw;
  renderActions();
}
function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(VIDEO_MENU(), '#/video/finish')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// face.json → the same-person score before → after (median over frames)
async function loadScore() {
  const j = jobOf(st.faceJob);
  if (ui.score || !synced(j, 'face.json')) return;
  const f = await act(async () => (await fetch(jobFileUrl(j, 'face.json'))).json());
  const m = (x) => x && x.median != null ? x.median.toFixed(2) : '—';
  if (f) ui.score = { a: m(f.before), b: m(f.after) };
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const d = a.dataset;
  switch (d.act) {
    case 'delCode': st.codes = st.codes.filter((c) => c !== d.code); renderSet(); return save();
    case 'start': return start();
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop':
      ui.askStop = false;
      for (const id of jobs().filter((x) => live(jobOf(x)))) await act(() => srv(`/jobs/${id}/cancel`, {}));
      await poll(); return;
    case 'resume':
      for (const id of jobs().filter((x) => { const j = jobOf(x); return j && j.state === 'failed'; })) await act(() => api('/api/retry', { job: id }));
      await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; return finish(false);
    case 'save': return finish(true);
    default:
  }
}
function onInput(e) {
  if (e.target.dataset.k !== 'title') return;
  st.title = e.target.value;
  const h = root.querySelector('[data-set] h1');
  if (h) h.textContent = st.title || (st.video && st.video.name) || t('fn.new');
  save();
}
async function onChange(e) {
  const dd = e.target.dataset;
  if ('file' in dd && e.target.files[0]) return upload(e.target.files[0]);
  if (dd.add === 'code' && e.target.value) { st.codes = st.codes.concat([e.target.value]).slice(0, 2); renderSet(); return save(); }
  if (dd.flag) {
    st[dd.flag] = e.target.checked;
    renderSet(); renderOut();
    if (dd.flag === 'up' && st.up && started() && await next()) { await poll(); renderOut(); }   // 업스케일 later
    return save();
  }
}
function onDrag(e) {
  if (!e.target.closest('[data-drop]')) return;
  e.preventDefault();
  const over = e.type === 'dragover' || e.type === 'dragenter';
  if (ui.drag !== over) { ui.drag = over; const z = root.querySelector('[data-drop]'); if (z) z.setAttribute('aria-expanded', String(over)); }
  if (e.type === 'drop' && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
}
async function loadOpen() {
  open = (await act(() => api('/api/docs?kind=drafts')) || []).filter((d) => d && d.screen === 'finish' && d.id !== st.id);
}

export default {
  async mount(el, { query }) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    for (const ev of ['dragenter', 'dragover', 'dragleave', 'drop']) el.addEventListener(ev, onDrag);
    chars = ((await act(() => api('/api/library?scope=shared'))) || {}).characters || [];
    const id = query.get('d');
    if (id) {
      const d = await act(() => api('/api/doc?kind=drafts&id=' + encodeURIComponent(id)));
      st = d && d.id ? Object.assign(fresh(), d) : fresh();
      ui = freshUi();
    }
    if (!st) { st = fresh(); ui = freshUi(); }
    if (st.video && st.video.local && !ui.uploading) delete st.video.local;   // an object URL lives only in its page load
    await loadOpen();
    renderAll();
  },
  async update() {
    if (!root || !st) return;
    await loadScore();
    if (await next()) { await save(); await poll(); }
    if (String(started()) !== root.dataset.started) { root.dataset.started = String(started()); renderSet(); }
    if (playing()) return;   // a video being watched is not torn down by the 3 s refresh
    renderOut();
  },
  unmount() { root = null; },
};
