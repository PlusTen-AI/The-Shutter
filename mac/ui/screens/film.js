// 영상 생성 › 워크플로 › 한 편 (design §12 A, 기능 4~7): every sequence's exported parts (one per episode stretch,
// video.edit) in order → one film by stream copy (video.join). Upscale is all or nothing (boss): "모든 조각 업스케일"
// upscales each part (video.upscale, C·D share the chunks) and the film then takes the upscaled ones; a mix of sizes
// is warned about (기능 7) and the odd parts are re-encoded by the join. 화 끊기 (off by default, boss): a white flash
// at each episode break + one file per episode. The film's state lives in projects/<p>/work/films/film.json.
import { S, t, html, icon, api, act, toast, poll, filesUrl, jobFileUrl, fmtDur, isStopped, failWord } from '../app.js';
import { sideMenu, VIDEO_MENU } from './wip.js';

const FLASH = 0.25;   // seconds of white at a break
let root = null, st = null, seqs = [], ui = {};
const fresh = () => ({ id: 'film', breaks: false, up: false, skip: [], ups: {}, join: null });

// ---- derived ----
const jobOf = (id) => (id && S.jobs.find((j) => j.id === id)) || null;
const synced = (j, name) => j && j.files.find((f) => f.name === name && f.synced);
const project = () => S.status && S.status.settings && S.status.settings.project;
const order = (a, b) => ((a.script && a.script.scene) || 1e9) - ((b.script && b.script.scene) || 1e9) || String(a.name).localeCompare(String(b.name));
// a sequence's exported parts in the library: [{path, name, brk}] (an export from before parts = one part)
function partsOf(sq) {
  const ex = sq.export;
  if (!ex || !ex.saved) return [];
  const names = Array.isArray(ex.saved) ? ex.saved : [ex.saved];
  return names.map((n, i) => ({ path: `projects/${project()}/videos/${n}`, name: n, brk: !!((ex.parts || [])[i] || {}).brk }));
}
const used = () => seqs.filter((q) => !st.skip.includes(q.id) && partsOf(q).length);
const allParts = () => used().flatMap((q) => partsOf(q).map((p) => Object.assign({ seq: q }, p)));
const upOf = (p) => { const j = jobOf(st.ups[p.path]); return synced(j, 'up.mp4') ? { job: j.id, name: 'up.mp4', j } : null; };
const upState = (p) => { const j = jobOf(st.ups[p.path]); return !st.ups[p.path] ? 'wait' : !j ? 'gone' : j.state === 'failed' ? 'failed' : upOf(p) ? 'done' : 'run'; };
// what goes into the join: the upscaled part when 업스케일 is on and it is there, else the part itself
const joinItems = () => allParts().map((p) => (st.up && upOf(p) ? { job: upOf(p).job, name: 'up.mp4' } : { path: p.path }));
const mixed = () => st.up && allParts().some((p) => !upOf(p));
const joinKey = () => JSON.stringify([joinItems(), st.breaks]);
const outFiles = (j) => j ? j.files.filter((f) => /^(film|ep_\d+)\.mp4$/.test(f.name)).sort((a, b) => a.name < b.name ? -1 : 1) : [];
function problems() {
  const p = [];
  if (!project()) p.push(t('vg.needProject'));
  if (!allParts().length) p.push(t('fm.needParts'));
  return p;
}

// ---- persistence / actions ----
async function save() { await act(() => api('/api/doc', { kind: 'films', id: 'film', doc: st })); }
async function upscaleAll() {
  for (const p of allParts()) {
    if (['run', 'done'].includes(upState(p))) continue;
    const r = await act(() => api('/api/upscale', { path: p.path, options: { asset_kind: p.name } }));
    if (!r) break;
    st.ups[p.path] = r.id;
  }
  await save();
  await poll();
  renderAll();
}
async function joinGo() {
  const items = joinItems();
  let k = 0;
  const breaks = [];
  for (const p of allParts()) { if (st.breaks && p.brk) breaks.push(k); k += 1; }
  const r = await act(() => api('/api/join', { parts: items, breaks, flash: st.breaks ? FLASH : 0, label: `${project()} · ${t('side.film')}` }));
  if (!r) return;
  st.join = { job: r.id, key: joinKey(), saved: null };
  await save();
  await poll();
  renderAll();
}

// ---- render ----
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  el.innerHTML = html`
    <div><div class="lbl">${t('fm.crumb')}</div><h1 class="h1">${t('side.film')}</h1><p class="lead">${t('lead.film')}</p></div>
    <div class="sec">
      <h2 class="h3">${t('fm.opts')}</h2>
      <label class="tog"><input type="checkbox" data-flag="breaks" ${st.breaks ? 'checked' : ''}>${t('fm.breaks')} <span class="sub">${t('fm.breaksNote', { s: FLASH })}</span></label>
      <label class="tog"><input type="checkbox" data-flag="up" ${st.up ? 'checked' : ''}>${t('fm.up')} <span class="sub">${t('fm.upNote')}</span></label>
      ${st.up ? html`<div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="upAll" ${allParts().some((p) => !['run', 'done'].includes(upState(p))) ? '' : 'disabled'}>${t('fm.upAll', { n: allParts().filter((p) => !['run', 'done'].includes(upState(p))).length })}</button>
        <span class="sub">${t('fm.upDone', { n: allParts().filter((p) => upOf(p)).length, all: allParts().length })}</span></div>` : ''}</div>
    <div class="sec"><span class="sub">${t('fm.howNote')}</span></div>`.__raw;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const pr = problems();
  const j = st.join && jobOf(st.join.job);
  const live = j && (j.state === 'pending' || j.state === 'running');
  const files = outFiles(j).filter((f) => f.synced);
  const done = j && j.state === 'done' && files.length && files.length === outFiles(j).length;
  const old = st.join && st.join.key !== joinKey();
  const total = used().reduce((a, q) => a + Number(q.sec || 0), 0);
  const eps = allParts().slice(0, -1).filter((p) => p.brk).length + 1;   // a break after the last part ends nothing
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <span class="sub mono">${t('fm.meta', { n: used().length, p: allParts().length, e: st.breaks ? eps : 1 })}</span>
      <div class="toprow-end"><div class="topact"><span class="note">${pr.length ? pr[0] : live ? t('fm.joining', { p: Math.round(j.progress * 100) }) : ''}</span>
        <button type="button" class="btn p" data-act="join" ${pr.length || live || (done && !old) ? 'disabled' : ''}>${t('fm.joinGo')}</button></div></div></div>
    ${mixed() ? html`<div class="row banner warn" role="status"><span>${t('fm.mixed', { n: allParts().filter((p) => !upOf(p)).length })}</span></div>` : ''}
    ${j && j.state === 'failed' ? html`<div class="row banner bad" role="status"><span>${failWord(isStopped(j))} · ${(j.errors || []).slice(-1)[0] || ''}</span></div>` : ''}
    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${t('fm.seqs')}</h2><span class="sub">${t('fm.seqsNote')}</span></div>
      ${seqs.length ? html`<div class="list" style="margin-top: 12px">${seqs.map((q) => {
        const ps = partsOf(q), on = !st.skip.includes(q.id);
        return html`<div class="item-row" style="gap: 10px; ${on && ps.length ? '' : 'opacity: .55'}">
          <input type="checkbox" data-seq="${q.id}" ${on ? 'checked' : ''} ${ps.length ? '' : 'disabled'} aria-label="${q.name || q.id}">
          <a href="#/video/seq?id=${q.id}" style="flex: 1 1 200px; min-width: 0">${q.name || t('sq.untitled')}</a>
          <span class="mono sub">${q.script ? t('fm.scene', { n: q.script.scene }) + ' · ' : ''}${t('sq.meta', { n: (q.cuts || []).length, s: Number(q.sec || 0).toFixed(1) })}</span>
          <span class="row" style="gap: 6px">${ps.length ? ps.map((p, i) => html`<span class="chip mono" title="${p.name}">${i + 1}${p.brk ? ' ▷' : ''}${st.up ? ' · ' + t('fm.up.' + upState(p)) : ''}</span>`) : html`<span class="sub">${t('fm.noExport')}</span>`}</span></div>`;
      })}</div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${t('fm.noSeqs')}</p>`}
    </section>
    <section class="card">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('fm.result')}</h2><span class="sub mono">${total ? fmtDur(total) : ''}</span><div class="grow"></div>
        <span class="sub mono">${j ? (j.state === 'failed' ? failWord(isStopped(j)) : done ? t('ch.st.done') : Math.round(j.progress * 100) + '%') : ''}</span></div>
      ${done ? html`${old ? html`<p class="sub" style="color: var(--warn-fg)">${t('fm.old')}</p>` : ''}
        <div class="row" style="margin-top: 12px; gap: 12px; align-items: flex-start">${files.map((f) => html`
          <div class="col" style="gap: 4px; flex: 0 1 ${f.name === 'film.mp4' ? '300px' : '200px'}">
            <div class="ph" style="aspect-ratio: 9 / 16"><video src="${jobFileUrl(j, f.name)}?v=${f.sha.slice(0, 8)}" controls preload="metadata" style="object-fit: contain; background: #000"></video></div>
            <span class="sub" style="text-align: center">${f.name === 'film.mp4' ? t('fm.film') : t('fm.ep', { n: Number(f.name.slice(3, 5)) })}${st.join.saved ? ' · ' + t('fm.saved') : ''}</span></div>`)}</div>` :
        html`<p class="sub" style="margin: 6px 0 0 38px">${t('fm.resultNote')}</p>`}
    </section>`.__raw;
}
function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(VIDEO_MENU(), '#/video/film')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  if (a.dataset.act === 'upAll') return upscaleAll();
  if (a.dataset.act === 'join') return joinGo();
}
async function onChange(e) {
  const d = e.target.dataset;
  if (d.flag) { st[d.flag] = e.target.checked; await save(); return renderAll(); }
  if (d.seq) {
    st.skip = e.target.checked ? st.skip.filter((x) => x !== d.seq) : st.skip.concat([d.seq]);
    await save();
    return renderAll();
  }
}

async function load() {
  st = fresh();
  if (!project()) { seqs = []; return; }
  const d = await act(() => api('/api/doc?kind=films&id=film'));
  if (d && d.id) st = Object.assign(fresh(), d);
  seqs = ((await act(() => api('/api/docs?kind=sequences'))) || []).filter((q) => q && q.id);
  // a sequence's running time: its cuts' used spans (the cut docs)
  const cuts = ((await act(() => api('/api/docs?kind=cuts'))) || []).filter(Boolean);
  for (const q of seqs) {
    q.sec = (q.cuts || []).map((id) => cuts.find((c) => c.id === id)).filter((c) => c && !c.genOnly)
      .reduce((a, c) => a + (c.use ? Number(c.use[1]) - Number(c.use[0]) : Number(c.sec) || 0), 0);
  }
  seqs.sort(order);
}

export default {
  async mount(el) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('change', onChange);
    await load();
    renderAll();
  },
  async update() {
    if (!root || !st) return;
    const j = st.join && jobOf(st.join.job);
    if (j && j.state === 'done' && !st.join.saved && outFiles(j).length && outFiles(j).every((f) => f.synced)) {   // to the library once
      const saved = [];
      for (const f of outFiles(j)) {
        const r = await act(() => api('/api/save', { job: j.id, name: f.name, title: `${project()}_${f.name.slice(0, -4)}` }));
        if (!r) return;
        saved.push(r.saved);
      }
      st.join.saved = saved;
      await save();
    }
    if ([...root.querySelectorAll('video')].some((v) => !v.paused && !v.ended)) return;
    renderAll();
  },
  unmount() { root = null; },
};
