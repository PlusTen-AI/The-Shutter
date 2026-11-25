// 이미지 › 생성 (image.t2i) / 편집 (image.edit) — two panes (design §11): settings left, candidates right. Candidates:
// ☆ save many (→ this project's 장면 on 보관함에 저장 / 닫기), "편집에 쓰기" puts one into 편집 as image 1. The draft
// lives on the Mac in projects/<p>/drafts/<id>.json until 보관함에 저장 / 닫기; #/image/gen?d=<id> reopens it.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, enter, isStopped, failWord, failColor } from '../app.js';
import { defaults, cfgFields, cfgInput, cfgOpts, cfgX } from '../models.js';
import { sideMenu, IMAGE_MENU } from './wip.js';

const ASPECTS = ['1:1', '2:3', '3:2', '9:16', '16:9'];
const tileRatio = () => st.aspect === 'src' ? '1' : st.aspect.replace(':', ' / ');   // tiles in the picked shape
const SEC = { 1: 19, 4: 111, edit: 40, load: 30 };   // seconds per image on one node (production, session #10)
const MAX_REFS = 3;

let root = null, screen = 'gen', st = null, lib = { scenes: [], chars: [], locs: [] }, open = [], ui = {};
const drafts = {};   // screen → the draft in memory
const freshUi = () => ({ adv: false, pick: false, askClose: false, uploading: false });

function fresh(scr) {
  const d = defaults('qi21');
  return { id: null, screen: scr, title: '', prompt: '', refs: [], aspect: scr === 'edit' ? 'src' : '2:3', mp: d.mp,
    cands: d.cands, seed: 1, steps: d.steps, cfg: d.cfg, neg: d.neg, jobs: [], saved: [] };
}

// ---- derived ----
const jobOf = (id) => S.jobs.find((j) => j.id === id) || null;
const key = () => JSON.stringify([st.prompt.trim(), st.refs.map((r) => r.rel), st.aspect, Number(st.mp), Number(st.steps), cfgOpts(st)]);
const started = () => st.jobs.length > 0;
const active = () => st.jobs.map((x) => jobOf(x.job)).some((j) => j && (j.state === 'pending' || j.state === 'running'));
const rel = (j, name) => (j.project ? `projects/${j.project}/jobs/${j.id}` : `jobs/${j.id}`) + '/' + name;
const isSaved = (c) => st.saved.some((x) => x.job === c.job && x.name === c.name);

function cands() {
  const out = [];
  for (const x of st.jobs) {
    const j = jobOf(x.job);
    if (!j) continue;
    const old = x.key !== key();
    for (const f of j.files.filter((y) => y.name.endsWith('.png') && y.synced)) {
      out.push({ job: j.id, name: f.name, url: jobFileUrl(j, f.name) + '?v=' + f.sha.slice(0, 8), rel: rel(j, f.name), seed: f.name.split('_')[1].split('.')[0], old });
    }
    for (let k = 0; k < j.count.pending + j.count.running; k++) out.push({ pending: true, run: j.count.running > k, pct: Math.round(j.progress * 100), old });
    for (let k = 0; k < j.count.failed; k++) out.push({ failed: true, stopped: isStopped(j), job: j.id, old });
  }
  return out;
}
function problems() {
  const p = [];
  if (!(S.status && S.status.settings && S.status.settings.project)) p.push(t('vg.needProject'));   // cuts and drafts live in the project
  if (screen === 'edit' && !st.refs.length) p.push(t('ig.needImage'));
  if (!st.prompt.trim()) p.push(t(screen === 'edit' ? 'ig.needInstr' : 'vg.needPrompt'));
  return p;
}
function estimate(n) {
  const per = screen === 'edit' && st.aspect === 'src' ? SEC.edit : SEC[Number(st.mp) > 2 ? 4 : 1] + (screen === 'edit' ? 20 : 0);
  return SEC.load + Math.ceil(n / 2) * per * cfgX(st);
}

// ---- persistence ----
async function save() {
  if (!st.id && !started()) return;
  const r = await act(() => api('/api/doc', { kind: 'drafts', id: st.id, doc: st }));
  if (r && !st.id) { st.id = r.id; history.replaceState(null, '', `#/image/${screen}?d=${r.id}`); }
}
let saveTimer = 0;
const saveSoon = () => { clearTimeout(saveTimer); if (st.id) saveTimer = setTimeout(save, 800); };

// ---- actions ----
async function generate() {
  const pr = problems();
  if (pr.length) return toast(pr[0], true);
  const k = key();
  const used = st.jobs.filter((x) => x.key === k).reduce((a, x) => a + (x.n || 0), 0);
  const o = { prompt: st.prompt.trim(), count: st.cands, seed: Number(st.seed) + used, steps: st.steps,
    asset_kind: st.title.trim() || t('side.' + screen) };
  if (st.aspect !== 'src') Object.assign(o, { aspect: st.aspect, mp: st.mp });
  Object.assign(o, cfgOpts(st));
  const r = await act(() => api('/api/image', { type: screen === 'edit' ? 'image.edit' : 'image.t2i', images: st.refs.map((x) => x.rel), options: o }));
  if (!r) return;
  st.jobs.push({ job: r.id, key: k, n: Number(st.cands) });
  await save();
  await poll();
  renderAll();
}
async function upload(f) {
  ui.uploading = true; renderSet();
  const r = await act(async () => {
    const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.status);
    return j;
  });
  ui.uploading = false;
  if (r && st.refs.length < MAX_REFS) { st.refs.push({ rel: 'assets/' + r.saved.split('/assets/').pop(), name: f.name }); ui.pick = false; saveSoon(); }
  renderAll();
}
async function finish(saveIt) {
  const title = (st.title.trim() || st.prompt.trim().slice(0, 24) || screen).replace(/\s+/g, '_');
  const items = st.saved.filter((c) => jobOf(c.job)).map((c, i) => ({ job: c.job, name: c.name, title: title + '_' + (i + 1) }));
  const r = await act(() => api('/api/finish', { items, jobs: st.jobs.map((x) => x.job), docs: st.id ? [{ kind: 'drafts', id: st.id }] : [] }));
  if (!r) return;
  toast(r.saved.length ? t('vg.savedN', { n: r.saved.length }) : t('vg.closed'));
  st = drafts[screen] = fresh(screen); ui = freshUi();
  history.replaceState(null, '', '#/image/' + screen);
  await loadOpen();
  await poll();
  renderAll();
}

// ---- left ----
function seg(k, opts, cur, label, cls) {
  return html`<div class="seg ${cls || ''}" role="group" aria-label="${label}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${k}" data-v="${v}" aria-pressed="${String(cur) === String(v)}">${l}</button>`)}</div>`;
}
export function libraryPicker(l, title) {   // shared with the scene screen: project scenes, character images, location anchors
  const group = (name, items) => items.length ? html`<div class="col" style="gap: 6px"><span class="sub">${name}</span>
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); gap: 6px">${items.map((x) => html`
      <button type="button" class="ph" data-act="libPick" data-rel="${x.rel}" data-name="${x.name}" title="${x.name}" style="padding: 0; aspect-ratio: 1; cursor: pointer">
        <img src="${filesUrl(x.rel)}" alt="${x.name}" loading="lazy"></button>`)}</div></div>` : '';
  const chars = l.chars.flatMap((c) => c.images.filter((i) => ['ref', 'sheet', 'sh_head_front', 'sh_full_front'].includes(i.kind) || i.kind.startsWith('ex_'))
    .filter((i, _, a) => a.filter((y) => y.kind === i.kind).every((y) => y.v <= i.v)).map((i) => ({ rel: `characters/${c.code}/${i.name}`, name: i.name })));
  return html`<div class="col" style="gap: 10px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px; max-height: 460px; overflow: auto">
    <div class="row" style="justify-content: space-between; flex-wrap: nowrap"><span class="lbl">${title}</span>
      <button type="button" class="ib m" data-act="unpick" aria-label="${t('common.cancel')}" title="${t('common.cancel')}">${icon('x', 14)}</button></div>
    <label class="btn ghost" style="align-self: flex-start; cursor: pointer">${icon('up', 14)}${t('vg.upload')}<input type="file" accept="image/*" data-file="ref" hidden></label>
    ${group(t('lib.kind.scenes'), l.scenes.map((i) => ({ rel: i.path, name: i.name })))}
    ${group(t('lib.kind.characters'), chars)}
    ${group(t('lib.kind.locations'), l.locs.filter((x) => x.master).map((x) => ({ rel: `locations/${x.code}/${x.master}`, name: x.code })))}</div>`;
}
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const edit = screen === 'edit';
  const n = Number(st.cands);
  const pr = problems();
  el.innerHTML = html`
    <div><div class="lbl">${t('ig.crumb')}</div><h1 class="h1">${st.title || t('side.' + screen)}</h1><p class="lead">${t('lead.' + screen)}</p></div>
    ${!started() && open.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${open.map((d) => html`<a class="chip" href="#/image/${screen}?d=${d.id}" style="display: inline-flex; align-items: center; text-decoration: none">${d.title || d.prompt.slice(0, 14) || d.id}</a>`)}</div>` : ''}
    <div class="sec">
      <label class="field">${t('vg.title')}<input class="in" data-k="title" value="${st.title}" placeholder="${t('ig.titlePh')}"></label>
      ${edit ? html`<div class="field">${t('ig.images')} <span class="sub">${t('ig.imagesNote')}</span>
        <div class="row" style="gap: 8px; align-items: stretch">${st.refs.map((r, i) => html`
          <div class="ph" style="width: 96px; height: 96px"><img src="${filesUrl(r.rel)}" alt="${r.name}" style="object-fit: contain">
            <span class="mono" style="position: absolute; left: 4px; top: 4px; padding: 0 6px; border-radius: 5px; background: #0E0E10CC; font-size: 11px">${i + 1}</span>
            <button type="button" class="ib" data-act="delRef" data-i="${i}" aria-label="${t('vg.remove')}" title="${t('vg.remove')}" style="position: absolute; right: 3px; top: 3px; width: 24px; height: 24px; background: #16161AE6">${icon('x', 12)}</button></div>`)}
          ${st.refs.length < MAX_REFS ? html`<button type="button" class="slot" data-act="pick" aria-expanded="${ui.pick}" style="width: 96px; height: 96px">${icon('plus', 16)}<span>${ui.uploading ? t('common.loading') : t('vg.fromLib')}</span></button>` : ''}</div></div>
        ${ui.pick ? libraryPicker(lib, t('ig.pickImage', { n: st.refs.length + 1 })) : ''}` : ''}
      <label class="field">${edit ? t('ig.instruction') : t('vg.prompt')}<textarea class="in" rows="6" data-k="prompt" placeholder="${edit ? t('ig.editPh') : t('ig.genPh')}">${st.prompt}</textarea></label></div>
    <div class="sec">
      <div class="field">${t('ig.size')}${seg('aspect', (edit ? [['src', t('ig.srcSize')]] : []).concat(ASPECTS.map((a) => [a, a])), st.aspect, t('ig.size'), 'fill')}</div>
      ${st.aspect !== 'src' ? html`<div class="field">${t('def.mp')}${seg('mp', [[1, t('def.mp.1')], [4, t('def.mp.4')]], st.mp, t('def.mp'))}</div>` : html`<span class="sub">${t('ig.srcNote')}</span>`}
      <div class="row" style="gap: 10px"><span class="lbl">${t('ch.cands')}</span>${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}<span class="sub">${t('ch.candsNote')}</span>
        <div class="grow"></div><button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      ${ui.adv ? html`<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('ch.seed')}<input class="in mono" data-k="seed" inputmode="numeric" value="${st.seed}"></label>
        <label class="field">${t('def.steps')}<input class="in mono" data-k="steps" inputmode="numeric" value="${st.steps}"></label>
        ${cfgFields(st)}</div>` : ''}</div>`.__raw;
  renderActions();
}
// the main action, top right of the results pane (boss, session #14): the note (what is missing · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const pr = problems(), n = Number(st.cands);
  el.innerHTML = html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : t('vg.estimate', { t: fmtDur(estimate(n)), n })}</span>
    <button type="button" class="btn p" data-act="go" ${pr.length || ui.uploading ? 'disabled' : ''}>${started() ? t('ig.more', { n }) : t('ch.start')}</button>`.__raw;
}

// ---- right ----
function tile(c, i) {
  if (c.pending) return html`<div class="ph sk" style="aspect-ratio: ${tileRatio()}; display: flex; align-items: center; justify-content: center; ${c.old ? 'opacity: .45' : ''}">
    <span class="mono sub">${c.run ? c.pct + '%' : t('ch.st.wait')}</span></div>`;
  if (c.failed) return html`<div class="ph" style="aspect-ratio: ${tileRatio()}; display: flex; align-items: center; justify-content: center"><span class="sub" style="color: ${failColor(c.stopped)}">${failWord(c.stopped)}</span></div>`;
  const sv = isSaved(c);
  return html`<div class="col ${enter('ig:' + c.job + '/' + c.name)}" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px; ${c.old ? 'opacity: .45' : ''}">
    <a class="ph" href="${c.url}" target="_blank" style="display: block; aspect-ratio: ${tileRatio()}"><img src="${c.url}" alt="" loading="lazy" style="object-fit: contain">
      <span class="mono" style="position: absolute; left: 6px; top: 6px; padding: 1px 7px; border-radius: 6px; background: #0E0E10CC; font-size: 11px; color: var(--fg2)">seed ${c.seed}${c.old ? ' · ' + t('vg.st.old') : ''}</span></a>
    <div class="row" style="gap: 2px">
      <button type="button" class="ib m" data-act="star" data-i="${i}" aria-pressed="${sv}" aria-label="${t('ch.saveCand')}" title="${t('ch.saveCand')}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="${sv ? '#FACC15' : 'none'}" stroke="${sv ? '#FACC15' : 'currentColor'}" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg></button>
      <button type="button" class="ib m" data-act="toEdit" data-i="${i}" aria-label="${t('ig.toEdit')}" title="${t('ig.toEdit')}">${icon('edit', 14)}</button></div></div>`;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const cs = cands();
  const run = cs.some((c) => c.pending);
  const failed = st.jobs.map((x) => jobOf(x.job)).filter((j) => j && j.state === 'failed');
  const unsaved = cs.filter((c) => c.url && !isSaved(c)).length;
  const runs = cs.filter((c) => c.pending);
  const pct = runs.length ? Math.round(runs.reduce((a, c) => a + (c.run ? c.pct : 0), 0) / runs.length) : 0;
  const stage = t(screen === 'edit' ? 'side.edit' : 'vg.gen');
  el.innerHTML = html`
    <div class="row" style="min-height: 40px; justify-content: space-between">
      <span class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
        <span class="dot ${run ? 'busy' : started() ? 'ok' : ''}"></span>${stage}<span class="sub mono">${run ? pct + '%' : started() ? t('ch.st.done') : t('ch.st.wait')}</span></span>
      <div class="toprow-end">${active() ? html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>` : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${failed.length ? html`<div class="row banner ${failed.every(isStopped) ? 'warn' : 'bad'}" role="status" style="justify-content: space-between"><span>${t(failed.every(isStopped) ? 'vg.stoppedN' : 'vg.failedN', { n: failed.length })}</span>
      <button type="button" class="btn p" data-act="resume" style="min-height: 34px">${icon('play', 12)}${t('jobs.resume')}</button></div>` : ''}
    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${stage}</h2><span class="sub mono">Qwen-Image 2.1 · ${t('ch.candN', { n: cs.filter((c) => c.url && !c.old).length })}</span></div>
      <p class="sub" style="margin: 6px 0 0 38px">${t('ig.rule')}</p>
      ${cs.length ? html`<div class="grid" style="margin-top: 14px; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr))">${cs.map(tile)}</div>` : ''}
    </section>
    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('vg.askClose', { n: unsaved })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button><button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between"><p class="sub" style="margin: 0">${t('ig.saveNote')}</p>
        <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="close" ${active() ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="save" ${st.saved.length && !active() ? '' : 'disabled'}>${icon('star', 16)}${t('ig.saveN', { n: st.saved.length })}</button></div></div>` : ''}`.__raw;
  renderActions();
}
function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(IMAGE_MENU(), '#/image/' + screen)}
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
    case 'set': st[d.k] = ['cands', 'mp'].includes(d.k) ? Number(d.v) : d.v; renderSet(); renderOut(); return saveSoon();
    case 'pick': ui.pick = !ui.pick; return renderSet();
    case 'unpick': ui.pick = false; return renderSet();
    case 'libPick': if (st.refs.length < MAX_REFS) st.refs.push({ rel: d.rel, name: d.name }); ui.pick = false; renderSet(); renderOut(); return saveSoon();
    case 'delRef': st.refs.splice(Number(d.i), 1); renderSet(); renderOut(); return saveSoon();
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'go': return generate();
    case 'star': {
      const c = cands()[Number(d.i)];
      st.saved = isSaved(c) ? st.saved.filter((x) => !(x.job === c.job && x.name === c.name)) : st.saved.concat([{ job: c.job, name: c.name }]);
      renderOut(); return save();
    }
    case 'toEdit': {   // the candidate becomes image 1 of 편집 (and is kept: ☆)
      const c = cands()[Number(d.i)];
      if (!isSaved(c)) { st.saved.push({ job: c.job, name: c.name }); await save(); }
      const ed = drafts.edit && !drafts.edit.jobs.length ? drafts.edit : (drafts.edit = fresh('edit'));
      ed.refs = [{ rel: c.rel, name: c.name }].concat(ed.refs.filter((r) => r.rel !== c.rel)).slice(0, MAX_REFS);
      location.hash = '#/image/edit';
      return;
    }
    case 'stop': for (const x of st.jobs) { const j = jobOf(x.job); if (j && (j.state === 'pending' || j.state === 'running')) await act(() => srv(`/jobs/${j.id}/cancel`, {})); } await poll(); return;
    case 'resume': for (const x of st.jobs) { const j = jobOf(x.job); if (j && j.state === 'failed') await act(() => api('/api/retry', { job: j.id })); } await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; return finish(false);
    case 'save': return finish(true);
    default:
  }
}
function onInput(e) {
  if (cfgInput(e, st)) { renderActions(); if (started()) renderOut(); return saveSoon(); }
  const k = e.target.dataset.k;
  if (!k) return;
  st[k] = ['seed', 'steps'].includes(k) ? Number(e.target.value) || 0 : e.target.value;
  if (k === 'title') { const h = root.querySelector('[data-set] h1'); if (h) h.textContent = st.title || t('side.' + screen); }
  renderActions();
  if (started() && k === 'prompt') renderOut();
  saveSoon();
}
function onChange(e) {
  if (e.target.dataset.file === 'ref' && e.target.files[0]) upload(e.target.files[0]);
}
async function loadOpen() {
  open = (await act(() => api('/api/docs?kind=drafts')) || []).filter((d) => d && d.screen === screen && d.id !== st.id);
}

export default {
  async mount(el, { parts, query }) {
    root = el;
    screen = parts[1] === 'edit' ? 'edit' : 'gen';
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    const id = query.get('d');
    if (id) {
      const d = await act(() => api('/api/doc?kind=drafts&id=' + encodeURIComponent(id)));
      drafts[screen] = d && d.id ? Object.assign(fresh(screen), d) : fresh(screen);
    }
    st = drafts[screen] = drafts[screen] || fresh(screen);
    ui = freshUi();
    if (screen === 'edit') {
      const shared = (await act(() => api('/api/library?scope=shared'))) || {};
      lib = { chars: shared.characters || [], locs: shared.locations || [],
        scenes: (((await act(() => api('/api/library?scope=project'))) || {}).items || []).filter((i) => i.kind === 'scenes') };
    }
    await loadOpen();
    renderAll();
  },
  update() {
    if (!root || !st) return;
    if (String(started()) !== root.dataset.started) { root.dataset.started = String(started()); renderSet(); }   // the button label
    renderOut();
  },
  unmount() { root = null; },
};
