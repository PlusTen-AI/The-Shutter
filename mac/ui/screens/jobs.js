// 작업: queue + progress + failed (resume) + recent, one screen (design §11)
import { S, t, html, icon, srv, api, act, toast, jobKind, jobTitle, fmtAgo, fmtDur, nodes, jobFileUrl, filesUrl, fmtBytes, poll, enter, isStopped, failColor } from '../app.js';

let root = null, ask = null, view = 'now', recs = null, rq = '', recTimer = 0;
const open = new Set();
let scope = 'project';
try { scope = localStorage.getItem('shutter.jobs.scope') || 'project'; } catch (e) { /* no storage */ }

// worker model ids → the names the screens use
const MODEL_NAMES = { qi21: 'Qwen-Image 2.1', 'qi21-turbo': 'Qwen-Image 2.1', h3: 'H3', ltx25: 'LTX-2.5', pih3: 'pi-H3', sv2: 'SeedVR2', face: 'hyperswap', cpu: 'CPU' };

function nodeCards() {
  const ns = nodes();
  if (!ns.length) return html`<div class="empty">${S.status && S.status.server_ok ? t('jobs.noWorkers') : t('hd.serverOff')}</div>`;
  return html`<div class="row" style="align-items: stretch">${ns.map((n) => {
    const j = n.unit && S.jobs.find((x) => n.unit.startsWith(x.id + '.'));
    const what = n.state === 'busy' ? t('jobs.nodeBusy', { what: j ? jobTitle(j) : n.unit }) :
      n.state === 'idle' ? (n.loaded ? t('jobs.nodeIdle', { model: MODEL_NAMES[String(n.loaded).replace(/_\d+$/, '')] || n.loaded }) : t('jobs.nodeEmpty')) : t('jobs.nodeOff');
    return html`<div class="card row" style="flex: 1 1 260px; gap: 10px; padding: 12px 16px">
      <span class="dot ${n.state === 'busy' ? 'busy' : n.state === 'idle' ? 'ok' : ''}"></span>
      <span class="mono" style="font-weight: 500">${n.name}</span><span class="lbl grow">${what}</span>
      <span class="sub mono" title="${t('jobs.cacheTip')}">${fmtBytes(n.cache)}</span></div>`;
  })}</div>`;
}

function unitsDone(j) { return j.count.done; }
function unitsAll(j) { return j.count.pending + j.count.running + j.count.done + j.count.failed; }

function confirmRow(j) {
  if (!ask || ask.id !== j.id) return '';
  const msg = ask.what === 'cancel' ? t('jobs.askCancel', { n: unitsDone(j) }) :
    unitsAll(j) > 1 && j.state === 'failed' ? t('jobs.askCloseMulti') : t('jobs.askClose');
  return html`<div class="confirm" role="alertdialog" style="flex-basis: 100%">
    <span>${msg}</span>
    <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="no">${t('common.cancel')}</button>
    <button type="button" class="btn p" data-act="yes" data-id="${j.id}">${ask.what === 'cancel' ? t('jobs.stop') : t('jobs.close')}</button></div></div>`;
}

function detail(j) {
  if (!open.has(j.id)) return '';
  const o = j.opts || {};
  const imgs = j.files.filter((f) => f.name.endsWith('.png'));
  const vids = j.files.filter((f) => f.name.endsWith('.mp4'));
  return html`<div class="col" style="flex-basis: 100%; gap: 10px; padding-top: 6px; border-top: 1px solid #222227">
    <div class="row sub" style="gap: 6px 18px">
      ${o.seed != null ? html`<span class="mono">seed ${o.seed}</span>` : ''}
      <span class="mono">${j.id}</span>
      ${j.project ? html`<span>${t('jobs.inProject', { p: j.project })}</span>` : ''}
      ${j.errors.length ? html`<span style="color: var(--bad-fg)">${j.errors[0]}</span>` : ''}</div>
    ${imgs.length ? html`<div class="grid" style="grid-template-columns: repeat(auto-fill, minmax(120px, 1fr))">${imgs.map((f) => html`
      <a class="ph" href="${jobFileUrl(j, f.name)}" target="_blank" title="${f.name}" style="aspect-ratio: 1; display: block">
        ${f.synced ? html`<img src="${jobFileUrl(j, f.name)}" alt="${f.name}" loading="lazy">` : html`<span class="sub" style="position: absolute; inset: auto 6px 6px">${t('jobs.arriving')}</span>`}</a>`)}</div>` : ''}
    ${vids.length ? html`<div class="row" style="gap: 8px">${vids.map((f) => f.synced ?
      html`<a class="btn ghost" href="${jobFileUrl(j, f.name)}" target="_blank">${icon('play', 12)}${f.name}</a>` :
      html`<span class="btn ghost" aria-disabled="true">${f.name} · ${t('jobs.arriving')}</span>`)}</div>` : ''}
    ${!j.files.length ? html`<span class="sub">${t('jobs.noFiles')}</span>` : ''}</div>`;
}

function row(j, kind) {
  const all = unitsAll(j), done = unitsDone(j);
  const pct = Math.round(j.progress * 100);
  const ns = nodes().filter((n) => n.unit && n.unit.startsWith(j.id + '.')).map((n) => n.name);
  const synced = j.files.filter((f) => f.synced && !f.tmp).length, total = j.files.filter((f) => !f.tmp).length;
  const showProj = scope === 'all';
  return html`<div class="item-row ${enter('job:' + j.id + ':' + kind)}">
    <span class="sub" style="min-width: 110px">${jobKind(j.type)}</span>
    <span style="font-weight: 500; min-width: 140px">${jobTitle(j)}</span>
    ${showProj ? html`<span class="sub">${j.project || t('jobs.noProject')}</span>` : ''}
    ${kind === 'active' ? html`
      <span class="sub mono" style="min-width: 40px">${ns.join('·') || t('jobs.waiting')}</span>
      <div class="col grow" style="gap: 4px; min-width: 160px">
        <div class="bar"><div style="width: ${pct}%"></div></div>
        <span class="sub mono">${pct}% · ${t('jobs.units', { done, all })} · ${fmtDur(Date.now() / 1000 - j.created)}</span></div>
      ${j.count.pending ? html`<button type="button" class="ib m" data-act="first" data-id="${j.id}" aria-label="${t('jobs.first')}" title="${t('jobs.first')}">${icon('up')}</button>` : ''}
      <button type="button" class="ib m" data-act="cancel" data-id="${j.id}" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`
    : kind === 'failed' ? html`
      <span class="sub grow" style="color: ${failColor(isStopped(j))}">${isStopped(j) ? t('ch.st.stopped') : (j.errors.find((e) => e !== 'cancelled') || '').split('\n')[0].slice(0, 120)}</span>
      <span class="sub mono">${t('jobs.units', { done, all })} · ${fmtAgo(j.ended || j.created)}</span>
      ${all > 1 ? html`<span class="sub">${t('jobs.autoClose')}</span>` : ''}
      <button type="button" class="btn p" data-act="retry" data-id="${j.id}" style="min-height: 34px">${icon('play', 12)}${all > 1 ? t('jobs.resume') : t('jobs.rerun')}</button>
      <button type="button" class="ib m" data-act="close" data-id="${j.id}" aria-label="${t('jobs.close')}" title="${t('jobs.close')}">${icon('x')}</button>`
    : html`
      <span class="sub grow">${fmtAgo(j.ended || j.created)}</span>
      <span class="sub">${total && synced === total ? t('jobs.onMac') : t('jobs.receiving', { n: synced, all: total })}</span>
      <button type="button" class="ib m" data-act="close" data-id="${j.id}" aria-label="${t('jobs.close')}" title="${t('jobs.closeTip')}">${icon('x')}</button>`}
    <button type="button" class="ib m" data-act="toggle" data-id="${j.id}" aria-expanded="${open.has(j.id)}" aria-label="${t('jobs.detail')}" title="${t('jobs.detail')}">${icon('down')}</button>
    ${confirmRow(j)}${detail(j)}</div>`;
}

function render() {
  if (!root) return;
  const cur = S.status && S.status.settings && S.status.settings.project;
  const jobs = S.jobs.filter((j) => !j.closed && (scope === 'all' || j.project === cur));
  const active = jobs.filter((j) => j.state === 'pending' || j.state === 'running');
  const failed = jobs.filter((j) => j.state === 'failed');
  const done = jobs.filter((j) => j.state === 'done').slice(0, 30);   // recent only; older ones stay in the record
  const sec = (title, list, kind, empty) => html`<section class="col" style="gap: 8px">
    <div class="row"><h2 class="h2">${title}</h2><span class="sub mono">${list.length}</span></div>
    ${list.length ? html`<div class="list">${list.map((j) => row(j, kind))}</div>` : html`<div class="empty">${empty}</div>`}</section>`;
  root.innerHTML = html`<div class="page">
    <div class="row" style="justify-content: space-between">
      <div><h1 class="h1">${t('nav.jobs')}</h1><p class="lead">${t('lead.jobs')}</p></div>
      <div class="seg" role="group" aria-label="${t('jobs.scope')}">
        <button type="button" data-act="scope" data-v="project" aria-pressed="${scope === 'project'}">${t('jobs.thisProject')}</button>
        <button type="button" data-act="scope" data-v="all" aria-pressed="${scope === 'all'}">${t('jobs.allProjects')}</button></div></div>
    <div class="seg" role="group" aria-label="${t('jobs.view')}" style="align-self: flex-start">
      <button type="button" data-act="view" data-v="now" aria-pressed="${view === 'now'}">${t('jobs.now')}</button>
      <button type="button" data-act="view" data-v="rec" aria-pressed="${view === 'rec'}">${t('jobs.records')}</button></div>
    ${view === 'rec' ? records() : html`
    ${nodeCards()}
    ${sec(t('jobs.active'), active, 'active', t('jobs.noActive'))}
    ${sec(t('jobs.failed'), failed, 'failed', t('jobs.noFailed'))}
    ${sec(t('jobs.done'), done, 'done', t('jobs.noDone'))}
    <p class="sub" style="margin: 0">${t('jobs.note')}</p>`}</div>`.__raw;
}

// ---- 기록: every ended job's job.json on the Mac (closed ones too), searchable; same-settings rerun ----
async function loadRecords() {
  const cur = S.status && S.status.settings && S.status.settings.project;
  const q = new URLSearchParams({ q: rq });
  if (scope === 'project' && cur) q.set('project', cur);
  recs = await act(() => api('/api/records?' + q)) || [];
  render();
}
function recFileUrl(r, name) {
  const base = r.project ? `projects/${r.project}/jobs/${r.id}` : `jobs/${r.id}`;
  return filesUrl(base + '/' + name);
}
function records() {
  const keyOpts = (o) => Object.entries(o).filter(([k, v]) => v !== '' && v != null && !['prompt', 'h3_prompt', 'edit_prompt', 'edit_prompt_part', 'bg_prompt', 'shots'].includes(k))
    .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
  return html`<div class="col" style="gap: 10px">
    <input class="in" type="search" data-rq placeholder="${t('jobs.recSearch')}" value="${rq}">
    ${recs === null ? html`<div class="empty sk">${t('common.loading')}</div>` : !recs.length ? html`<div class="empty">${t('jobs.recEmpty')}</div>` :
      html`<div class="list">${recs.map((r) => html`<div class="item-row">
        <span class="dot ${r.state === 'done' ? 'ok' : 'bad'}"></span>
        <span class="sub mono" style="min-width: 120px">${new Date(r.created * 1000).toLocaleString(S.lang === 'en' ? 'en-GB' : 'ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
        <span class="sub" style="min-width: 110px">${jobKind(r.type)}</span>
        <span style="font-weight: 500; min-width: 120px">${[r.asset_code, r.asset_kind].filter(Boolean).join(' · ') || '—'}</span>
        <span class="sub mono grow">${r.seed != null ? 'seed ' + r.seed : ''}${r.seeds.length > 1 ? ' (' + r.seeds.slice(0, 6).join(', ') + (r.seeds.length > 6 ? '…' : '') + ')' : ''}</span>
        ${scope === 'all' ? html`<span class="sub">${r.project || t('jobs.noProject')}</span>` : ''}
        <button type="button" class="btn ghost" data-act="rerun" data-id="${r.id}" style="min-height: 32px" title="${t('jobs.rerunTip')}">${icon('redo', 14)}${t('jobs.rerunSame')}</button>
        <button type="button" class="ib m" data-act="toggle" data-id="${r.id}" aria-expanded="${open.has(r.id)}" aria-label="${t('jobs.detail')}">${icon('down')}</button>
        ${open.has(r.id) ? html`<div class="col" style="flex-basis: 100%; gap: 8px; padding-top: 6px; border-top: 1px solid #222227">
          <span class="sub mono" style="word-break: break-all">${r.id} · ${keyOpts(r.opts)}</span>
          ${r.opts.prompt || r.opts.h3_prompt ? html`<details><summary class="sub">${t('jobs.prompt')}</summary><pre class="mono sub" style="white-space: pre-wrap; margin: 6px 0 0">${r.opts.prompt || r.opts.h3_prompt}</pre></details>` : ''}
          ${r.errors.length ? html`<span class="sub" style="color: var(--bad-fg)">${r.errors[0]}</span>` : ''}
          <div class="row" style="gap: 6px">${r.files.map((f) => r.here.includes(f.name) ?
            html`<a class="chip" href="${recFileUrl(r, f.name)}" target="_blank" style="display: inline-flex; align-items: center; text-decoration: none">${f.name}</a>` :
            html`<span class="chip" title="${t('jobs.recGone')}" style="opacity: .5">${f.name}</span>`)}</div>
          <span class="sub">${t('jobs.recFile')}</span></div>` : ''}</div>`)}</div>`}</div>`;
}

async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a) return;
  const id = a.dataset.id;
  switch (a.dataset.act) {
    case 'scope': scope = a.dataset.v; try { localStorage.setItem('shutter.jobs.scope', scope); } catch (x) { /* */ } if (view === 'rec') return loadRecords(); break;
    case 'view': view = a.dataset.v; if (view === 'rec') { recs = null; render(); return loadRecords(); } break;
    case 'rerun': { const r = await act(() => api('/api/rerun', { job: id })); if (r) { toast(t('jobs.rerunDone', { id: r.id })); await poll(); } return; }
    case 'toggle': open.has(id) ? open.delete(id) : open.add(id); break;
    case 'first': await act(() => srv(`/jobs/${id}/first`, {}), t('jobs.firstDone')); await poll(); return;
    case 'cancel': case 'close': ask = { id, what: a.dataset.act }; break;
    case 'no': ask = null; break;
    case 'yes': {
      const what = ask.what;
      ask = null;
      await act(() => srv(`/jobs/${id}/${what}`, {}));
      await poll();
      return;
    }
    case 'retry': await act(() => api('/api/retry', { job: id }), t('jobs.resumed')); await poll(); return;
    default: return;
  }
  render();
}

export default {
  mount(el, { parts }) {
    root = el;
    if (parts[1] === 'records') { view = 'rec'; recs = null; loadRecords(); }
    el.addEventListener('click', onClick);
    el.addEventListener('input', (e) => {
      if (!('rq' in e.target.dataset)) return;
      rq = e.target.value;
      clearTimeout(recTimer);
      recTimer = setTimeout(async () => { await loadRecords(); const i = root && root.querySelector('[data-rq]'); if (i) { i.focus(); i.setSelectionRange(rq.length, rq.length); } }, 300);
    });
    render();
  },
  update() { if (view === 'now') render(); },
  unmount() { root = null; ask = null; },
};
