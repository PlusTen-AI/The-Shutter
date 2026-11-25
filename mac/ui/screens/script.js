// 영상 생성 › 워크플로 › 대본 (design §12 기능 16): the project's episode files (script/*.md, docs/script.md) read by
// the Mac — errors by file:line, the scenes with their cuts, first frames and running time. Picked scenes become
// sequences (one per scene: its cuts as cut docs, its first frames in the sequence; an existing one takes the edited
// script and keeps its results), then 제작 시작 lets each run on by itself (prod.js): first frames → ● → video.
import { S, t, html, icon, api, act, toast, poll, fmtDur } from '../app.js';
import { sideMenu, VIDEO_MENU } from './wip.js';

let root = null, sc = null, seqs = {}, picked = new Set(), mode = 'all';
const project = () => S.status && S.status.settings && S.status.settings.project;

async function load() {
  sc = project() ? await act(() => api('/api/script')) : null;
  const all = project() ? (await act(() => api('/api/docs?kind=sequences'))) || [] : [];
  seqs = Object.fromEntries(all.filter((q) => q && q.script).map((q) => [String(q.script.scene), q]));
}
const made = (n) => seqs[String(n)];
const state = (n) => { const q = made(n); return !q ? 'notMade' : q.prod && q.prod.done ? 'done' : q.prod && q.prod.started ? 'started' : 'made'; };

async function makeGo() {
  const r = await act(() => api('/api/script/make', { scenes: [...picked], mode }));
  if (!r) return;
  toast(t('sc2.made') + ' · ' + r.made.length);
  await load();
  renderAll();
}
async function startGo() {
  for (const n of picked) {
    const q = made(n);
    if (!q || (q.prod && q.prod.started)) continue;
    q.prod = Object.assign({}, q.prod, { started: true, done: false });
    if (!(await act(() => api('/api/doc', { kind: 'sequences', id: q.id, doc: q })))) return;
  }
  await poll();
  renderAll();
}

function renderAll() {
  if (!root) return;
  const scenes = (sc && sc.scenes) || [], errs = (sc && sc.errors) || [], warns = (sc && sc.warnings) || [];
  const pk = scenes.filter((x) => picked.has(x.no));
  const toStart = pk.filter((x) => state(x.no) === 'made');
  const why = !project() ? t('vg.needProject') : errs.length ? t('sc2.fixErrors') : !pk.length ? t('sc2.needPick') : '';
  const total = scenes.reduce((a, x) => a + x.sec, 0);
  root.innerHTML = html`<div class="split">${sideMenu(VIDEO_MENU(), '#/video/script')}
    <section class="pane-set" aria-label="${t('ch.settings')}">
      <div><div class="lbl">${t('sc2.crumb')}</div><h1 class="h1">${t('side.script')}</h1><p class="lead">${t('lead.script')}</p></div>
      <div class="sec">
        <div class="field">${t('sc2.dir')}<span class="mono sub" style="word-break: break-all">${(sc && sc.dir) || '—'}</span></div>
        <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="reload">${icon('redo', 14)}${t('sc2.reload')}</button>
          ${sc ? html`<span class="sub mono">${(sc.files || []).length} md</span>` : ''}</div>
        <div class="field">${t('sc2.mode')}<div class="seg" role="group">
          ${[['all', t('ch.modeAll')], ['step', t('ch.modeStep')]].map(([v, l]) => html`<button type="button" data-act="mode" data-v="${v}" aria-pressed="${mode === v}">${l}</button>`)}</div>
          <span class="sub">${mode === 'all' ? t('pr.autoNote') : t('pr.stepNote')}</span></div></div>
      ${errs.length || warns.length ? html`<div class="sec">
        ${errs.length ? html`<h2 class="h3" style="color: var(--bad-fg)">${t('sc2.errors', { n: errs.length })}</h2>
          <div class="col" style="gap: 4px; max-height: 320px; overflow: auto">${errs.map((e) => html`<span class="mono sub" style="font-size: 12px">${e.file}:${e.line} ${e.msg}</span>`)}</div>` : ''}
        ${warns.length ? html`<h2 class="h3">${t('sc2.warnings', { n: warns.length })}</h2>
          <div class="col" style="gap: 4px; max-height: 200px; overflow: auto">${warns.map((e) => html`<span class="mono sub" style="font-size: 12px">${e.file}:${e.line} ${e.msg}</span>`)}</div>` : ''}</div>` : ''}
    </section>
    <main class="pane-out" aria-label="${t('ch.results')}">
      <div class="row" style="justify-content: space-between; min-height: 40px">
        <span class="sub mono">${sc ? t('sc2.meta', { s: scenes.length, c: scenes.reduce((a, x) => a + x.cuts.length, 0), f: scenes.reduce((a, x) => a + x.frames.length, 0), t: fmtDur(total) }) : ''}</span>
        <div class="toprow-end"><div class="topact"><span class="note">${why}</span>
          ${toStart.length ? html`<button type="button" class="btn ghost" data-act="start">${t('sc2.start', { n: toStart.length })}</button>` : ''}
          <button type="button" class="btn p" data-act="make" ${why ? 'disabled' : ''}>${pk.some((x) => made(x.no)) ? t('sc2.update', { n: pk.length }) : t('sc2.make', { n: pk.length })}</button></div></div></div>
      <section class="card">
        <div class="row"><span class="num">1</span><h2 class="h2">${t('sc2.scenes')}</h2><span class="sub">${t('sc2.scenesNote')}</span><div class="grow"></div>
          ${scenes.length ? html`<label class="tog" style="margin: 0"><input type="checkbox" data-all ${pk.length === scenes.length ? 'checked' : ''}>${t('sc2.all')}</label>` : ''}</div>
        ${scenes.length ? html`<div class="list" style="margin-top: 12px">${scenes.map((x) => {
          const q = made(x.no), st = state(x.no);
          const fdone = q ? (q.frames || []).filter((f) => f.rel).length : 0;
          return html`<div class="item-row" style="gap: 10px">
            <input type="checkbox" data-scene="${x.no}" ${picked.has(x.no) ? 'checked' : ''} aria-label="${t('fm.scene', { n: x.no })}">
            <span class="mono" style="min-width: 30px">${x.no}</span>
            <span style="flex: 1 1 200px; min-width: 0">${x.title} <span class="sub mono">· ${x.file}</span></span>
            <span class="mono sub">${t('sq.meta', { n: x.cuts.length, s: x.sec.toFixed(1) })} · ${x.frames.length}F${x.cuts.some((c) => c.break) ? ' · ▷' : ''}</span>
            <span class="row" style="gap: 6px; min-width: 150px; justify-content: flex-end; font-size: 12px; color: var(--fg3)">
              <span class="dot ${st === 'done' ? 'ok' : st === 'started' ? 'busy' : ''}"></span>${q ? html`<a href="#/video/seq?id=${q.id}">${t('sc2.' + (st === 'made' ? 'made' : st))}${st !== 'made' ? ` · ●${fdone}/${(q.frames || []).length}` : ''}</a>` : t('sc2.notMade')}</span></div>`;
        })}</div>` : html`<p class="sub" style="margin: 6px 0 0 38px">${t('sc2.none')}</p>`}
        ${Object.keys(seqs).length ? html`<div class="row" style="justify-content: flex-end; margin-top: 10px"><a class="btn ghost" href="#/video/film">${t('sc2.film')}</a></div>` : ''}
      </section>
    </main></div>`.__raw;
}

async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const d = a.dataset;
  if (d.act === 'reload') { await load(); return renderAll(); }
  if (d.act === 'mode') { mode = d.v; return renderAll(); }
  if (d.act === 'make') return makeGo();
  if (d.act === 'start') return startGo();
}
function onChange(e) {
  const d = e.target.dataset;
  if (d.scene) { e.target.checked ? picked.add(Number(d.scene)) : picked.delete(Number(d.scene)); return renderAll(); }
  if (e.target.matches('[data-all]')) { picked = e.target.checked ? new Set(((sc && sc.scenes) || []).map((x) => x.no)) : new Set(); return renderAll(); }
}

export default {
  async mount(el) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('change', onChange);
    await load();
    renderAll();
  },
  update() {},
  unmount() { root = null; },
};
