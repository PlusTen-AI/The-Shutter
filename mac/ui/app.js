// shutter UI core: state, API, i18n, header, router. Screens live in screens/*.js and export
// { mount(root, params), update() } — update runs after every poll (status + jobs, every 3 s).
import { DICT } from './i18n.js';

export const S = { status: null, jobs: [], lang: 'ko', hash: '', watch: new Set(), openSeq: null };
// jobs a screen follows beyond the open list's newest 200 (a production has thousands) → merged into S.jobs
export async function fetchJobs(ids) {
  const miss = [...new Set(ids)].filter((id) => id && !S.jobs.some((j) => j.id === id));
  for (let i = 0; i < miss.length; i += 300) {
    const part = miss.slice(i, i + 300);
    S.jobs = S.jobs.concat(await srv('/jobs?limit=300&ids=' + part.join(',')));
  }
}

// ---- text ----
export function t(key, vars) {
  let s = (DICT[S.lang] && DICT[S.lang][key]) ?? DICT.ko[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '');
  return s;
}
// a text with a fallback for keys made at run time (an expression panel written as a sentence: pn.ex_smirk → smirk)
export const tOr = (key, fb) => { const s = t(key); return s === key ? fb : s; };
export const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const raw = (s) => ({ __raw: String(s) });
// html`...${x}...` escapes values; raw() and arrays of html results pass through
export function html(strings, ...vals) {
  const put = (v) => v == null || v === false ? '' : Array.isArray(v) ? v.map(put).join('') : v.__raw !== undefined ? v.__raw : esc(v);
  return raw(strings.reduce((a, s, i) => a + s + (i < vals.length ? put(vals[i]) : ''), ''));
}

// ---- icons (24 px stroke paths) ----
const P = {
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  redo: '<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  open: '<path d="M14 4h6v6M20 4l-9 9M10 6H5v13h13v-5"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
  sliders: '<path d="M4 7h7M15 7h5M4 17h3M11 17h9"/><circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  star: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
  expand: '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
};
export const icon = (n, size = 16) => raw(`<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[n]}</svg>`);

// ---- API ----
export async function api(path, body, method) {
  const opt = body === undefined ? { method: method || 'GET' } : { method: method || 'POST', body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j = null;
  try { j = await r.json(); } catch (e) { j = null; }
  if (!r.ok) throw new Error((j && j.error) || r.status + ' ' + r.statusText);
  return j;
}
export const srv = (path, body, method) => api('/api/srv' + path, body, method);

let toastTimer = 0;
export function toast(msg, bad) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast' + (bad ? ' bad' : '');
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.add('out');
    toastTimer = setTimeout(() => { el.hidden = true; }, 180);
  }, bad ? 6000 : 3000);
}

// ---- motion ----
// Screens redraw their panes on every poll, so an element animates in only the first time its key is drawn (enter),
// and looping animations (busy dots, skeletons, progress stripes) keep one shared clock: each new element gets a
// negative delay (--phase) that puts it where the loop already is, instead of restarting on every redraw.
const shown = new Set();
export const enter = (key) => shown.has(key) ? '' : (shown.add(key), 'enter');
const LOOP_MS = 1600, LOOPS = '.sk, .dot.busy, .bar > div';
const phase = (el) => el.style.setProperty('--phase', -(performance.now() % LOOP_MS) + 'ms');
new MutationObserver((muts) => {
  for (const m of muts) for (const n of m.addedNodes) {
    if (n.nodeType !== 1) continue;
    if (n.matches(LOOPS)) phase(n);
    n.querySelectorAll(LOOPS).forEach(phase);
  }
}).observe(document.body, { childList: true, subtree: true });

// ---- lightbox: result images open large in the page, ←/→ through the ones beside it (⌘-click still opens a tab) ----
const IMG = /\.(png|jpe?g|webp)(\?|$)/i;
let box = null;
function showBox() {
  const el = document.getElementById('lb');
  const [url, name] = box.list[box.i];
  el.innerHTML = html`<div class="lb-bar">
      <span class="mono sub">${box.list.length > 1 ? `${box.i + 1} / ${box.list.length} · ` : ''}${name}</span><div class="grow"></div>
      ${box.list.length > 1 ? html`<button type="button" class="ib" data-lb="prev" aria-label="${t('lb.prev')}" title="${t('lb.prev')}" style="transform: rotate(90deg)">${icon('down')}</button>
        <button type="button" class="ib" data-lb="next" aria-label="${t('lb.next')}" title="${t('lb.next')}" style="transform: rotate(-90deg)">${icon('down')}</button>` : ''}
      <a class="ib" href="${url}" target="_blank" data-lb="tab" aria-label="${t('lb.open')}" title="${t('lb.open')}">${icon('open')}</a>
      <button type="button" class="ib" data-lb="close" aria-label="${t('jobs.close')}" title="${t('jobs.close')} (Esc)">${icon('x')}</button></div>
    <div class="lb-img" data-lb="close"><img src="${url}" alt="${name}"></div>`.__raw;
  el.hidden = false;
}
function closeBox() { box = null; document.getElementById('lb').hidden = true; }
document.addEventListener('click', (e) => {
  const l = e.target.closest('[data-lb]');
  if (l) {
    const a = l.dataset.lb;
    if (a === 'tab') return;
    if (a === 'close' && e.target.tagName === 'IMG') return;   // a click on the picture itself keeps it open
    if (a === 'close') return closeBox();
    box.i = (box.i + (a === 'next' ? 1 : -1) + box.list.length) % box.list.length;
    return showBox();
  }
  const link = e.target.closest('a[target="_blank"]');
  if (!link || e.metaKey || e.ctrlKey || e.shiftKey || e.button || !IMG.test(link.getAttribute('href'))) return;
  e.preventDefault();
  const scope = link.closest('[data-out], .pane-out, .page, main') || document;
  const links = [...scope.querySelectorAll('a[target="_blank"]')].filter((x) => IMG.test(x.getAttribute('href')));
  const uniq = [...new Map(links.map((x) => [x.href.split('?')[0], x])).values()];
  const list = uniq.map((x) => [x.href, decodeURIComponent(x.href.split('?')[0].split('/').pop())]);
  box = { list, i: Math.max(0, list.findIndex(([u]) => u.split('?')[0] === link.href.split('?')[0])) };
  showBox();
});
document.addEventListener('keydown', (e) => {
  if (!box) return;
  if (e.key === 'Escape') closeBox();
  else if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && box.list.length > 1) {
    box.i = (box.i + (e.key === 'ArrowRight' ? 1 : -1) + box.list.length) % box.list.length;
    showBox();
  } else return;
  e.preventDefault();
});
// run an action, report its error as a toast; returns the result (or undefined on error)
export async function act(fn, okMsg) {
  try {
    const r = await fn();
    if (okMsg) toast(okMsg);
    return r;
  } catch (e) {
    toast(e.message, true);
  }
}

// ---- small shared helpers ----
export function filesUrl(rel) { return '/files/' + rel.split('/').map(encodeURIComponent).join('/'); }
// a job's result folder under the storage location (mac.py job_dir): projects/<p>/work/jobs/<id>, no project → default
export const jobBase = (j) => `projects/${j.project || (S.status || {}).default_project}/work/jobs/${j.id}`;
export function jobFileUrl(job, name) { return filesUrl(jobBase(job) + '/' + name); }
export function fmtBytes(n) {
  if (n == null) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(n < 10 ? 1 : 0) : n) + ' ' + u[i];
}
export function fmtAgo(sec) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - sec));
  if (s < 60) return t('ago.s', { n: s });
  if (s < 3600) return t('ago.m', { n: Math.floor(s / 60) });
  if (s < 86400) return t('ago.h', { n: Math.floor(s / 3600) });
  return t('ago.d', { n: Math.floor(s / 86400) });
}
export function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  return h ? t('dur.hm', { h, m }) : m ? t('dur.ms', { m, s }) : t('dur.s', { s });
}
export const jobKind = (type) => t('type.' + type);
// stopped by the user (중지: its remaining units become failed with error 'cancelled') — shown as 중지됨 in grey, not as a
// red failure (boss, session #14). A job: every failed unit was cancelled; a unit: its error
export const isStopped = (j) => !!j && j.state === 'failed' && (j.errors || []).length > 0 && j.errors.every((e) => e === 'cancelled');
export const unitStopped = (u) => !!u && u.state === 'failed' && u.error === 'cancelled';
export const failWord = (stopped) => t(stopped ? 'ch.st.stopped' : 'ch.st.failed');
export const failColor = (stopped) => stopped ? 'var(--fg3)' : 'var(--bad-fg)';
// a character on screen: FL01, or one of its outfits (wardrobe) FL01-suit → "FL01 · suit · <clothing, first part>"
export function codeLabel(c) {
  if (!c || !c.card || !c.card.of) return c ? c.code : '';
  const style = c.code.slice(c.card.of.length + 1);
  const cloth = String(c.card.clothing || '').split(/[,;]/)[0].trim();
  return [c.card.of, /^\d+$/.test(style) ? t('ch.outfitN', { n: style }) : style, cloth.length > 18 ? cloth.slice(0, 18) + '…' : cloth]
    .filter(Boolean).join(' · ');
}
export const codeLabelOf = (chars, code) => codeLabel(chars.find((x) => x.code === code) || { code });
export function jobTitle(j) {
  const o = j.opts || {};
  return [o.asset_code, o.asset_kind].filter(Boolean).join(' · ') || jobKind(j.type);
}
// node state from the server's worker list: busy (a unit), idle, off
export function nodes() {
  const w = (S.status && S.status.workers) || {};
  return Object.keys(w).sort().map((name) => {
    const v = w[name];
    const off = !S.status.server_ok || v.ago > 60;
    return { name, ...v, state: off ? 'off' : v.unit ? 'busy' : 'idle' };
  });
}

// ---- header ----
let menuOpen = false, menuPop = false;
function renderHeader() {
  const st = S.status || {};
  const set = st.settings || {};
  const route = S.hash.split('/')[1] || 'image';
  const active = S.jobs.filter((j) => !j.closed && (j.state === 'pending' || j.state === 'running')).length;
  const failed = S.jobs.filter((j) => !j.closed && j.state === 'failed' && !isStopped(j)).length;   // stopped ones are no failure
  const ns = nodes();
  const dot = { busy: 'busy', idle: 'ok', off: '' };
  const projs = st.projects || [];
  document.getElementById('hd').innerHTML = html`
    <div class="brand">shutter</div>
    <div style="position: relative">
      <button type="button" class="btn ghost" data-act="menu" aria-expanded="${menuOpen}" style="min-height: 36px; padding: 0 10px">
        <span class="sub">${t('hd.project')}</span>${set.project || t('hd.noProject')}${icon('down', 14)}</button>
      ${menuOpen ? html`<div class="menu ${menuPop ? 'pop' : ''}" role="menu">
        ${projs.map((p) => html`<button type="button" class="item" role="menuitem" data-act="project" data-p="${p}" aria-current="${p === set.project}">${p}</button>`)}
        ${projs.length ? html`<div style="height: 1px; background: var(--line); margin: 6px 0"></div>` : ''}
        <form class="row" data-form="newProject" style="gap: 6px; padding: 4px; flex-wrap: nowrap">
          <input class="in" name="name" placeholder="${t('hd.newProject')}" style="min-height: 34px; flex: 1; min-width: 0" required>
          <button class="ib p" aria-label="${t('hd.newProject')}" title="${t('hd.newProject')}">${icon('plus')}</button>
        </form></div>` : ''}
    </div>
    <nav class="row" style="gap: 4px" aria-label="${t('hd.cats')}">
      <a class="tab" href="#/image/character" aria-current="${route === 'image' ? 'page' : 'false'}">${t('nav.image')}</a>
      <a class="tab" href="#/video/one" aria-current="${route === 'video' ? 'page' : 'false'}">${t('nav.video')}</a>
      <a class="tab" href="#/swap" aria-current="${route === 'swap' ? 'page' : 'false'}">${t('nav.swap')}</a>
    </nav>
    <div class="grow"></div>
    <a class="tab" href="#/jobs" aria-current="${route === 'jobs' ? 'page' : 'false'}">${t('nav.jobs')}
      ${active ? html`<span class="badge">${active}</span>` : ''}${failed ? html`<span class="dot bad" title="${t('hd.failed', { n: failed })}"></span>` : ''}</a>
    <a class="tab" href="#/library" aria-current="${route === 'library' ? 'page' : 'false'}">${t('nav.library')}</a>
    <span class="row lbl" style="gap: 6px" title="${st.server_ok ? t('hd.nodesTip') : t('hd.serverOff')}">
      ${ns.length ? ns.map((n) => html`<span class="mono">${n.name}</span><span class="dot ${dot[n.state]}"></span>`)
        : html`<span class="dot"></span><span class="sub">${t('hd.noNodes')}</span>`}</span>
    <span class="row lbl" style="gap: 6px" title="${st.storage_ok ? set.storage : t('hd.storageOff')}">
      <span class="dot ${st.storage_ok ? 'ok' : 'bad'}"></span>${t('hd.storage')}</span>
    <a class="ib m" href="#/settings" aria-label="${t('nav.settings')}" title="${t('nav.settings')}" aria-current="${route === 'settings' ? 'page' : 'false'}">${icon('gear', 18)}</a>`.__raw;
}
document.getElementById('hd').addEventListener('click', async (e) => {
  const a = e.target.closest('[data-act]');
  if (!a) return;
  if (a.dataset.act === 'menu') { menuOpen = menuPop = !menuOpen; renderHeader(); menuPop = false; }
  if (a.dataset.act === 'project') {
    menuOpen = false;
    await act(() => api('/api/settings', { project: a.dataset.p }));
    await poll();
    route();
  }
});
document.getElementById('hd').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = new FormData(e.target).get('name');
  if (await act(() => api('/api/projects', { name }))) {
    menuOpen = false;
    await poll();
    route();
  }
});
// composedPath(), not target.closest(): the header click above re-renders the header, so by the time the click
// reaches the document its target is detached and closest('.hd') would close the menu it just opened
document.addEventListener('click', (e) => {
  if (menuOpen && !e.composedPath().includes(document.getElementById('hd'))) { menuOpen = false; renderHeader(); }
});

// ---- notifications on job end ----
const seen = new Map();
function notifyEnds(jobs) {
  const on = S.status && S.status.settings && S.status.settings.notify;
  for (const j of jobs) {
    const was = seen.get(j.id);
    seen.set(j.id, j.state);
    if (!was || was === j.state || !['done', 'failed'].includes(j.state) || j.type === 'video.probe') continue;   // a probe is no work result
    const stop = isStopped(j);
    const msg = t(j.state === 'done' ? 'note.done' : stop ? 'note.stopped' : 'note.failed', { what: jobTitle(j) });
    toast(msg, j.state === 'failed' && !stop);
    if (on && 'Notification' in window && Notification.permission === 'granted') new Notification('shutter', { body: msg });
  }
}

// ---- poll ----
export async function poll() {
  try {
    S.status = await api('/api/status');
    S.lang = (S.status.settings && S.status.settings.lang) || 'ko';
    document.documentElement.lang = S.lang;
  } catch (e) {
    S.status = { server_ok: false, workers: {}, settings: {}, projects: [] };
  }
  try {
    S.jobs = await srv('/jobs?open=1&limit=200');
    notifyEnds(S.jobs);
    if (S.watch.size) await fetchJobs([...S.watch]);
  } catch (e) { /* server unreachable: keep the last list */ }
  renderHeader();
  if (current && current.update) await current.update();
  import('./prod.js').then((m) => m.tick());   // started script productions keep moving on any screen (기능 16)
}

// ---- router ----
const SCREENS = {
  image: () => import('./screens/image.js'),
  video: () => import('./screens/video.js'),
  swap: () => import('./screens/swap.js'),
  jobs: () => import('./screens/jobs.js'),
  library: () => import('./screens/library.js'),
  settings: () => import('./screens/settings.js'),
};
let current = null;
export async function route() {
  let h = location.hash || '';
  if (!h || h === '#/' || h === '#') {
    h = localStorage.getItem('shutter.last') || '#/image/character';   // opens on the last-used screen
    history.replaceState(null, '', h);
  }
  S.hash = h.slice(1);
  try { localStorage.setItem('shutter.last', h); } catch (e) { /* private window */ }
  const [path, query] = S.hash.split('?');
  const parts = path.split('/').filter(Boolean);
  const load = SCREENS[parts[0]] || SCREENS.image;
  const mod = (await load()).default;
  if (current && current.unmount) current.unmount();
  const root = document.getElementById('view');
  root.innerHTML = '';
  root.replaceWith(root.cloneNode(false));   // drop the old screen's listeners
  current = mod;
  renderHeader();
  const view = document.getElementById('view');
  view.classList.add('view-in');
  mod.mount(view, { parts, query: new URLSearchParams(query || '') });
}
window.addEventListener('hashchange', route);
export function rerender() { route(); }

(async () => {
  await poll();
  await route();
  setInterval(poll, 3000);
})();
