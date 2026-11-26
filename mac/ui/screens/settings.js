// 설정 (page): 일반 · 저장 · 노드 · 기본값 (design §11)
import { S, t, html, icon, api, srv, act, toast, poll, rerender, nodes, fmtBytes } from '../app.js';
import { MODELS, defaults } from '../models.js';

let root = null, sec = 'general', model = 'qi21', askClear = false, cfg = null;

function general(s) {
  return html`<div class="col" style="gap: 18px">
    <div class="field">${t('set.lang')}
      <div class="seg" role="group" aria-label="${t('set.lang')}" style="align-self: flex-start">
        <button type="button" data-act="lang" data-v="ko" aria-pressed="${s.lang === 'ko'}">한국어</button>
        <button type="button" data-act="lang" data-v="en" aria-pressed="${s.lang === 'en'}">English</button></div></div>
    <label class="tog"><input type="checkbox" data-act="notify" ${s.notify ? 'checked' : ''}>${t('set.notify')}
      <span class="sub">${t('set.notifyNote')}</span></label>
    <form class="col" data-form="records" style="gap: 8px">
      <div class="row" style="gap: 12px; align-items: flex-end">
        <label class="field">${t('set.recordDays')}<input class="in mono" name="record_days" inputmode="numeric" value="${s.record_days ?? 30}" style="width: 120px"></label>
        <button class="btn p">${t('common.apply')}</button></div>
      <p class="sub" style="margin: 0">${t('set.recordNote')}</p></form></div>`;
}

const fixed = (k) => ((S.status || {}).fixed || []).includes(k);   // set on the node (shutter.conf), read-only here

function storage(s) {
  const st = S.status || {};
  const p = s.project || t('set.projectName');
  const where = fixed('storage') ? st.storage_label : s.storage;
  const dot = html`<span class="row lbl" style="gap: 6px"><span class="dot ${st.storage_ok ? 'ok' : 'bad'}"></span>${st.storage_ok ? t('set.connected') : t('set.missing')}</span>`;
  return html`<form class="col" data-form="storage" style="gap: 18px">
    ${fixed('storage') ? html`<div class="field">${t('set.storage')}
      <div class="row" style="gap: 8px"><span class="mono" style="flex: 1 1 320px">${where}</span>${dot}</div></div>` : html`
    <label class="field">${t('set.storage')}
      <div class="row" style="gap: 8px"><input class="in mono" name="storage" value="${s.storage}" style="flex: 1 1 320px; width: auto">${dot}</div></label>
    <div class="row"><button class="btn p">${t('common.apply')}</button></div>`}
    <div class="card col" style="gap: 6px">
      <h3 class="h3">${t('set.tree')}</h3>
      <pre class="mono sub" style="margin: 0; line-height: 1.8">${where || '…'}/
  characters/    ${t('set.treeChars')}
  locations/     ${t('set.treeLocs')}
  props/         ${t('set.treeProps')}
  uploads/       ${t('set.treeUploads')}
  projects/
    ${p}/
      images/    ${t('set.treeImages')}
      videos/    ${t('set.treeVideos')}
      script/    ${t('set.treeScript')}
      work/      ${t('set.treeWork')}</pre></div>
    <p class="sub" style="margin: 0">${fixed('storage') ? t('set.storageFixed') + ' · ' : ''}${t('set.storageNote')}</p></form>
`;
}

function nodeList() {
  const ns = nodes();
  return ns.length ? html`<div class="row" style="align-items: stretch">${ns.map((n) => html`
    <div class="card row" style="flex: 1 1 240px; gap: 10px; padding: 12px 16px">
      <span class="dot ${n.state === 'busy' ? 'busy' : n.state === 'idle' ? 'ok' : ''}"></span><span class="mono">${n.name}</span>
      <span class="lbl grow">${t('set.node.' + n.state)}</span><span class="sub mono">${t('set.cacheNow', { size: fmtBytes(n.cache) })}</span></div>`)}</div>` :
    html`<div class="empty">${t('jobs.noWorkers')}</div>`;
}

function nodesSec(s) {
  return html`<div class="col" style="gap: 18px">
    <div data-nodes>${nodeList()}</div>
    ${fixed('server') ? '' : html`<form class="row" data-form="server" style="gap: 12px; align-items: flex-end">
      <label class="field" style="flex: 0 1 360px">${t('set.server')}<input class="in mono" name="server" value="${s.server}"></label>
      <button class="btn p">${t('common.apply')}</button></form>`}
    ${cfg ? html`<form class="col" data-form="cache" style="gap: 12px">
      <div class="row" style="gap: 16px; align-items: flex-end">
        <label class="field">${t('set.cacheDays')}<input class="in mono" name="cache_days" inputmode="numeric" value="${cfg.cache_days}" style="width: 120px"></label>
        <label class="field">${t('set.cacheGb')}<input class="in mono" name="cache_gb" inputmode="numeric" value="${cfg.cache_gb}" style="width: 120px"></label>
        <button class="btn p">${t('common.apply')}</button></div>
      <p class="sub" style="margin: 0">${t('set.cacheNote')}</p></form>` : html`<div class="sub">${t('hd.serverOff')}</div>`}
    ${askClear ? html`<div class="confirm" role="alertdialog"><span>${t('set.askClear')}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClear">${t('common.cancel')}</button>
        <button type="button" class="btn p" data-act="yesClear">${t('set.clear')}</button></div></div>` :
      html`<div class="row"><button type="button" class="btn ghost" data-act="clear">${icon('trash', 14)}${t('set.clear')}</button></div>`}</div>`;
}

function defaultsSec() {
  const m = MODELS[model], d = defaults(model);
  const field = (f) => {
    const label = t('def.' + f.k), v = d[f.k];
    if (f.type === 'bool') return html`<label class="tog"><input type="checkbox" name="${f.k}" ${v ? 'checked' : ''}>${label}</label>`;
    if (f.type === 'num') return html`<label class="field">${label}<input class="in mono" name="${f.k}" inputmode="decimal" value="${v}" style="width: 140px">
      ${f.note ? html`<span class="sub">${t('def.' + f.note)}</span>` : ''}</label>`;
    if (f.type === 'text') return html`<label class="field">${label}<textarea class="in" name="${f.k}" rows="2">${v}</textarea>
      <span class="sub">${t('def.' + f.k + 'Note')}</span></label>`;
    if (f.type === 'seg') return html`<div class="field">${label}<div class="seg" role="group" aria-label="${label}" style="align-self: flex-start">
      ${f.opts.map((o) => html`<button type="button" class="mono" data-act="seg" data-k="${f.k}" data-v="${o}" aria-pressed="${String(v) === String(o)}" style="min-width: 44px">${o}</button>`)}</div>
      <input type="hidden" name="${f.k}" value="${v}"></div>`;
    return html`<label class="field">${label}<select class="in" name="${f.k}" style="width: 220px">
      ${f.opts.map((o) => html`<option value="${o}" ${String(v) === String(o) ? 'selected' : ''}>${t('def.' + f.k + '.' + o) === 'def.' + f.k + '.' + o ? o : t('def.' + f.k + '.' + o)}</option>`)}</select></label>`;
  };
  return html`<form class="col" data-form="defaults" style="gap: 16px">
    <label class="field">${t('set.model')}<select class="in" data-model style="width: 260px">
      ${Object.entries(MODELS).map(([k, x]) => html`<option value="${k}" ${k === model ? 'selected' : ''}>${x.name} · ${t('set.kind.' + x.kind)}</option>`)}</select></label>
    <div class="card col" style="gap: 14px">${m.fields.map(field)}</div>
    <div class="row"><button class="btn p">${t('common.save')}</button><button type="button" class="btn ghost" data-act="reset">${t('set.reset')}</button></div>
    <p class="sub" style="margin: 0">${t('set.defNote')}</p></form>`;
}

function render() {
  if (!root) return;
  const s = (S.status && S.status.settings) || {};
  const secs = ['general', 'storage', 'nodes', 'defaults'];
  const body = { general, storage, nodes: nodesSec, defaults: defaultsSec }[sec](s);
  root.innerHTML = html`<div class="split">
    <nav class="side" aria-label="${t('nav.settings')}">
      <div class="grp" style="margin-top: 0">${t('nav.settings')}</div>
      ${secs.map((k) => html`<a href="#/settings/${k}" aria-current="${k === sec ? 'page' : 'false'}">${t('set.sec.' + k)}</a>`)}</nav>
    <div class="col" style="flex: 999 1 560px; min-width: 0; padding: 24px 28px 64px; border-left: 1px solid #1C1C21; gap: 18px; max-width: 880px">
      <div><h1 class="h1">${t('set.sec.' + sec)}</h1><p class="lead">${t('lead.set.' + sec)}</p></div>${body}</div></div>`.__raw;
}

async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a) return;
  switch (a.dataset.act) {
    case 'lang': await act(() => api('/api/settings', { lang: a.dataset.v })); await poll(); return rerender();
    case 'notify': {
      const on = a.checked;
      if (on && 'Notification' in window && Notification.permission === 'default') await Notification.requestPermission();
      await act(() => api('/api/settings', { notify: on }));
      await poll();
      return;
    }
    case 'seg': {
      const form = a.closest('form');
      form.querySelector(`input[name="${a.dataset.k}"]`).value = a.dataset.v;
      a.parentElement.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === a)));
      return;
    }
    case 'reset': {
      const all = Object.assign({}, (S.status.settings || {}).defaults);
      delete all[model];
      await act(() => api('/api/settings', { defaults: all }), t('set.resetDone'));
      await poll();
      break;
    }
    case 'clear': askClear = true; break;
    case 'noClear': askClear = false; break;
    case 'yesClear': {
      askClear = false;
      const r = await act(() => srv('/cache/clear', {}));
      if (r) toast(t('set.cleared', { size: fmtBytes(r.freed) }));
      break;
    }
    default: return;
  }
  render();
}

async function onSubmit(e) {
  e.preventDefault();
  const f = e.target, fd = new FormData(f);
  if (f.dataset.form === 'storage') {
    await act(() => api('/api/settings', { storage: fd.get('storage') }), t('common.saved'));
  } else if (f.dataset.form === 'server') {
    await act(() => api('/api/settings', { server: fd.get('server') }), t('common.saved'));
  } else if (f.dataset.form === 'records') {
    await act(() => api('/api/settings', { record_days: Number(fd.get('record_days')) || 30 }), t('common.saved'));
  } else if (f.dataset.form === 'cache') {
    cfg = await act(() => srv('/config', { cache_days: Number(fd.get('cache_days')), cache_gb: Number(fd.get('cache_gb')) }), t('common.saved')) || cfg;
  } else if (f.dataset.form === 'defaults') {
    const out = {};
    for (const fl of MODELS[model].fields) {
      if (fl.type === 'bool') out[fl.k] = f.querySelector(`[name="${fl.k}"]`).checked;
      else {
        const v = fd.get(fl.k);
        out[fl.k] = typeof fl.def === 'number' ? Number(v) : v;
      }
    }
    const all = Object.assign({}, (S.status.settings || {}).defaults, { [model]: out });
    await act(() => api('/api/settings', { defaults: all }), t('common.saved'));
  }
  await poll();
  render();
}

export default {
  async mount(el, { parts }) {
    root = el;
    sec = ['general', 'storage', 'nodes', 'defaults'].includes(parts[1]) ? parts[1] : 'general';
    el.addEventListener('click', onClick);
    el.addEventListener('submit', onSubmit);
    el.addEventListener('change', (e) => { if ('model' in e.target.dataset) { model = e.target.value; render(); } });
    render();
    if (sec === 'nodes') { cfg = await act(() => srv('/config')) || null; render(); }
  },
  update() {   // only the live node cards; the forms keep what is being typed
    const n = root && root.querySelector('[data-nodes]');
    if (n) n.innerHTML = nodeList().__raw;
  },
  unmount() { root = null; askClear = false; },
};
