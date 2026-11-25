// 보관함: shared (characters · locations) and this project (scenes · videos), detail on the right (design §11)
import { S, t, html, icon, api, act, toast, filesUrl, fmtAgo, codeLabel } from '../app.js';

let root = null, tab = 'shared', kind = 'all', q = '', data = null, sel = null, ask = false;
const picked = new Set();   // videos picked for a batch upscale

const latest = (c, k) => c.images.filter((i) => i.kind === k).sort((a, b) => b.v - a.v)[0];
const codeUrl = (cat, c, name) => filesUrl(`${cat}/${c.code}/${name}`);
// a code's subtitle: its name, else who it is in a few Korean words (the card's English prompt text is not shown)
const codeName = (c) => c.card.name || (/^[FM][LSE]\d/.test(c.code) ? [t(c.code[0] === 'F' ? 'ch.female' : 'ch.male'),
  t({ L: 'ch.lead', S: 'ch.support', E: 'ch.extraRank' }[c.code[1]]), c.card.age].filter(Boolean).join(' · ') : '');

async function load() {
  data = null;
  render();
  data = await act(() => api('/api/library?scope=' + tab)) || {};
  render();
}

function sharedItems(all) {   // all: outfits too (the selected item may be one)
  const out = [];
  for (const cat of ['characters', 'locations']) {
    if (kind !== 'all' && kind !== cat) continue;
    for (const c of (data[cat] || [])) if (all || !c.card.of) out.push({ key: cat + '/' + c.code, cat, c });
  }
  return out.filter((x) => !q || (x.c.code + ' ' + JSON.stringify(x.c.card)).toLowerCase().includes(q.toLowerCase()));
}
function projectItems() {
  return (data.items || []).filter((i) => (kind === 'all' || kind === i.kind) &&
    (!q || (i.name + ' ' + JSON.stringify(i.meta)).toLowerCase().includes(q.toLowerCase())))
    .sort((a, b) => b.mtime - a.mtime);
}

function tile(x) {
  const on = sel === x.key;
  let thumb, title, sub;
  if (x.c) {
    const m = x.c.master;
    thumb = m ? html`<img src="${codeUrl(x.cat, x.c, m)}" alt="" loading="lazy">` : html`<span class="sub" style="position: absolute; inset: 40% 0; text-align: center">${t('lib.noMaster')}</span>`;
    title = x.c.code; sub = codeName(x.c) || t('lib.cat.' + x.cat);
  } else {
    thumb = x.i.kind === 'videos' ? html`<video src="${filesUrl(x.i.path)}#t=0.5" preload="metadata" muted></video>` :
      html`<img src="${filesUrl(x.i.path)}" alt="" loading="lazy">`;
    title = x.i.name; sub = fmtAgo(x.i.mtime);
  }
  const video = x.i && x.i.kind === 'videos';
  return html`<div style="position: relative">
    <button type="button" data-act="sel" data-key="${x.key}" class="col" style="gap: 6px; width: 100%; padding: 8px; text-align: left; background: var(--bg2); border: 1px solid ${on ? 'var(--blue-line)' : '#24242A'}; border-radius: 10px">
      <div class="ph" style="aspect-ratio: ${x.cat === 'locations' || video ? '16 / 9' : '3 / 4'}; width: 100%">${thumb}</div>
      <span class="mono" style="font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${title}</span>
      <span class="sub" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${sub}</span></button>
    ${video ? html`<label class="tog" title="${t('lib.pick')}" style="position: absolute; top: 12px; left: 14px; min-height: 0">
      <input type="checkbox" data-pick="${x.i.path}" ${picked.has(x.i.path) ? 'checked' : ''} aria-label="${t('lib.pick')}"></label>` : ''}</div>`;
}

// 옷장: the character and its outfits (FL01-<style>); a click shows that one, + adds an outfit in the character screen
function wardrobe(c) {
  const base = c.card.of || c.code;
  const all = (data.characters || []).filter((o) => o.code === base || o.card.of === base);
  return html`<div class="col" style="gap: 8px"><span class="lbl">${t('ch.wardrobe')}</span>
    <div class="row" style="gap: 8px; align-items: flex-start">${all.map((o) => html`
      <button type="button" class="col" data-act="sel" data-key="characters/${o.code}" title="${codeLabel(o)}"
        style="gap: 4px; width: 76px; padding: 4px; background: transparent; border: 1px solid ${o.code === c.code ? 'var(--blue-line)' : '#2A2A30'}; border-radius: 8px">
        <div class="ph" style="width: 100%; aspect-ratio: 3 / 4">${o.master ? html`<img src="${codeUrl('characters', o, o.master)}" alt="" loading="lazy">` : ''}</div>
        <span class="sub" style="font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%">${o.card.of ? o.code.slice(base.length + 1) : t('ch.outfitBase')}</span></button>`)}
      <a class="slot" href="#/image/character?code=${base}&outfit=new" title="${t('ch.addOutfit')}" style="width: 76px; aspect-ratio: 3 / 4; text-decoration: none">${icon('plus', 16)}<span>${t('ch.addOutfitShort')}</span></a></div></div>`;
}

function detailShared(x) {
  const c = x.c, cat = x.cat;
  const sheet = latest(c, 'sheet');
  const kinds = [...new Set(c.images.map((i) => i.kind))].filter((k) => !['ref', 'anchor', 'sheet'].includes(k));
  const card = Object.entries(c.card).filter(([k, v]) => !['code', 'gender'].includes(k) && String(v).trim());
  const goto = cat === 'characters' ?
    [['#/image/character?code=' + c.code, t('lib.edit'), 'edit'], ['#/video/one?code=' + c.code, t('lib.useVideo'), 'play'], ['#/swap?code=' + c.code, t('lib.useSwap'), 'redo']] :
    [['#/image/location?code=' + c.code, t('lib.edit'), 'edit']];
  return html`<div class="col">
    <div class="row"><h2 class="h2 mono">${c.card.of ? codeLabel(c) : c.code}</h2><span class="lbl">${c.card.of ? '' : codeName(c)}</span></div>
    ${cat === 'characters' ? wardrobe(c) : ''}
    ${c.master ? html`<a class="ph" href="${codeUrl(cat, c, c.master)}" target="_blank" style="display: block; aspect-ratio: ${cat === 'locations' ? '16/9' : '3/4'}; max-height: 420px">
      <img src="${codeUrl(cat, c, c.master)}" alt="${c.master}" style="object-fit: contain; background: #111"></a>
      <span class="sub mono">● ${c.master}</span>` : html`<div class="empty">${t('lib.noMaster')}</div>`}
    ${sheet ? html`<a class="ph" href="${codeUrl(cat, c, sheet.name)}" target="_blank" style="display: block; aspect-ratio: 3440 / 1280">
      <img src="${codeUrl(cat, c, sheet.name)}" alt="${sheet.name}"></a><span class="sub mono">${t('lib.sheet')} · ${sheet.name}</span>` : ''}
    ${kinds.length ? html`<div class="row sub" style="gap: 4px 10px">${kinds.map((k) => html`<span class="mono">${k} v${latest(c, k).v}</span>`)}</div>` : ''}
    ${card.length ? html`<div class="col" style="gap: 4px">${card.map(([k, v]) => html`<div class="sub"><span style="color: var(--fg3)">${t('card.' + k)}</span> · ${v}</div>`)}</div>` : ''}
    <div class="row" style="gap: 8px">${goto.map(([href, label, ic]) => html`<a class="btn ghost" href="${href}">${icon(ic, 14)}${label}</a>`)}</div>
    ${delRow(c.users && c.users.length ? t('lib.askDelCode', { code: c.code, n: c.users.length }) : t('lib.askDelCodeFree', { code: c.code }))}</div>`;
}

function detailProject(x) {
  const i = x.i, m = i.meta || {};
  const video = i.kind === 'videos';
  return html`<div class="col">
    <h2 class="h2 mono" style="word-break: break-all">${i.name}</h2>
    <div class="ph" style="aspect-ratio: ${video ? '16 / 9' : 'auto'}">${video ?
      html`<video src="${filesUrl(i.path)}" controls style="object-fit: contain; background: #000"></video>` :
      html`<a href="${filesUrl(i.path)}" target="_blank"><img src="${filesUrl(i.path)}" alt="" style="object-fit: contain"></a>`}</div>
    <div class="row sub" style="gap: 4px 14px">
      ${m.type ? html`<span>${t('type.' + m.type)}</span>` : ''}${m.asset_code ? html`<span class="mono">${m.asset_code}</span>` : ''}
      ${m.seed != null ? html`<span class="mono">seed ${m.seed}</span>` : ''}${m.job ? html`<span class="mono">${m.job}</span>` : ''}</div>
    ${video ? html`<div class="row"><button type="button" class="btn ghost" data-act="upscale" data-path="${i.path}">${icon('expand', 14)}${t('lib.upscale')}</button></div>` : ''}
    ${delRow(t('lib.askDel', { name: i.name }))}</div>`;
}

function delRow(msg) {
  return ask ? html`<div class="confirm" role="alertdialog"><span>${msg}</span><div class="row" style="gap: 8px">
      <button type="button" class="btn ghost" data-act="noDel">${t('common.cancel')}</button>
      <button type="button" class="btn p" data-act="yesDel">${t('lib.delete')}</button></div></div>` :
    html`<div class="row"><button type="button" class="btn ghost" data-act="del" style="color: var(--bad-fg)">${icon('trash', 14)}${t('lib.delete')}</button></div>`;
}

function render() {
  if (!root) return;
  const kinds = tab === 'shared' ? ['all', 'characters', 'locations'] : ['all', 'scenes', 'videos'];
  const items = !data ? [] : tab === 'shared' ? sharedItems() : projectItems().map((i) => ({ key: i.path, i }));
  const cur = (tab === 'shared' && data && !data.error ? sharedItems(true) : items).find((x) => x.key === sel);
  const proj = S.status && S.status.settings && S.status.settings.project;
  root.innerHTML = html`<div class="split">
    <div class="col" style="flex: 999 1 560px; min-width: 0; padding: 24px 28px 64px; gap: 16px">
      <div class="row" style="justify-content: space-between">
        <div><h1 class="h1">${t('nav.library')}</h1><p class="lead">${t('lead.library')}</p></div>
        <div class="seg" role="group" aria-label="${t('lib.scope')}">
          <button type="button" data-act="tab" data-v="shared" aria-pressed="${tab === 'shared'}">${t('lib.shared')}</button>
          <button type="button" data-act="tab" data-v="project" aria-pressed="${tab === 'project'}">${t('lib.project', { p: proj || '—' })}</button></div></div>
      <div class="row" style="gap: 8px">
        <input class="in" type="search" data-q placeholder="${t('lib.search')}" value="${q}" style="flex: 1 1 220px; width: auto">
        ${kinds.map((k) => html`<button type="button" class="chip" data-act="kind" data-v="${k}" aria-pressed="${kind === k}">${t('lib.kind.' + k)}</button>`)}</div>
      ${picked.size ? html`<div class="row confirm"><span>${t('lib.picked', { n: picked.size })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="unpick">${t('common.cancel')}</button>
        <button type="button" class="btn p" data-act="upscaleAll">${icon('expand', 14)}${t('lib.upscaleN', { n: picked.size })}</button></div></div>` : ''}
      ${!data ? html`<div class="empty sk">${t('common.loading')}</div>` : data.error ? html`<div class="banner bad">${data.error}</div>` :
        items.length ? html`<div class="grid">${items.map(tile)}</div>` :
        html`<div class="empty">${tab === 'project' && !proj ? t('lib.noProject') : t('lib.empty')}</div>`}</div>
    <aside class="pane-out" style="flex: 1 1 340px; max-width: 460px; background: var(--bg2)" aria-label="${t('lib.detail')}">
      ${cur ? (cur.c ? detailShared(cur) : detailProject(cur)) : html`<div class="empty">${t('lib.pickOne')}</div>`}</aside></div>`.__raw;
}

async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a) return;
  const v = a.dataset.v;
  switch (a.dataset.act) {
    case 'tab': tab = v; kind = 'all'; sel = null; ask = false; picked.clear(); return load();
    case 'kind': kind = v; break;
    case 'sel': sel = a.dataset.key; ask = false; break;
    case 'del': ask = true; break;
    case 'noDel': ask = false; break;
    case 'yesDel': {
      ask = false;
      const items = tab === 'shared' ? sharedItems() : projectItems().map((i) => ({ key: i.path, i }));
      const cur = items.find((x) => x.key === sel);
      if (!cur) break;
      const body = cur.c ? { code: cur.c.code } : { kind: cur.i.kind, name: cur.i.name };
      if (await act(() => api('/api/delete', body), t('lib.deleted'))) { sel = null; return load(); }
      break;
    }
    case 'upscale': await act(() => api('/api/upscale', { path: a.dataset.path }), t('lib.upscaleQueued', { n: 1 })); return;
    case 'unpick': picked.clear(); break;
    case 'upscaleAll': {
      const paths = [...picked];
      for (const p of paths) await act(() => api('/api/upscale', { path: p }));
      toast(t('lib.upscaleQueued', { n: paths.length }));
      picked.clear();
      break;
    }
    default: return;
  }
  render();
}

export default {
  mount(el, { query }) {
    root = el;
    if (query.get('tab')) tab = query.get('tab');
    el.addEventListener('click', onClick);
    el.addEventListener('change', (e) => {
      const p = e.target.dataset.pick;
      if (p === undefined) return;
      e.target.checked ? picked.add(p) : picked.delete(p);
      render();
    });
    el.addEventListener('input', (e) => {
      if (!('q' in e.target.dataset)) return;
      q = e.target.value;
      const pos = e.target.selectionStart;
      render();
      const inp = root.querySelector('[data-q]');
      inp.focus();
      inp.setSelectionRange(pos, pos);
    });
    load();
  },
  unmount() { root = null; },
};
