// 이미지 › 워크플로 › 배경: card → 앵커 candidates (place.anchor) → ● base → 다른 시점 (place.derive, one image each,
// optional) → 등록 (anchor + views + ☆ into locations/<code>/). Same two panes as the character screen; the session lives
// on the Mac in locations/<code>/session.json until 등록 / 닫기. #/image/location?code=L02 opens a registered one.
// 사물 (기능 10, #/image/prop): the same steps for an object that comes back in many scenes (robot, vehicle) — card
// (name, English noun, description) → product reference picture (prop.ref) → ● → 방향 칸 앞 · 옆 · 뒤 · 3/4
// (prop.derive) → 등록 into props/<code>/ (P01 …).
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, enter, isStopped, failWord, failColor } from '../app.js';
import { defaults, cfgFields, cfgInput, cfgOpts, cfgX } from '../models.js';
import { sideMenu, IMAGE_MENU } from './wip.js';

const CFG = {
  location: { cat: 'locations', g: 'L', route: '#/image/location', anchor: 'place.anchor', derive: 'place.derive', master: 'anchor',
    aspect: '16:9', ratio: '16 / 9', keys: ['name', 'kind', 'space', 'scale', 'body'], need: 'space', views: [] },
  prop: { cat: 'props', g: 'P', route: '#/image/prop', anchor: 'prop.ref', derive: 'prop.derive', master: 'ref',
    aspect: '2:3', ratio: '2 / 3', keys: ['name', 'what', 'desc'], need: 'desc',
    views: [['앞', 'Show it from straight in front.'], ['옆', 'Show it from its left side, in a full side profile.'],
      ['뒤', 'Show it from straight behind.'], ['3/4', 'Show it from a three-quarter front angle, turned about 45 degrees.']] },
};
let K = CFG.location, CARD_KEYS = K.keys;
const L = (k, v) => { const x = K.cat === 'props' && t('pp.' + k, v); return x && x !== 'pp.' + k ? x : t('lo.' + k, v); };   // 사물 texts, else the 배경 ones
const SEC = { 1: 19, 4: 150, load: 30 };   // seconds per anchor (1MP, 4MP set ≈ 2~3 min — knowhow §2), model load

let root = null, st = null, codePreview = '', open = [], detail = {}, ui = {}, autoBusy = false;
const freshUi = () => ({ en: false, enText: '', askClose: false, askStop: false, adv: false });

function fresh() {
  const d = defaults('qi21');
  return { code: null, card: K.cat === 'locations' ? { kind: 'interior' } : {}, size: 1, mode: 'all', cands: d.cands, seed: 1, prompt: null, cfg: d.cfg, neg: d.neg,
    masterJobs: [], base: null, saved: [], views: K.views.map(([name, text], i) => ({ name, text, kind: 'view_' + (i + 1), job: null })),
    oldViewJobs: [], cardAt: null, registered: null };
}

// ---- derived ----
const job = (id) => S.jobs.find((j) => j.id === id) || detail[id] || null;
const active = (j) => j && (j.state === 'pending' || j.state === 'running');
const allJobs = () => [...st.masterJobs, ...st.views.map((v) => v.job)].filter(Boolean);
const running = () => allJobs().some((id) => active(job(id)));
const started = () => !!st.code && (st.masterJobs.length > 0 || !!st.registered);
const cardKey = () => JSON.stringify([st.size, CARD_KEYS.map((k) => (st.card[k] || '').trim()), st.prompt]);
const dirty = () => started() && st.cardAt !== null && st.cardAt !== cardKey();
const sameRef = (a, b) => !!a && !!b && a.job === b.job && a.name === b.name && a.registered === b.registered;
const pngOf = (j) => j && j.files.find((f) => f.name.endsWith('.png') && f.synced);

function candidates() {
  const out = [];
  if (st.registered) out.push({ registered: st.registered, url: filesUrl(`${K.cat}/${st.code}/${st.registered}`) });
  for (const id of st.masterJobs) {
    const j = job(id);
    if (!j) continue;
    for (const f of j.files.filter((x) => x.name.endsWith('.png') && x.synced)) {
      out.push({ job: id, name: f.name, url: jobFileUrl(j, f.name) + '?v=' + f.sha.slice(0, 8), seed: f.name.split('_')[1].split('.')[0] });
    }
    for (let k = 0; k < j.count.pending + j.count.running; k++) out.push({ pending: true });
    for (let k = 0; k < j.count.failed; k++) out.push({ failed: true, stopped: isStopped(j) });
  }
  return out;
}
function stage(kind) {
  if (kind === 'a') {
    const js = st.masterJobs.map(job).filter(Boolean);
    if (js.some(active)) return { state: 'run', pct: Math.round(100 * js.filter(active).reduce((a, j) => a + j.progress, 0) / js.filter(active).length) };
    if (candidates().some((c) => c.url)) return { state: st.base ? 'done' : 'pick' };
    return { state: js.some((j) => j.state === 'failed') ? 'failed' : 'wait', stopped: js.filter((j) => j.state === 'failed').every(isStopped) };
  }
  const js = st.views.map((v) => job(v.job)).filter(Boolean);
  if (!st.views.length) return { state: 'off' };
  if (js.some(active)) return { state: 'run', pct: Math.round(100 * js.reduce((a, j) => a + (j.state === 'done' ? 1 : j.progress), 0) / st.views.length) };
  if (js.some((j) => j.state === 'failed')) return { state: 'failed', stopped: js.filter((j) => j.state === 'failed').every(isStopped) };
  return { state: st.views.every((v) => pngOf(job(v.job))) ? 'done' : 'wait' };
}
const stText = (s) => ({ wait: t('ch.st.wait'), run: (s.pct || 0) + '%', done: t('ch.st.done'), failed: failWord(s.stopped), pick: t('ch.st.pick'), off: t('vg.st.off') }[s.state]);
const stDot = (s) => ({ run: 'busy', done: 'ok', failed: s.stopped ? '' : 'bad', pick: 'warn' }[s.state] || '');

// ---- persistence / actions ----
async function save() { if (st.code) await act(() => api('/api/session', { code: st.code, session: st })); }
async function saveCard() {
  const c = Object.fromEntries(CARD_KEYS.map((k) => [k, (st.card[k] || '').trim()]).filter(([, v]) => v));
  return api('/api/codes', { code: st.code, card: c });
}
async function submitAnchors(n) {
  const used = st.masterJobs.reduce((a, id) => a + ((job(id) || {}).count ? Object.values(job(id).count).reduce((x, y) => x + y, 0) : 0), 0);
  const o = { count: n || st.cands, seed: st.seed + used, mp: st.size, aspect: K.aspect, ...cfgOpts(st) };
  if (st.prompt) o.prompt = st.prompt;
  const r = await api('/api/codejob', { type: K.anchor, code: st.code, options: o });
  st.masterJobs.push(r.id);
}
async function submitView(i) {
  const v = st.views[i];
  if (!v.text.trim() || !st.base) return;
  const r = await api('/api/codejob', { type: K.derive, code: st.code, file_kind: v.kind, options: { view: v.text.trim(), seed: st.seed + 50 + i, ...cfgOpts(st) },
    master: st.base.job ? st.base : undefined });
  if (v.job) st.oldViewJobs.push(v.job);
  v.job = r.id;
  v.from = JSON.stringify(st.base);
  v.at = v.text.trim();
}
async function start() {
  if (!(st.card[K.need] || '').trim() && !st.prompt) return toast(L('needSpace'), true);
  const ok = await act(async () => {
    if (!st.code) st.code = (await api(`/api/nextcode?g=${K.g}&r=`)).code;
    await saveCard();
    st.cardAt = cardKey();
    st.base = null;
    await submitAnchors();
    await save();
    return true;
  });
  if (ok) { await poll(); renderAll(); }
}
// 한 번에 쭉: the first anchor becomes ● (no checks for places), then the listed views are made from it
async function auto() {
  if (autoBusy || !started() || st.mode !== 'all' || running()) return;
  autoBusy = true;
  if (!st.base) {
    const c = candidates().find((x) => x.url && !x.registered);
    if (c) { st.base = { job: c.job, name: c.name }; await save(); }
  }
  if (st.base) {
    for (let i = 0; i < st.views.length; i++) {
      if (st.views[i].text.trim() && !st.views[i].job) await act(() => submitView(i));
    }
    await save();
  }
  autoBusy = false;
}

// ---- left ----
function seg(name, opts, cur, label, dis) {
  return html`<div class="seg" role="group" aria-label="${label}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${name}" data-v="${v}" aria-pressed="${String(cur) === String(v)}" ${dis ? 'disabled' : ''}>${l}</button>`)}</div>`;
}
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const resumable = open.filter((c) => c !== st.code);
  const n = st.cands;
  el.innerHTML = html`
    <div><div class="lbl">${L('crumb')}</div>
      <h1 class="h1">${st.card.name || (st.code ? '' : L('new'))} <span class="mono" style="font-weight: ${st.card.name || !st.code ? 400 : 600}; color: ${st.card.name || !st.code ? 'var(--fg3)' : 'var(--fg)'}">${st.code || codePreview}${st.code ? '' : ' · ' + t('ch.auto')}</span></h1><p class="lead">${t(K.cat === 'props' ? 'lead.prop' : 'lead.location')}</p></div>
    ${!started() && resumable.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${resumable.map((c) => html`<a class="chip" href="${K.route}?code=${c}" style="display: inline-flex; align-items: center; text-decoration: none">${c}</a>`)}</div>` : ''}
    <div class="col" style="gap: 6px">${seg('mode', [['all', t('ch.modeAll')], ['step', t('ch.modeStep')]], st.mode, t('ch.mode'))}
      <span class="sub">${st.mode === 'all' ? t('lo.modeAllHint') : t('lo.modeStepHint')}</span></div>
    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${t('ch.card')}</h2><div class="grow"></div>
        <button type="button" class="btn ghost" data-act="en" aria-expanded="${ui.en}" style="min-height: 30px; padding: 0 8px; font-size: 12px; color: var(--blue)">${t('ch.english')}</button></div>
      ${ui.en ? html`<div class="col" style="gap: 6px; padding: 12px; background: var(--bg2); border: 1px solid #2E3A50; border-radius: 10px">
        <textarea class="in mono" rows="7" data-en style="font-size: 12px">${st.prompt || ui.enText}</textarea>
        <div class="row" style="justify-content: space-between"><span class="sub">${st.prompt ? t('ch.enEdited') : t('ch.enFromCard')}</span>
          ${st.prompt ? html`<button type="button" class="ib m" data-act="enReset" aria-label="${t('ch.enReset')}" title="${t('ch.enReset')}">${icon('redo')}</button>` : ''}</div></div>` : ''}
      <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px">
        <label class="field">${t('card.name')}<input class="in" data-card="name" value="${st.card.name || ''}" placeholder="${L('ph.name')}"></label>
        ${K.cat === 'props' ? html`<label class="field">${t('pp.what')}<input class="in" data-card="what" value="${st.card.what || ''}" placeholder="${t('pp.ph.what')}"></label>
        <label class="field" style="grid-column: 1 / -1"><span>${t('pp.desc')} <span class="sub">${t('pp.descNote')}</span></span>
          <textarea class="in" rows="5" data-card="desc" placeholder="${t('pp.ph.desc')}">${st.card.desc || ''}</textarea></label>` : html`
        <div class="field">${t('lo.kind')}${seg('kind', [['interior', t('lo.interior')], ['exterior', t('lo.exterior')]], st.card.kind || 'interior', t('lo.kind'))}</div>
        <label class="field" style="grid-column: 1 / -1">${t('lo.space')}<input class="in" data-card="space" value="${st.card.space || ''}" placeholder="${t('lo.ph.space')}"></label>
        <label class="field" style="grid-column: 1 / -1">${t('lo.scale')}<input class="in" data-card="scale" value="${st.card.scale || ''}" placeholder="${t('lo.ph.scale')}"></label>
        <label class="field" style="grid-column: 1 / -1"><span>${t('lo.body')} <span class="sub">${t('lo.bodyNote')}</span></span>
          <textarea class="in" rows="5" data-card="body" placeholder="${t('lo.ph.body')}">${st.card.body || ''}</textarea></label>`}</div></div>
    <div class="sec">
      <div class="row" style="gap: 10px"><h2 class="h3">${L('anchor')}</h2><span class="sub mono">Qwen-Image 2.1 · ${K.aspect}</span></div>
      <div class="field">${t('def.mp')}${seg('size', [[1, t('lo.size1')], [4, t('lo.size4')]], st.size, t('def.mp'))}</div>
      <div class="row" style="gap: 10px"><span class="lbl">${t('ch.cands')}</span>${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}<span class="sub">${t('ch.candsNote')}</span>
        <div class="grow"></div><button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      ${ui.adv ? html`<div class="col" style="gap: 10px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        <label class="field">${t('ch.seed')}<input class="in mono" data-num="seed" inputmode="numeric" value="${st.seed}" style="width: 120px"></label>
        ${cfgFields(st)}</div>` : ''}</div>
    <div class="sec">
      <div class="row" style="gap: 4px 10px"><h2 class="h3">${L('views')}</h2><span class="sub">${L('viewsNote')}</span></div>
      ${st.views.map((v, i) => html`<div class="row" style="gap: 6px; flex-wrap: nowrap">
        <input class="in" data-vname="${i}" value="${v.name || ''}" placeholder="${t('lo.ph.vname')}" title="${t('lo.vnameTip')}" style="flex: 0 0 92px">
        <input class="in" data-view="${i}" value="${v.text}" placeholder="${t('lo.ph.view')}" style="flex: 1 1 0; min-width: 0">
        <button type="button" class="ib m" data-act="delView" data-i="${i}" aria-label="${t('vg.remove')}" title="${t('vg.remove')}" ${active(job(v.job)) ? 'disabled' : ''}>${icon('x', 14)}</button></div>`)}
      <button type="button" class="btn add" data-act="addView">${icon('plus', 14)}${L('addView')}</button></div>`.__raw;
  renderActions();
}
// the main action, top right of the results pane (boss, session #14): the note (what is missing · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const n = st.cands, miss = !(st.card[K.need] || '').trim() && !st.prompt;
  el.innerHTML = (!started() ? html`<span class="note ${miss ? 'warn' : ''}">${miss ? L('needSpace') : t('vg.estimate', { t: fmtDur(SEC.load + Math.ceil(n / 2) * SEC[st.size] * cfgX(st)), n })}</span>
      <button type="button" class="btn p" data-act="start" ${miss ? 'disabled' : ''}>${t('ch.start')}</button>` :
    dirty() && !running() ? html`<span class="note">${L('rerunNote')}</span><button type="button" class="btn p" data-act="rerun">${t('ch.rerun')}</button>` : html``).__raw;
}

// ---- right ----
function tile(c, i) {
  if (c.pending) return html`<div class="ph sk" style="aspect-ratio: ${K.ratio}"></div>`;
  if (c.failed) return html`<div class="ph" style="aspect-ratio: ${K.ratio}; display: flex; align-items: center; justify-content: center"><span class="sub" style="color: ${failColor(c.stopped)}">${failWord(c.stopped)}</span></div>`;
  const ref = c.registered ? { registered: c.registered } : { job: c.job, name: c.name };
  const isBase = sameRef(st.base, ref), saved = !c.registered && st.saved.some((x) => sameRef(x, ref));
  return html`<div class="col ${enter('lo:' + (c.registered || c.job + '/' + c.name))}" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid ${isBase ? 'var(--blue-line)' : '#24242A'}; border-radius: 10px">
    <a class="ph" href="${c.url}" target="_blank" style="display: block; aspect-ratio: ${K.ratio}"><img src="${c.url}" alt="" loading="lazy">
      <span class="mono" style="position: absolute; left: 6px; top: 6px; padding: 1px 7px; border-radius: 6px; background: #0E0E10CC; font-size: 11px; color: var(--fg2)">${c.registered ? L('registered') : 'seed ' + c.seed}</span></a>
    <div class="row" style="gap: 2px">
      ${c.registered ? '' : html`<button type="button" class="ib m" data-act="star" data-i="${i}" aria-pressed="${saved}" aria-label="${t('ch.saveCand')}" title="${t('ch.saveCand')}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="${saved ? '#FACC15' : 'none'}" stroke="${saved ? '#FACC15' : 'currentColor'}" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg></button>`}
      <button type="button" class="ib m" data-act="base" data-i="${i}" aria-pressed="${isBase}" aria-label="${L('baseCand')}" title="${L('baseCand')}">
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="${isBase ? '#8AB4F8' : 'none'}"/></svg></button></div></div>`;
}
function viewTile(v, i) {
  const j = job(v.job), f = pngOf(j), busy = active(j);
  const old = v.job && (v.from !== JSON.stringify(st.base) || v.at !== v.text.trim());
  return html`<div class="col" style="gap: 4px; ${old ? 'opacity: .45' : ''}">
    <div class="ph ${busy ? 'sk' : ''}" style="aspect-ratio: ${K.ratio}">
      ${f && !busy ? html`<a class="${enter('lov:' + f.sha)}" href="${jobFileUrl(j, f.name)}" target="_blank"><img src="${jobFileUrl(j, f.name)}?v=${f.sha.slice(0, 8)}" alt="" loading="lazy"></a>` : ''}
      ${j && j.state === 'failed' ? html`<span class="sub" style="position: absolute; inset: 40% 0; text-align: center; color: ${failColor(isStopped(j))}">${failWord(isStopped(j))}</span>` : ''}
      ${st.base && !busy && v.text.trim() ? html`<button type="button" class="ib" data-act="redoView" data-i="${i}" aria-label="${t('ch.redoOne')}" title="${v.job ? t('ch.redoOne') : t('lo.makeView')}"
        style="position: absolute; right: 4px; bottom: 4px; width: 28px; height: 28px; background: #16161AE6">${icon(v.job ? 'redo' : 'play', 13)}</button>` : ''}</div>
    <span class="sub" style="font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${i + 1} · ${v.text || '—'}${old ? ' · ' + t('vg.st.old') : ''}</span></div>`;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const a = stage('a'), v = stage('v');
  const cs = candidates();
  const steps = [[L('anchor'), a]].concat(st.views.length ? [[L('views'), v]] : []);
  const unsaved = cs.filter((c) => c.url && !c.registered && !sameRef(st.base, c) && !st.saved.some((x) => sameRef(x, c))).length;
  const canRegister = st.base && !running();
  el.innerHTML = html`
    <div class="row" style="justify-content: space-between; min-height: 40px">
      <ol class="row" style="list-style: none; margin: 0; padding: 0; gap: 8px" aria-label="${t('ch.progress')}">${steps.map(([label, x]) => html`
        <li class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
          <span class="dot ${stDot(x)}"></span>${label}<span class="sub mono">${stText(x)}</span></li>`)}</ol>
      <div class="toprow-end">${running() ? (ui.askStop ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.askStop')}</span>
          <button type="button" class="btn ghost" data-act="noStop" style="min-height: 32px">${t('common.cancel')}</button>
          <button type="button" class="btn p" data-act="yesStop" style="min-height: 32px">${t('jobs.stop')}</button></div>` :
        html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>`) : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    ${dirty() ? html`<div class="banner warn" role="status">${L('dirty')}</div>` : ''}
    <section class="card" style="${dirty() ? 'opacity: .45' : ''}">
      <div class="row"><span class="num">1</span><h2 class="h2">${L('anchor')}</h2>
        ${started() ? html`<span class="sub mono">${t('ch.candN', { n: cs.filter((c) => c.url).length })}</span>` : ''}<div class="grow"></div>
        <span class="sub mono">${stText(a)}</span>
        ${a.state !== 'wait' && !running() ? html`<button type="button" class="ib m" data-act="more" aria-label="${t('ch.more')}" title="${t('vg.moreTip', { n: st.cands })}">${icon('plus')}</button>` : ''}</div>
      ${a.state === 'run' ? html`<div class="bar" style="margin-top: 12px"><div style="width: ${a.pct}%"></div></div>` : ''}
      <p class="sub" style="margin: 6px 0 0 38px; color: ${a.state === 'pick' ? 'var(--warn-fg)' : 'var(--fg4)'}">${a.state === 'pick' ? L('pickBase') : L('anchorRule', { n: st.cands })}</p>
      ${cs.length ? html`<div class="grid" style="margin-top: 14px; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr))">${cs.map(tile)}</div>` : ''}
    </section>
    <section class="card" style="${st.views.length ? '' : 'opacity: .55'}">
      <div class="row"><span class="num">2</span><h2 class="h2">${L('views')}</h2><span class="sub">${L('viewsMeta')}</span><div class="grow"></div><span class="sub mono">${stText(v)}</span></div>
      ${st.views.length ? html`<div style="margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px">${st.views.map(viewTile)}</div>` :
        html`<p class="sub" style="margin: 6px 0 0 38px">${L('noViews')}</p>`}
    </section>
    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('ch.askClose', { n: unsaved })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button><button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between; padding-top: 4px"><p class="sub" style="margin: 0">${L('registerNote')}</p>
        <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="close" ${running() ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="register" ${canRegister ? '' : 'disabled'}>${L('register', { code: st.code })}</button></div></div>` : ''}`.__raw;
  renderActions();
}
function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(IMAGE_MENU(), K.route)}
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
    case 'set':
      if (d.k === 'kind') st.card.kind = d.v; else st[d.k] = ['cands', 'size'].includes(d.k) ? Number(d.v) : d.v;
      renderSet(); renderOut(); return save();
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'en': ui.en = !ui.en; if (ui.en) await loadEnglish(); return renderSet();
    case 'enReset': st.prompt = null; await loadEnglish(); renderSet(); return renderOut();
    case 'addView': st.views.push({ text: '', kind: 'view_' + (st.views.reduce((m, v) => Math.max(m, Number(v.kind.split('_')[1])), 0) + 1), job: null }); renderSet(); renderOut(); return save();
    case 'delView': { const v = st.views.splice(Number(d.i), 1)[0]; if (v.job) st.oldViewJobs.push(v.job); renderSet(); renderOut(); return save(); }
    case 'start': return start();
    case 'rerun': await act(async () => { await saveCard(); st.cardAt = cardKey(); st.base = null; await submitAnchors(); await save(); }); await poll(); return renderAll();
    case 'more': await act(() => submitAnchors()); await save(); await poll(); return renderOut();
    case 'star': case 'base': {
      const c = candidates()[Number(d.i)];
      const ref = c.registered ? { registered: c.registered } : { job: c.job, name: c.name };
      if (d.act === 'base') st.base = ref;
      else st.saved = st.saved.some((x) => sameRef(x, ref)) ? st.saved.filter((x) => !sameRef(x, ref)) : st.saved.concat([ref]);
      renderOut(); return save();
    }
    case 'redoView': await act(() => submitView(Number(d.i))); await save(); await poll(); return renderOut();
    case 'stop': ui.askStop = true; return renderOut();
    case 'noStop': ui.askStop = false; return renderOut();
    case 'yesStop': ui.askStop = false; for (const id of allJobs().filter((x) => active(job(x)))) await act(() => srv(`/jobs/${id}/cancel`, {})); await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; if (await act(() => api('/api/discard', { code: st.code }))) return reset(); return renderOut();
    case 'register': {
      const r = await act(() => api('/api/register', { code: st.code }));
      if (r) { toast(t('ch.registered2', { code: st.code, n: r.saved.length })); return reset(); }
      return;
    }
    default:
  }
}
function onInput(e) {
  const d = e.target.dataset;
  if (cfgInput(e, st)) renderActions();
  else if (d.card !== undefined) {
    st.card[d.card] = e.target.value;
    if (d.card === 'name') { const h = root.querySelector('[data-set] h1'); if (h) h.firstChild.textContent = (e.target.value || L('new')) + ' '; }
    if (started()) renderOut(); else renderActions();
  } else if (d.vname !== undefined) {   // the view's short name: the script's `L01 소파` (docs/script.md)
    st.views[Number(d.vname)].name = e.target.value.trim();
    clearTimeout(onInput.t);
    onInput.t = setTimeout(save, 800);
    return;
  } else if (d.view !== undefined) {
    st.views[Number(d.view)].text = e.target.value;
    renderOut();
  } else if (d.en !== undefined) {
    st.prompt = e.target.value.trim() === ui.enText.trim() ? null : e.target.value;
    renderActions();
  } else if (d.num !== undefined) {
    st[d.num] = Number(e.target.value) || 0;
  } else return;
  clearTimeout(onInput.timer);
  if (st.code) onInput.timer = setTimeout(save, 800);
}
async function loadEnglish() {
  const c = st.card;
  const r = await act(() => srv('/prompt', K.cat === 'props' ? { type: K.anchor, options: { what: c.what || 'object', desc: c.desc || '…' } } :
    { type: 'place.anchor', options: { kind: c.kind || 'interior', space: c.space || '…', scale: c.scale || '', body: c.body || '' } }));
  ui.enText = (r && r.prompt) || '';
}
async function reset() {
  st = fresh(); detail = {}; ui = freshUi();
  history.replaceState(null, '', K.route);
  await load(null);
}
async function load(code) {
  open = await act(() => api('/api/sessions?cat=' + K.cat)) || [];
  if (code) {
    const r = await act(() => api('/api/session?code=' + encodeURIComponent(code)));
    if (r && r.session && r.session.code) st = Object.assign(fresh(), r.session);
    else if (r) {   // a registered location opened again: its card, its anchor as the starting ●
      st = fresh();
      st.code = code;
      for (const k of CARD_KEYS) if (r.card[k]) st.card[k] = r.card[k];
      const an = r.images.filter((i) => i.kind === K.master).sort((a, b) => b.v - a.v);
      if (an.length) { st.registered = an[0].name; st.base = { registered: an[0].name }; st.cardAt = cardKey(); }
    }
  }
  if (!st.code) codePreview = ((await act(() => api(`/api/nextcode?g=${K.g}&r=`))) || {}).code || '';
  renderAll();
}

export default {
  async mount(el, { query, parts }) {
    root = el;
    const k = parts && parts[1] === 'prop' ? CFG.prop : CFG.location;
    if (k !== K) { K = k; CARD_KEYS = K.keys; st = null; }
    if (!st || (query.get('code') && query.get('code') !== st.code)) { st = fresh(); ui = freshUi(); }
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', (e) => { if (e.target.dataset.num !== undefined) st[e.target.dataset.num] = Number(e.target.value) || 0; });
    await load(query.get('code') || st.code);
  },
  async update() {
    if (!root || !st) return;
    for (const id of allJobs()) if (!job(id)) { try { detail[id] = await srv('/jobs/' + id); } catch (e) { /* away */ } }
    await auto();
    renderOut();
  },
  unmount() { root = null; },
};
