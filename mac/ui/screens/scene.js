// 이미지 › 워크플로 › 장면 (분경도 = a cut's first frame, knowhow §3 · §8.4): two ways, two panes.
//   한 사람 — scene.compose: image 1 = the person's picture at the shot size (원판 = medium, 얼굴 칸 · 표정 = close-up),
//             image 2 = the place → candidates.
//   여러 사람 — scene.plate: everyone seated on the wide place at once → ● plate; then cuts drawn on it (drag a box),
//             each cut re-rendered sharp with the people's pictures (scene.zoom) — same room and seats in every cut.
// ☆ → this project's 장면 (the video screens' first frames). The draft lives in projects/<p>/drafts/<id>.json.
import { S, t, html, icon, api, srv, act, toast, poll, filesUrl, jobFileUrl, fmtDur, codeLabel, isStopped, failWord, failColor, tOr } from '../app.js';
import { defaults, cfgFields, cfgInput, cfgOpts, cfgX } from '../models.js';
import { sideMenu, IMAGE_MENU } from './wip.js';
import { libraryPicker } from './imagegen.js';

const SIZE = { h: [1920, 1088], v: [1088, 1920] };
const SEC = { one: 40, load: 30 };   // seconds per 1920x1088 edit image on one node (estimate), model load
const FRAME = {   // the shot size, said about the person (knowhow §3: frame)
  wide: 'full length, from the top of {pos} head down to {pos} shoes',
  medium: 'a medium shot from the top of {pos} head to {pos} waist',
  close: 'a close-up from the top of {pos} head to {pos} chest',
};

let root = null, st = null, chars = [], lib = null, open = [], meta = {}, ui = {}, drag = null;
const freshUi = () => ({ pickPlace: false, askClose: false, adv: false });
function fresh() {
  const d = defaults('qi21');
  return { id: null, screen: 'scene', title: '', method: 'one', place: null, shot: 'medium', orient: 'h', light: '', extra: '', plate: '',
    people: [person()], cands: d.cands, seed: 1, retry: d.autoRetry, cfg: d.cfg, neg: d.neg, jobs: [], saved: [], base: null, cuts: [] };
}
function person(code) { return { code: code || '', pic: '', frame: '', action: '', clothing: '', behind: '', where: '' }; }

// ---- derived ----
const jobOf = (id) => S.jobs.find((j) => j.id === id) || null;
const started = () => st.jobs.length > 0;
const charOf = (code) => chars.find((c) => c.code === code);
const genderOf = (code) => (charOf(code) || {}).card ? charOf(code).card.gender || (code[0] === 'F' ? 'female' : 'male') : (code[0] === 'F' ? 'female' : 'male');
const pos = (code) => genderOf(code) === 'female' ? 'her' : 'his';
const childOf = (code) => !!(((charOf(code) || {}).card || {}).child);   // 기능 13: girl / boy
const nounOf = (code) => genderOf(code) === 'female' ? (childOf(code) ? 'girl' : 'woman') : (childOf(code) ? 'boy' : 'man');
const rel = (j, name) => (j.project ? `projects/${j.project}/jobs/${j.id}` : `jobs/${j.id}`) + '/' + name;
const isSaved = (c) => st.saved.some((x) => x.job === c.job && x.name === c.name);
const people = () => st.people.filter((p) => p.code);
function pics(code) {   // the pictures a person can come in: master, face panel, expressions (latest versions)
  const c = charOf(code);
  if (!c) return [];
  const last = (k) => c.images.filter((i) => i.kind === k).sort((a, b) => b.v - a.v)[0];
  const kinds = ['ref', 'sh_head_front'].concat([...new Set(c.images.filter((i) => i.kind.startsWith('ex_')).map((i) => i.kind))]);
  return kinds.map(last).filter(Boolean).map((i) => ({ rel: `characters/${code}/${i.name}`, label: i.kind === 'ref' ? t('sc.picRef') : i.kind === 'sh_head_front' ? t('sc.picFace') : tOr('ex.' + i.kind.slice(3), i.kind.slice(3)) }));
}
const picOf = (p) => p.pic || (pics(p.code)[st.shot === 'close' ? 1 : 0] || pics(p.code)[0] || {}).rel;
const cutPic = (c, p, k) => (c.pics || [])[k] || picOf(p);   // a cut may take another picture of the person (표정 칸 for a close-up)
const autoFrame = (p, shot) => FRAME[shot].replace(/\{pos\}/g, pos(p.code));
const key = () => JSON.stringify([st.method, st.place && st.place.rel, st.shot, st.orient, st.light, st.extra, st.plate, !!st.blur, people().map((p) => [p.code, picOf(p), p.frame, p.action, p.clothing, p.behind])]);
const countLine = () => people().length > 1 ? `There are exactly ${people().length} people in the image, each appearing only once.` : '';

function problems() {
  const p = [];
  if (!(S.status && S.status.settings && S.status.settings.project)) p.push(t('vg.needProject'));   // cuts and drafts live in the project
  if (!st.place) p.push(t('sc.needPlace'));
  if (!people().length) p.push(t('sc.needPerson'));
  if (st.method === 'one') {
    const q = people()[0];
    if (q && (!q.action.trim() || !q.behind.trim())) p.push(t('sc.needAction'));
  } else {
    if (!st.plate.trim()) p.push(t('sc.needPlate'));
    if (people().some((q) => !q.action.trim())) p.push(t('sc.needActions'));
  }
  return p;
}

// ---- jobs ----
function options() {
  const [w, h] = SIZE[st.orient];
  const ps = people();
  const o = { width: w, height: h, count: st.cands, retries: st.retry ? 3 : 0, body_retries: 0, asset_code: ps.map((p) => p.code).join(' '),
    asset_kind: st.title.trim() || t('side.scene'), ...cfgOpts(st) };
  if (st.method === 'one') {
    const p = ps[0];
    if (st.blur && st.shot !== 'wide') o.blur = true;   // 기능 14: shallow depth of field (close-up: on by default — design §2)
    return Object.assign(o, { gender: genderOf(p.code), child: childOf(p.code), frame: p.frame || autoFrame(p, st.shot), action: p.action, behind: p.behind,
      clothing: p.clothing, shot: st.shot, light: st.light, extra: st.extra });
  }
  return Object.assign(o, { plate: st.plate, extra: st.extra || countLine(),
    people: ps.map((p) => ({ gender: genderOf(p.code), child: childOf(p.code), frame: p.frame || autoFrame(p, 'wide'), action: p.action, clothing: p.clothing })) });
}
async function generate() {
  const pr = problems();
  if (pr.length) return toast(pr[0], true);
  const k = key();
  const used = st.jobs.filter((x) => x.key === k).reduce((a, x) => a + x.n, 0);
  const o = Object.assign(options(), { seed: Number(st.seed) + used });
  const ps = people();
  const images = st.method === 'one' ? [picOf(ps[0]), st.place.rel] : [st.place.rel].concat(ps.map((p) => (pics(p.code)[0] || {}).rel));
  const r = await act(() => api('/api/image', { type: st.method === 'one' ? 'scene.compose' : 'scene.plate', images, options: o }));
  if (!r) return;
  st.jobs.push({ job: r.id, key: k, n: Number(st.cands) });
  await save();
  await poll();
  renderAll();
}
const inCut = (c) => people().map((p, k) => [p, k]).filter(([, k]) => !c.out || !c.out.includes(k));   // who this cut shows
async function makeCut(i) {   // one cut of the ● plate: crop on the Mac, re-render it with the pictures of the people in it
  const c = st.cuts[i], ps = inCut(c);
  if (!ps.length) return toast(t('sc.needWho'), true);
  const bj = jobOf(st.base.job);
  if (!c.rel) {
    const r = await act(() => api('/api/crop', { rel: rel(bj, st.base.name), box: c.box }));
    if (!r) return;
    c.rel = r.rel;
  }
  const [w, h] = SIZE[c.orient || st.orient];
  const o = { width: w, height: h, count: 1, seed: Number(st.seed) + 200 + i + 1000 * (c.tries || 0), retries: st.retry ? 3 : 0, ...cfgOpts(st),
    what: st.plate, frame: c.frame || FRAME.medium.replace(/\{pos\}/g, 'the person\'s'),
    people: ps.map(([p, k]) => ({ gender: genderOf(p.code), child: childOf(p.code), where: ps.length === 1 ? 'the ' + nounOf(p.code) : p.where || defaultWhere(p, k), action: (c.actions || [])[k] || '', clothing: p.clothing })),
    asset_code: ps.map(([p]) => p.code).join(' '), asset_kind: (st.title.trim() || t('side.scene')) + ' · ' + t('sc.cutN', { k: i + 1 }) };
  const images = [c.rel].concat(ps.map(([p, k]) => cutPic(c, p, k)));
  const r = await act(() => api('/api/image', { type: 'scene.zoom', images, options: o }));
  if (!r) return;
  if (c.job) st.oldJobs = (st.oldJobs || []).concat([c.job]);
  c.job = r.id; c.tries = (c.tries || 0) + 1; c.from = JSON.stringify(st.base);
  await save();
  await poll();
  renderOut();
}
function defaultWhere(p, k) {
  const n = people().length, w = nounOf(p.code);
  return n === 1 ? 'the ' + w : `the ${w} ${k === 0 ? 'on the left' : k === n - 1 ? 'on the right' : 'in the middle'}`;
}

// ---- persistence ----
async function save() {
  if (!st.id && !started()) return;
  const r = await act(() => api('/api/doc', { kind: 'drafts', id: st.id, doc: st }));
  if (r && !st.id) { st.id = r.id; history.replaceState(null, '', '#/image/scene?d=' + r.id); }
}
let saveTimer = 0;
const saveSoon = () => { clearTimeout(saveTimer); if (st.id) saveTimer = setTimeout(save, 800); };
async function finish() {
  const title = (st.title.trim() || 'scene').replace(/\s+/g, '_');
  // a cut keeps its box number in the name (<title>_컷3 — the sequence import shows which cut it is, boss, session #14);
  // whole-scene and single-person candidates are numbered on their own
  let n = 0;
  const items = st.saved.filter((c) => jobOf(c.job)).map((c) => {
    const k = c.cut ?? st.cuts.findIndex((x) => x.job === c.job);
    return { job: c.job, name: c.name, title: title + '_' + (k >= 0 ? t('sc.cutFile', { k: k + 1 }) : ++n) };
  });
  const jobs = st.jobs.map((x) => x.job).concat(st.cuts.map((c) => c.job).filter(Boolean), st.oldJobs || []);
  const r = await act(() => api('/api/finish', { items, jobs, docs: st.id ? [{ kind: 'drafts', id: st.id }] : [] }));   // ☆ on 닫기 too
  if (!r) return;
  toast(r.saved.length ? t('vg.savedN', { n: r.saved.length }) : t('vg.closed'));
  st = fresh(); ui = freshUi(); meta = {};
  history.replaceState(null, '', '#/image/scene');
  await loadOpen();
  await poll();
  renderAll();
}

// ---- left ----
function seg(k, opts, cur, label) {
  return html`<div class="seg" role="group" aria-label="${label}">${opts.map(([v, l]) => html`
    <button type="button" data-act="set" data-k="${k}" data-v="${v}" aria-pressed="${String(cur) === String(v)}">${l}</button>`)}</div>`;
}
function personBox(p, i) {
  const one = st.method === 'one';
  const opts = chars.filter((c) => c.master && (c.code === p.code || !st.people.some((q) => q.code === c.code)));
  return html`<div class="col" style="gap: 8px; padding: 10px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
    <div class="row" style="gap: 8px">
      ${p.code && picOf(p) ? html`<img src="${filesUrl(picOf(p))}" alt="" style="width: 40px; height: 52px; object-fit: cover; border-radius: 6px">` : ''}
      <select class="in" data-p="${i}" data-f="code" style="width: 110px; min-height: 34px" aria-label="${t('vg.people')}"><option value="">${t('sc.pickPerson')}</option>
        ${opts.map((c) => html`<option value="${c.code}" ${c.code === p.code ? 'selected' : ''}>${codeLabel(c)}</option>`)}</select>
      ${p.code ? html`<select class="in" data-p="${i}" data-f="pic" style="flex: 1 1 120px; width: auto; min-height: 34px" aria-label="${t('sc.pic')}">
        ${pics(p.code).map((x) => html`<option value="${x.rel}" ${x.rel === picOf(p) ? 'selected' : ''}>${x.label}</option>`)}</select>` : ''}
      ${!one && st.people.length > 1 ? html`<button type="button" class="ib m" data-act="delPerson" data-i="${i}" aria-label="${t('sw.delPerson')}" title="${t('sw.delPerson')}">${icon('x', 14)}</button>` : ''}</div>
    ${p.code ? html`
      <input class="in" data-p="${i}" data-f="frame" value="${p.frame}" placeholder="${autoFrame(p, one ? st.shot : 'wide')}" aria-label="${t('sc.frame')}" title="${t('sc.frame')}">
      <textarea class="in" rows="2" data-p="${i}" data-f="action" placeholder="${one ? t('sc.ph.action') : t('sc.ph.seat')}" aria-label="${t('sc.action')}">${p.action}</textarea>
      <input class="in" data-p="${i}" data-f="clothing" value="${p.clothing}" placeholder="${t('sc.ph.clothing')}" aria-label="${t('sc.clothing')}">
      ${one ? html`<input class="in" data-p="${i}" data-f="behind" value="${p.behind}" placeholder="${t('sc.ph.behind')}" aria-label="${t('sc.behind')}">` :
        html`<input class="in" data-p="${i}" data-f="where" value="${p.where}" placeholder="${defaultWhere(p, i)}" aria-label="${t('sc.where')}" title="${t('sc.whereTip')}">`}` : ''}</div>`;
}
function renderSet() {
  const el = root && root.querySelector('[data-set]');
  if (!el) return;
  const pr = problems();
  const one = st.method === 'one';
  el.innerHTML = html`
    <div><div class="lbl">${t('sc.crumb')}</div><h1 class="h1">${st.title || t('sc.new')}</h1><p class="lead">${t('lead.scene')}</p></div>
    ${!started() && open.length ? html`<div class="row" style="gap: 6px"><span class="sub">${t('ch.inProgress')}</span>
      ${open.map((d) => html`<a class="chip" href="#/image/scene?d=${d.id}" style="display: inline-flex; align-items: center; text-decoration: none">${d.title || d.id}</a>`)}</div>` : ''}
    <label class="field">${t('vg.title')}<input class="in" data-k="title" value="${st.title}" placeholder="${t('sc.titlePh')}"></label>
    <div class="col" style="gap: 6px">${seg('method', [['one', t('sc.one')], ['many', t('sc.many')]], st.method, t('sc.method'))}
      <span class="sub">${one ? t('sc.oneNote') : t('sc.manyNote')}</span></div>
    <div class="sec">
      <h2 class="h3">${t('sc.place')}</h2>
      ${st.place ? html`<div class="ph" style="aspect-ratio: 16 / 9"><img src="${filesUrl(st.place.rel)}" alt="${st.place.name}" style="object-fit: contain">
          <button type="button" class="ib" data-act="clearPlace" aria-label="${t('vg.remove')}" title="${t('vg.remove')}" style="position: absolute; right: 4px; top: 4px; width: 28px; height: 28px; background: #16161AE6">${icon('x', 13)}</button></div>` :
        html`<button type="button" class="slot" data-act="pickPlace" aria-expanded="${ui.pickPlace}" style="aspect-ratio: 16 / 9">${icon('plus', 18)}<span>${t('sc.pickPlace')}</span></button>`}
      ${ui.pickPlace ? libraryPicker(Object.assign({}, lib, { chars: [] }), t('sc.pickPlaceFor')) : ''}
      ${one ? '' : html`<label class="field">${t('sc.plate')}<input class="in" data-k="plate" value="${st.plate}" placeholder="${t('sc.ph.plate')}"></label>`}</div>
    <div class="sec">
      <div class="row" style="gap: 4px 10px"><h2 class="h3">${t('vg.people')}</h2><span class="sub">${one ? t('sc.peopleOne') : t('sc.peopleMany')}</span></div>
      ${(one ? st.people.slice(0, 1) : st.people).map(personBox)}
      ${!one && st.people.length < 3 && st.people.every((q) => q.code) ? html`<button type="button" class="btn add" data-act="addPerson">${icon('plus', 14)}${t('vg.addPerson')}</button>` : ''}</div>
    <div class="sec">
      ${one ? html`<div class="field">${t('sc.shot')}${seg('shot', [['wide', t('sc.wide')], ['medium', t('sc.medium')], ['close', t('sc.close')]], st.shot, t('sc.shot'))}</div>` : ''}
      ${one && st.shot !== 'wide' ? html`<label class="tog"><input type="checkbox" data-flag="blur" ${st.blur ? 'checked' : ''}>${t('sc.blur')} <span class="sub">${t('sc.blurNote')}</span></label>` : ''}
      <div class="field">${t('sw.orient')}${seg('orient', [['h', t('vg.landscape')], ['v', t('vg.portrait')]], st.orient, t('sw.orient'))}
        <span class="sub mono">${SIZE[st.orient].join('×')}</span></div>
      <div class="row" style="gap: 10px"><span class="lbl">${t('ch.cands')}</span>${seg('cands', [[1, '1'], [2, '2'], [4, '4']], st.cands, t('ch.cands'))}<div class="grow"></div>
        <button type="button" class="ib m" data-act="adv" aria-expanded="${ui.adv}" aria-label="${t('ch.adv')}" title="${t('ch.adv')}">${icon('sliders')}</button></div>
      ${ui.adv ? html`<div class="col" style="gap: 10px; padding: 12px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px">
        ${one ? html`<label class="field">${t('sc.light')}<input class="in" data-k="light" value="${st.light}" placeholder="${t('sc.ph.light')}"></label>` : ''}
        <label class="field">${t('sc.extra')}<textarea class="in" rows="2" data-k="extra" placeholder="${one ? '' : countLine()}">${st.extra}</textarea></label>
        <label class="field">${t('ch.seed')}<input class="in mono" data-k="seed" inputmode="numeric" value="${st.seed}" style="width: 120px"></label>
        ${cfgFields(st)}
        <label class="tog"><input type="checkbox" data-flag="retry" ${st.retry ? 'checked' : ''}>${t('def.autoRetry')}</label></div>` : ''}</div>`.__raw;
  renderActions();
}
// the main action, top right of the results pane (boss, session #14): the note (what is missing · estimate), then the button
function renderActions() {
  const el = root && root.querySelector('[data-actions]');
  if (!el) return;
  const pr = problems();
  el.innerHTML = html`<span class="note ${pr.length ? 'warn' : ''}">${pr.length ? pr.join(' · ') : t('vg.estimate', { t: fmtDur(SEC.load + Math.ceil(st.cands / 2) * SEC.one * cfgX(st)), n: st.cands })}</span>
    <button type="button" class="btn p" data-act="go" ${pr.length ? 'disabled' : ''}>${started() ? t('ig.more', { n: st.cands }) : t('ch.start')}</button>`.__raw;
}

// ---- right ----
function cands() {
  const out = [];
  for (const x of st.jobs) {
    const j = jobOf(x.job);
    if (!j) continue;
    const old = x.key !== key(), m = meta[j.id] || {};
    for (const f of j.files.filter((y) => y.name.endsWith('.png') && y.synced)) {
      const mm = m[f.name] || {};
      out.push({ job: j.id, name: f.name, url: jobFileUrl(j, f.name) + '?v=' + f.sha.slice(0, 8), seed: f.name.split('_')[1].split('.')[0], old,
        sim: Array.isArray(mm.face_sim) && mm.face_sim.length ? mm.face_sim[mm.face_sim.length - 1] : null, passed: mm.passed });
    }
    for (let k = 0; k < j.count.pending + j.count.running; k++) out.push({ pending: true, pct: Math.round(j.progress * 100), run: j.count.running > k, old });
    for (let k = 0; k < j.count.failed; k++) out.push({ failed: true, stopped: isStopped(j), old });
  }
  return out;
}
const star = (sv, i, act2) => html`<button type="button" class="ib m" data-act="${act2}" data-i="${i}" aria-pressed="${sv}" aria-label="${t('ch.saveCand')}" title="${t('ch.saveCand')}">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="${sv ? '#FACC15' : 'none'}" stroke="${sv ? '#FACC15' : 'currentColor'}" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg></button>`;
function tile(c, i) {
  const ratio = st.orient === 'h' ? '16 / 9' : '9 / 16';
  if (c.pending) return html`<div class="ph sk" style="aspect-ratio: ${ratio}; display: flex; align-items: center; justify-content: center; ${c.old ? 'opacity: .45' : ''}"><span class="mono sub">${c.run ? c.pct + '%' : t('ch.st.wait')}</span></div>`;
  if (c.failed) return html`<div class="ph" style="aspect-ratio: ${ratio}; display: flex; align-items: center; justify-content: center"><span class="sub" style="color: ${failColor(c.stopped)}">${failWord(c.stopped)}</span></div>`;
  const isBase = st.base && st.base.job === c.job && st.base.name === c.name;
  return html`<div class="col" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid ${isBase ? 'var(--blue-line)' : '#24242A'}; border-radius: 10px; ${c.old ? 'opacity: .45' : ''}">
    <a class="ph" href="${c.url}" target="_blank" style="display: block; aspect-ratio: ${ratio}"><img src="${c.url}" alt="" loading="lazy" style="object-fit: contain">
      <span class="mono" style="position: absolute; left: 6px; top: 6px; padding: 1px 7px; border-radius: 6px; background: #0E0E10CC; font-size: 11px; color: var(--fg2)">seed ${c.seed}${c.old ? ' · ' + t('vg.st.old') : ''}</span></a>
    <div class="row" style="gap: 2px">${star(isSaved(c), i, 'star')}
      ${st.method === 'many' ? html`<button type="button" class="ib m" data-act="base" data-i="${i}" aria-pressed="${isBase}" aria-label="${t('sc.basePlate')}" title="${t('sc.basePlate')}">
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="${isBase ? '#8AB4F8' : 'none'}"/></svg></button>` : ''}
      <div class="grow"></div>
      ${c.sim != null ? html`<span class="sub row" style="gap: 5px" title="${t('sc.faceTip')}"><span class="dot ${c.passed ? 'ok' : 'warn'}"></span>${Number(c.sim).toFixed(2)}</span>` : ''}</div></div>`;
}
function cutTile(c, i) {
  const j = jobOf(c.job), f = j && j.files.find((x) => x.name.endsWith('.png') && x.synced), busy = j && (j.state === 'pending' || j.state === 'running');
  const old = c.job && c.from !== JSON.stringify(st.base);
  const ps = people();
  const ratio = (c.orient || st.orient) === 'h' ? '16 / 9' : '9 / 16';
  const out = f ? { job: j.id, name: f.name } : null;
  return html`<div class="col" style="gap: 6px; padding: 8px; background: var(--bg2); border: 1px solid #24242A; border-radius: 10px; ${old ? 'opacity: .45' : ''}">
    <div class="row" style="gap: 6px"><span class="num" style="width: 22px; height: 22px">${i + 1}</span>
      <input class="in" data-cut="${i}" data-f="frame" value="${c.frame || ''}" placeholder="${t('sc.ph.cutFrame')}" style="flex: 1 1 140px; width: auto; min-height: 32px" aria-label="${t('sc.frame')}">
      <button type="button" class="ib m" data-act="delCut" data-i="${i}" aria-label="${t('sq.delCut')}" title="${t('sq.delCut')}">${icon('x', 13)}</button></div>
    ${ps.map((p, k) => { const on = !c.out || !c.out.includes(k); return html`<div class="col" style="gap: 4px">
      <div class="row" style="gap: 6px; flex-wrap: nowrap">
        <button type="button" class="chip" data-act="cutWho" data-i="${i}" data-k="${k}" aria-pressed="${on}" title="${t('sc.whoTip')}" style="min-height: 30px; padding: 0 10px">${p.code}</button>
        ${on ? html`<select class="in" data-cut="${i}" data-f="pic" data-k="${k}" style="flex: 1 1 auto; width: auto; min-width: 0; min-height: 30px; font-size: 12px" aria-label="${t('sc.pic')}" title="${t('sc.cutPicTip')}">
          ${pics(p.code).map((x) => html`<option value="${x.rel}" ${x.rel === cutPic(c, p, k) ? 'selected' : ''}>${x.label}</option>`)}</select>` : html`<span class="sub">${t('sc.notInCut')}</span>`}</div>
      ${on ? html`<input class="in" data-cut="${i}" data-f="action" data-k="${k}" value="${(c.actions || [])[k] || ''}" placeholder="${t('sc.ph.cutAction2')}" style="min-height: 30px; font-size: 13px">` : ''}</div>`; })}
    <div class="ph ${busy ? 'sk' : ''}" style="aspect-ratio: ${ratio}">
      ${f && !busy ? html`<a href="${jobFileUrl(j, f.name)}" target="_blank"><img src="${jobFileUrl(j, f.name)}?v=${f.sha.slice(0, 8)}" alt="" loading="lazy" style="object-fit: contain"></a>` :
        c.rel ? html`<img src="${filesUrl(c.rel)}" alt="" style="object-fit: contain; opacity: .5">` : ''}
      ${j && j.state === 'failed' ? html`<span class="sub" style="position: absolute; inset: 44% 0; text-align: center; color: ${failColor(isStopped(j))}">${failWord(isStopped(j))}</span>` : ''}</div>
    <div class="row" style="gap: 2px">
      ${out ? star(isSaved(out), i, 'starCut') : ''}<div class="grow"></div>
      <button type="button" class="btn ${c.job ? 'ghost' : 'p'}" data-act="makeCut" data-i="${i}" style="min-height: 32px" ${busy ? 'disabled' : ''}>${icon(c.job ? 'redo' : 'play', 12)}${c.job ? t('ch.redoOne') : t('sc.makeCut')}</button></div></div>`;
}
function renderOut() {
  const el = root && root.querySelector('[data-out]');
  if (!el) return;
  const cs = cands();
  const run = cs.some((c) => c.pending);
  const many = st.method === 'many';
  const bj = st.base && jobOf(st.base.job);
  const unsaved = cs.filter((c) => c.url && !isSaved(c)).length;
  const anyActive = st.jobs.concat(st.cuts.filter((c) => c.job).map((c) => ({ job: c.job }))).map((x) => jobOf(x.job)).some((j) => j && (j.state === 'pending' || j.state === 'running'));
  el.innerHTML = html`
    <div class="row" style="min-height: 40px; justify-content: space-between">
      <span class="row" style="gap: 8px; min-height: 34px; padding: 0 12px; border: 1px solid var(--line); border-radius: 999px; background: #131316; font-size: 13px">
        <span class="dot ${run ? 'busy' : started() ? 'ok' : ''}"></span>${many ? t('sc.plateStage') : t('vg.gen')}<span class="sub mono">${run ? t('ig.running') : started() ? (many && !st.base ? t('ch.st.pick') : t('ch.st.done')) : t('ch.st.wait')}</span></span>
      <div class="toprow-end">${anyActive ? html`<button type="button" class="ib" data-act="stop" aria-label="${t('jobs.stop')}" title="${t('jobs.stopTip')}">${icon('stop')}</button>` : ''}<div class="topact" data-actions></div></div></div>
    ${!started() ? html`<p class="sub" style="margin: 0">${t('ch.fillHint')}</p>` : ''}
    <section class="card">
      <div class="row"><span class="num">1</span><h2 class="h2">${many ? t('sc.plateStage') : t('vg.gen')}</h2><span class="sub mono">${many ? 'scene.plate' : 'scene.compose'} · ${t('ch.candN', { n: cs.filter((c) => c.url && !c.old).length })}</span></div>
      <p class="sub" style="margin: 6px 0 0 38px">${many ? t('sc.plateRule') : t('ig.rule')}</p>
      ${cs.length ? html`<div class="grid" style="margin-top: 14px; grid-template-columns: repeat(auto-fill, minmax(${st.orient === 'h' ? '240px' : '150px'}, 1fr))">${cs.map(tile)}</div>` : ''}
    </section>
    ${many ? html`<section class="card" style="${st.base ? '' : 'opacity: .55'}">
      <div class="row"><span class="num">2</span><h2 class="h2">${t('sc.cuts')}</h2><span class="sub">${t('sc.cutsNote')}</span><div class="grow"></div>
        <span class="sub">${t('sc.cutOrient')}</span>${seg('cutOrient', [['h', t('vg.landscape')], ['v', t('vg.portrait')]], ui.cutOrient || st.orient, t('sw.orient'))}</div>
      ${bj ? html`<div data-cropper class="ph" style="margin-top: 12px; aspect-ratio: ${SIZE[st.orient][0]} / ${SIZE[st.orient][1]}; cursor: crosshair; touch-action: none; user-select: none">
          <img src="${jobFileUrl(bj, st.base.name)}" alt="" draggable="false" style="object-fit: fill; pointer-events: none">
          ${st.cuts.map((c, i) => html`<div aria-hidden="true" style="position: absolute; left: ${c.box[0] * 100}%; top: ${c.box[1] * 100}%; width: ${(c.box[2] - c.box[0]) * 100}%; height: ${(c.box[3] - c.box[1]) * 100}%; border: 2px solid var(--blue); background: #8AB4F81A">
            <span class="mono" style="position: absolute; left: 4px; top: 2px; font-size: 12px; color: var(--blue-fg)">${i + 1}</span></div>`)}
          <div data-drag aria-hidden="true" style="position: absolute; display: none; border: 2px dashed var(--yellow)"></div></div>
        <p class="sub" style="margin: 6px 0 0">${t('sc.dragNote')}</p>
        ${st.cuts.length ? html`<div style="margin-top: 12px; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px">${st.cuts.map(cutTile)}</div>` : ''}` :
        html`<p class="sub" style="margin: 6px 0 0 38px">${t('sc.needBase')}</p>`}
    </section>` : ''}
    ${started() ? html`
      ${ui.askClose ? html`<div class="confirm" role="alertdialog"><span>${t('vg.askClose', { n: unsaved })}</span><div class="row" style="gap: 8px">
        <button type="button" class="btn ghost" data-act="noClose">${t('common.cancel')}</button><button type="button" class="btn p" data-act="yesClose">${t('jobs.close')}</button></div></div>` : ''}
      <div class="row" style="justify-content: space-between"><p class="sub" style="margin: 0">${t('sc.saveNote')}</p>
        <div class="row" style="gap: 8px"><button type="button" class="btn ghost" data-act="close" ${anyActive ? 'disabled' : ''}>${t('jobs.close')}</button>
          <button type="button" class="btn p lg" data-act="save" ${st.saved.length && !anyActive ? '' : 'disabled'}>${icon('star', 16)}${t('ig.saveN', { n: st.saved.length })}</button></div></div>` : ''}`.__raw;
  renderActions();
}
function renderAll() {
  if (!root) return;
  root.innerHTML = html`<div class="split">${sideMenu(IMAGE_MENU(), '#/image/scene')}
    <section class="pane-set" data-set aria-label="${t('ch.settings')}"></section>
    <main class="pane-out" data-out aria-label="${t('ch.results')}"></main></div>`.__raw;
  renderSet();
  renderOut();
}

// ---- cut boxes: drag on the ● plate, the box keeps the cut's aspect ----
function boxAt(e) {
  const r = drag.el.getBoundingClientRect();
  const [ow, oh] = SIZE[ui.cutOrient || st.orient];
  let dx = e.clientX - drag.x0, dy = e.clientY - drag.y0;
  const w = Math.max(Math.abs(dx), Math.abs(dy) * ow / oh), h = w * oh / ow;
  let x0 = dx < 0 ? drag.x0 - w : drag.x0, y0 = dy < 0 ? drag.y0 - h : drag.y0;
  x0 = Math.min(Math.max(x0, r.left), r.right - Math.min(w, r.width));
  y0 = Math.min(Math.max(y0, r.top), r.bottom - Math.min(h, r.height));
  const fx = (x0 - r.left) / r.width, fy = (y0 - r.top) / r.height;
  return [fx, fy, Math.min(1, fx + Math.min(w, r.width) / r.width), Math.min(1, fy + Math.min(h, r.height) / r.height)];
}
// people whose face on the ● plate (its result json, worker._plate_people) lies outside box b start off in that cut
// (boss, session #14); a face not found or an older plate leaves them on, and a box with no one's face keeps all on
function outOf(b) {
  const pf = st.base && ((meta[st.base.job] || {})[st.base.name] || {}).people;
  const ps = people();
  if (!pf) return [];
  const out = ps.map((p, k) => k).filter((k) => {
    const f = pf[k];
    if (!f) return false;
    const cx = (f[0] + f[2]) / 2, cy = (f[1] + f[3]) / 2;
    return cx < b[0] || cx > b[2] || cy < b[1] || cy > b[3];
  });
  return out.length === ps.length ? [] : out;
}
function onDown(e) {
  const el = e.target.closest('[data-cropper]');
  if (!el) return;
  drag = { el, x0: e.clientX, y0: e.clientY, box: null };
  el.setPointerCapture(e.pointerId);
}
function onMove(e) {
  if (!drag) return;
  drag.box = boxAt(e);
  const d = drag.el.querySelector('[data-drag]'), b = drag.box;
  Object.assign(d.style, { display: 'block', left: b[0] * 100 + '%', top: b[1] * 100 + '%', width: (b[2] - b[0]) * 100 + '%', height: (b[3] - b[1]) * 100 + '%' });
}
function onUp() {
  if (!drag) return;
  const b = drag.box;
  drag = null;
  if (b && b[2] - b[0] > 0.04) { st.cuts.push({ box: b, orient: ui.cutOrient || st.orient, frame: '', actions: [], out: outOf(b) }); save(); }
  renderOut();
}

// ---- events ----
async function onClick(e) {
  const a = e.target.closest('[data-act]');
  if (!a || a.disabled) return;
  const d = a.dataset, i = Number(d.i);
  switch (d.act) {
    case 'set':
      if (d.k === 'cutOrient') { ui.cutOrient = d.v; return renderOut(); }
      st[d.k] = d.k === 'cands' ? Number(d.v) : d.v;
      if (d.k === 'shot') st.blur = d.v === 'close';   // CU·MCU blurred by default, wider in natural focus (boss, session #18)
      if (d.k === 'method' && d.v === 'one') st.people = st.people.slice(0, 1);
      renderSet(); renderOut(); return saveSoon();
    case 'pickPlace': ui.pickPlace = !ui.pickPlace; return renderSet();
    case 'unpick': ui.pickPlace = false; return renderSet();
    case 'libPick': st.place = { rel: d.rel, name: d.name }; ui.pickPlace = false; renderSet(); renderOut(); return saveSoon();
    case 'clearPlace': st.place = null; renderSet(); return saveSoon();
    case 'addPerson': st.people.push(person()); return renderSet();
    case 'delPerson': st.people.splice(i, 1); renderSet(); renderOut(); return saveSoon();
    case 'adv': ui.adv = !ui.adv; return renderSet();
    case 'go': return generate();
    case 'star': case 'base': {
      const c = cands()[i];
      if (d.act === 'base') st.base = { job: c.job, name: c.name };
      else st.saved = isSaved(c) ? st.saved.filter((x) => !(x.job === c.job && x.name === c.name)) : st.saved.concat([{ job: c.job, name: c.name }]);
      renderOut(); return save();
    }
    case 'starCut': {
      const c = st.cuts[i], j = jobOf(c.job), f = j && j.files.find((x) => x.name.endsWith('.png') && x.synced);
      if (!f) return;
      const ref = { job: j.id, name: f.name, cut: i };
      st.saved = isSaved(ref) ? st.saved.filter((x) => !(x.job === ref.job && x.name === ref.name)) : st.saved.concat([ref]);
      renderOut(); return save();
    }
    case 'makeCut': return makeCut(i);
    case 'cutWho': {
      const c = st.cuts[i], k = Number(d.k);
      c.out = (c.out || []).includes(k) ? c.out.filter((x) => x !== k) : (c.out || []).concat([k]);
      renderOut(); return save();
    }
    case 'delCut': { const c = st.cuts.splice(i, 1)[0]; if (c.job) st.oldJobs = (st.oldJobs || []).concat([c.job]); renderOut(); return save(); }
    case 'stop':
      for (const id of st.jobs.map((x) => x.job).concat(st.cuts.map((c) => c.job).filter(Boolean))) {
        const j = jobOf(id);
        if (j && (j.state === 'pending' || j.state === 'running')) await act(() => srv(`/jobs/${id}/cancel`, {}));
      }
      await poll(); return;
    case 'close': ui.askClose = true; return renderOut();
    case 'noClose': ui.askClose = false; return renderOut();
    case 'yesClose': ui.askClose = false; return finish();
    case 'save': return finish();
    default:
  }
}
function onInput(e) {
  const d = e.target.dataset;
  if (cfgInput(e, st)) { renderActions(); return saveSoon(); }
  if (d.p !== undefined && e.target.tagName !== 'SELECT') st.people[Number(d.p)][d.f] = e.target.value;
  else if (d.cut !== undefined) {
    if (e.target.tagName === 'SELECT') return;   // a cut's picture: onChange
    const c = st.cuts[Number(d.cut)];
    if (d.f === 'frame') c.frame = e.target.value;
    else { c.actions = c.actions || []; c.actions[Number(d.k)] = e.target.value; }
  } else if (d.k && e.target.tagName !== 'SELECT') st[d.k] = d.k === 'seed' ? Number(e.target.value) || 0 : e.target.value;
  else return;
  if (d.k === 'title') { const h = root.querySelector('[data-set] h1'); if (h) h.textContent = st.title || t('sc.new'); }
  renderActions();
  saveSoon();
}
function onChange(e) {
  const d = e.target.dataset;
  if (d.p !== undefined && e.target.tagName === 'SELECT') {
    const p = st.people[Number(d.p)];
    p[d.f] = e.target.value;
    if (d.f === 'code') p.pic = '';
    renderSet(); renderOut(); return saveSoon();
  }
  if (d.cut !== undefined && e.target.tagName === 'SELECT') {
    const c = st.cuts[Number(d.cut)];
    c.pics = c.pics || [];
    c.pics[Number(d.k)] = e.target.value;
    renderOut(); return saveSoon();
  }
  if (d.flag) { st[d.flag] = e.target.checked; if (d.flag === 'blur') renderOut(); return saveSoon(); }
  if (d.file === 'ref' && e.target.files[0]) return uploadPlace(e.target.files[0]);
}
async function uploadPlace(f) {
  const r = await act(async () => {
    const res = await fetch('/api/assets', { method: 'POST', headers: { 'X-Name': encodeURIComponent(f.name) }, body: f });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.status);
    return j;
  });
  if (r) { st.place = { rel: 'assets/' + r.saved.split('/assets/').pop(), name: f.name }; ui.pickPlace = false; saveSoon(); }
  renderAll();
}
async function loadOpen() {
  open = (await act(() => api('/api/docs?kind=drafts')) || []).filter((d) => d && d.screen === 'scene' && d.id !== st.id);
}
async function loadMeta() {
  for (const x of st.jobs) {
    const j = jobOf(x.job);
    const n = j ? j.files.filter((f) => f.synced && f.name.endsWith('.json')).length : 0;
    if (j && n && (!meta[j.id] || meta[j.id]._n !== n)) {
      try { meta[j.id] = Object.assign(await api('/api/meta?job=' + j.id), { _n: n }); } catch (e) { /* next poll */ }
    }
  }
}

export default {
  async mount(el, { query }) {
    root = el;
    el.addEventListener('click', onClick);
    el.addEventListener('input', onInput);
    el.addEventListener('change', onChange);
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    const shared = (await act(() => api('/api/library?scope=shared'))) || {};
    chars = shared.characters || [];
    lib = { chars: [], locs: shared.locations || [], scenes: (((await act(() => api('/api/library?scope=project'))) || {}).items || []).filter((i) => i.kind === 'scenes') };
    const id = query.get('d');
    if (id) {
      const d = await act(() => api('/api/doc?kind=drafts&id=' + encodeURIComponent(id)));
      st = d && d.id ? Object.assign(fresh(), d) : fresh();
      ui = freshUi();
    } else if (!st) { st = fresh(); ui = freshUi(); }
    if (query.get('code') && !started()) st.people[0] = person(query.get('code'));
    await loadOpen();
    await loadMeta();
    renderAll();
  },
  async update() {
    if (!root || !st || drag) return;
    await loadMeta();
    if (root.contains(document.activeElement) && document.activeElement.matches('[data-out] input, [data-out] textarea')) return;
    renderOut();
  },
  unmount() { root = null; },
};
