// 대본 제작의 단계 잇기 (design §12 기능 16): a sequence made from the script (sq.script) holds its first frames
// (sq.frames, from the 화 파일) and its cuts. Once started (sq.prod.started) each step runs on: frame job → ● (자동
// 진행: the first candidate that passed its face check, else the best, else the first; 단계별: the user's pick) → the
// picture into the project library → the first / last frame of every cut that uses it → the cut's video candidates
// → ● → refine (cut.js). advance() is one round over one sequence; the open sequence screen runs it on its own
// sequence, tick() (every poll, whatever screen is open) on the other started ones, one per poll. Jobs already queued
// keep running with the browser closed; the next steps wait for it (one tab drives — Web Locks).
import { S, api, t, fetchJobs } from './app.js';
import * as C from './screens/cut.js';

const jobOf = (id) => (id && S.jobs.find((j) => j.id === id)) || null;
const synced = (j, name) => j && j.files.find((f) => f.name === name && f.synced);
const live = (j) => j && (j.state === 'pending' || j.state === 'running');
const project = () => S.status && S.status.settings && S.status.settings.project;
export const specKey = (f) => JSON.stringify(f.spec);
const pngs = (j) => j ? j.files.filter((f) => /^\d+_\d+\.png$/.test(f.name)).map((f) => f.name).sort() : [];

// one frame's state: done (● picked, picture saved) · pick (results, no ●) · run · failed · wait (dependency) · todo
export function frameState(sq, f, ext) {
  if (f.rel && f.key === specKey(f)) return 'done';
  const jobs = (f.jobs || []).filter((x) => x.key === specKey(f)).map((x) => jobOf(x.job)).filter(Boolean);
  if (jobs.some(live) || jobs.some((j) => j.state === 'done' && !pngs(j).every((n) => synced(j, n)))) return 'run';
  if (jobs.some((j) => j.state === 'done' && pngs(j).length)) return 'pick';
  if (jobs.length && jobs.every((j) => j.state === 'failed')) return 'failed';
  return deps(sq, f, ext || EXT) ? 'todo' : 'wait';
}
// what a frame needs first: a box its plate's ● (and the faces found on it), an edit / file the frames it names —
// in this sequence or (reused across scenes) another one's (ext, from /api/script/frames)
function deps(sq, f, ext = {}) {
  const fr = Object.fromEntries((sq.frames || []).map((x) => [x.id, x]));
  const ok = (id) => (fr[id] ? fr[id].rel && fr[id].key === specKey(fr[id]) : !!(ext[id] && ext[id].rel));
  if (f.spec.type === 'box') return ok(f.spec.plate);
  const refs = (f.spec.images || []).concat(f.spec.file ? [f.spec.file] : []);
  return refs.every((r) => r.kind !== 'frame' || ok(r.name));
}
const EXT = {};   // cross-scene frames fetched so far, by id (frameState on screen uses them too)
const framesRel = (sq, ext = EXT) => Object.assign(Object.fromEntries(Object.entries(ext).map(([k, v]) => [k, v.rel])),
  Object.fromEntries((sq.frames || []).filter((f) => f.rel).map((f) => [f.id, f.rel])));
// the frame ids a sequence names but does not hold (plates, pictures and first frames of other scenes)
function foreign(sq, cuts) {
  const own = new Set((sq.frames || []).map((f) => f.id)), out = new Set();
  for (const f of sq.frames || []) {
    if (f.spec.type === 'box' && !own.has(f.spec.plate)) out.add(f.spec.plate);
    for (const r of (f.spec.images || []).concat(f.spec.file ? [f.spec.file] : [])) if (r.kind === 'frame' && !own.has(r.name)) out.add(r.name);
  }
  for (const c of cuts) for (const k of ['frame', 'last']) if (c.script && c.script[k] && !own.has(c.script[k])) out.add(c.script[k]);
  for (const c of cuts) for (const x of (c.script && c.script.keys) || []) if (!own.has(x.frame)) out.add(x.frame);
  return [...out];
}
const RETRY = 60000;   // a frame / cut stopped by a missing reference is tried again after this

export async function submitFrame(sq, f) {
  const fr = Object.fromEntries((sq.frames || []).map((x) => [x.id, x]));
  const label = `S${String(sq.script.scene).padStart(2, '0')} · ${f.id}`;
  if (f.spec.type === 'file') {
    const r = await api('/api/script/file', { spec: f.spec, frames: framesRel(sq) });
    Object.assign(f, { rel: r.rel, key: specKey(f), base: null });
    return true;
  }
  const body = { spec: Object.assign({ id: f.id }, f.spec), frames: framesRel(sq), seed: 1 + (f.jobs || []).length, count: 1, label };
  if (f.spec.type === 'box') {
    const p = fr[f.spec.plate] || EXT[f.spec.plate];
    body.plate = { spec: p.spec, rel: p.rel, people: p.people || [] };
  }
  const r = await api('/api/script/frame', body);
  f.jobs = (f.jobs || []).concat([{ job: r.id, key: specKey(f), box: r.box || null }]);
  return true;
}
// ● on a frame result: the picture goes into the project library (it outlives the job), the plate keeps its faces
export async function pickFrame(sq, f, job, name) {
  const r = await api('/api/save', { job, name, title: `S${String(sq.script.scene).padStart(2, '0')}_${f.id}` });
  f.base = { job, name };
  f.rel = `projects/${project()}/scenes/${r.saved}`;
  f.key = (f.jobs.find((x) => x.job === job) || {}).key || specKey(f);
  if (f.spec.type === 'plate') {
    const m = await api('/api/meta?job=' + job);
    f.people = ((m || {})[name] || {}).people || [];
  }
}
async function autoPick(sq, f) {
  const js = (f.jobs || []).filter((x) => x.key === specKey(f)).map((x) => jobOf(x.job)).filter((j) => j && j.state === 'done');
  const cands = [];
  for (const j of js) {
    const m = await api('/api/meta?job=' + j.id);
    for (const n of pngs(j)) cands.push({ job: j.id, name: n, m: (m || {})[n] || {} });
  }
  if (!cands.length) return false;
  const score = (c) => (c.m.passed === true ? 2 : 0) + (Array.isArray(c.m.face_sim) ? Math.max(...c.m.face_sim.filter((x) => x != null), 0) : 0);
  const best = cands.reduce((a, c) => score(c) > score(a) ? c : a, cands[0]);
  await pickFrame(sq, f, best.job, best.name);
  return true;
}

// A sequence open on a screen is driven by that screen alone: it holds the lock 'shutter-seq-<id>' while open. The same
// sequence open in another tab only shows it, and the background step of every tab skips it. A background round holds
// 'shutter-run-<id>' (read → submit → save); a screen opening the sequence waits for it before reading the doc
// (withRun), so it never acts on a copy from before that round (the same frame was queued twice — replay bench,
// session #18).
export function withRun(id, fn) {
  if (!id || !navigator.locks) return fn();
  return navigator.locks.request('shutter-run-' + id, () => fn());
}
const held = {};   // sequence id → release
export function holdSeq(id) {   // → true when this tab drives the sequence
  if (!id || !navigator.locks) return Promise.resolve(true);
  if (held[id]) return Promise.resolve(true);
  return new Promise((ok) => {
    navigator.locks.request('shutter-seq-' + id, { ifAvailable: true }, (lock) => {
      if (!lock) return ok(false);
      ok(true);
      return new Promise((rel) => { held[id] = rel; });
    }).catch(() => ok(true));
  });
}
export function releaseSeq(id) {
  if (held[id]) { held[id](); delete held[id]; }
}
async function onScreens() {   // ids of the sequences a screen holds, in any tab
  if (!navigator.locks || !navigator.locks.query) return new Set();
  const q = await navigator.locks.query();
  return new Set((q.held || []).map((l) => l.name).filter((n) => n.startsWith('shutter-seq-')).map((n) => n.slice(12)));
}

// one round over a sequence → {sq: changed?, cuts: Set of the changed cuts}
export async function advance(sq, cuts) {
  const out = { sq: false, cuts: new Set() };
  if (!sq || !sq.script || !(sq.prod && sq.prod.started)) return out;
  const auto = (sq.common || {}).mode !== 'step';
  const fx = foreign(sq, cuts);
  if (fx.length) Object.assign(EXT, await api('/api/script/frames?ids=' + encodeURIComponent(fx.join(','))));   // kept across sequences
  for (const f of sq.frames || []) {
    const s = frameState(sq, f);
    if (f.err && f.err.key === specKey(f) && Date.now() - (f.err.at || 0) < RETRY) continue;   // a missing reference: tried again later
    try {
      if (s === 'todo') { await submitFrame(sq, f); out.sq = true; }
      else if (s === 'pick' && auto && await autoPick(sq, f)) out.sq = true;
    } catch (e) {
      f.err = { key: specKey(f), msg: String(e.message || e), at: Date.now() };
      out.sq = true;
    }
  }
  const rel = framesRel(sq), byId = Object.fromEntries(cuts.map((c) => [c.id, c]));
  for (const c of cuts) {
    const sc = c.script || {};
    for (const [k, fid] of [['first', sc.frame], ['last', sc.last]]) {
      const r = fid && rel[fid];
      if (r && (!c[k] || c[k].rel !== r)) { c[k] = { rel: r, name: r.split('/').pop() }; out.cuts.add(c); }
    }
    const ks = (sc.keys || []).map((x) => ({ rel: rel[x.frame], name: (rel[x.frame] || '').split('/').pop(), at: x.at }));
    if (ks.every((x) => x.rel) && JSON.stringify(ks) !== JSON.stringify(c.keys || [])) { c.keys = ks; out.cuts.add(c); }
    if (c.src || sc.gone) continue;
    const label = `S${String(sq.script.scene).padStart(2, '0')} · ${sc.id || c.title}`;
    const ready = (!sc.frame || (c.first && c.first.rel === rel[sc.frame])) && (!sc.last || (c.last && c.last.rel === rel[sc.last])) &&
      (sc.keys || []).every((x, i) => c.keys && c.keys[i] && c.keys[i].rel === rel[x.frame]);
    try {
      if (ready && !(c.prodErr && c.prodErr.key === C.key(c) && Date.now() - (c.prodErr.at || 0) < RETRY) && !c.cand.some((x) => x.key === C.key(c)) && c.prompt.trim()) {
        await C.submitCands(c, c.cands || 1, label);
        out.cuts.add(c);
      }
      if (C.autoBase(c)) out.cuts.add(c);
      if (c.mode === 'all' && c.base && await C.nextStep(c, label)) out.cuts.add(c);
    } catch (e) {   // e.g. a sheet the H3 cut needs is not there yet: shown on the cut, tried again when it changes
      c.prodErr = { key: C.key(c), msg: String(e.message || e), at: Date.now() };
      out.cuts.add(c);
    }
  }
  return out;
}

// every job a production sequence follows (its frames' and its cuts')
export const jobsOf = (sq, cuts) => (sq.frames || []).flatMap((f) => (f.jobs || []).map((x) => x.job)).concat(cuts.flatMap(C.cutJobs));

// ---- the background step: started productions other than the one open on screen, one per poll ----
const D = { list: [], at: 0, i: 0, busy: false };
export async function tick() {
  if (D.busy || !project() || !(S.status && S.status.storage_ok)) return;
  D.busy = true;
  try {
    const run = async () => {
      if (Date.now() - D.at > 30000) {   // the started, unfinished productions (every 30 s)
        const all = await api('/api/docs?kind=sequences');
        D.list = (all || []).filter((q) => q && q.script && q.prod && q.prod.started && !q.prod.done).map((q) => q.id);
        D.at = Date.now();
      }
      const shown = await onScreens();
      const ids = D.list.filter((id) => id !== S.openSeq && !shown.has(id));
      if (!ids.length) return;
      const id = ids[D.i++ % ids.length];
      if (navigator.locks) await navigator.locks.request('shutter-run-' + id, { ifAvailable: true }, async (l) => { if (l) await one(id); });
      else await one(id);
    };
    const one = async (id) => {
      if (S.openSeq === id || (await onScreens()).has(id)) return;   // opened on a screen meanwhile
      const sq = await api('/api/doc?kind=sequences&id=' + encodeURIComponent(id));
      if (!sq || !sq.id || S.openSeq === id) return;
      const cuts = (await api('/api/docs?kind=cuts&ids=' + encodeURIComponent((sq.cuts || []).join(',')))).filter(Boolean).map((c) => C.freshCut(c));
      await fetchJobs(jobsOf(sq, cuts));
      const r = await advance(sq, cuts);
      if (S.openSeq === id) return;   // opened meanwhile: the screen's own copy wins
      for (const c of r.cuts) await api('/api/doc', { kind: 'cuts', id: c.id, doc: c });
      const done = (sq.frames || []).every((f) => frameState(sq, f) === 'done') && cuts.every((c) => c.src || (c.script && c.script.gone) || C.resultOf(c));
      if (done) { sq.prod.done = true; r.sq = true; }
      if (r.sq) await api('/api/doc', { kind: 'sequences', id: sq.id, doc: sq });
    };
    if (navigator.locks) await navigator.locks.request('shutter-prod', { ifAvailable: true }, async (lock) => { if (lock) await run(); });
    else await run();
  } catch (e) {
    console.warn('prod tick', e);
  } finally {
    D.busy = false;
  }
}
export const prodWord = (s) => t('pr.st.' + s);
