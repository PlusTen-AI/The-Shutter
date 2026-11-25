// One cut of 영상 생성 (design §11) — what the 한 컷 and 시퀀스 screens share, no rendering: settings, candidates (one
// job each, seeds from the cut seed up, C·D share them), ● base, refine (H3), upscale and the cut's result.
import { S, t, html, api, jobFileUrl, isStopped } from '../app.js';
import { defaults, MODELS } from '../models.js';

export const MODEL_IDS = ['ltx25', 'pih3', 'h3'];
export const OUT = { 'video.i2v': 'i2v.mp4', 'video.pih3': 'pih3.mp4', 'video.r2v': 'shot_00_pre.mp4' };
export const SIZE = { v: [768, 1344], h: [1344, 768] };
// measured seconds on one node (docs/progress.md): per second of video, model load; face pass per candidate; refine
// (H3 pass 2) and upscale (SeedVR2) per second of video
const PER_SEC = { ltx25: 12, pih3: 15, pih3_4: 28, h3: 44 }, LOAD = { ltx25: 55, pih3: 40, h3: 60 };
const FACE = 25, REFINE = 46, UP = 53, VOICE = 30;   // VOICE: per cut (separation + conversion, estimate)
// ---- editing (design §12 A) ----
export const useOf = (c) => c.use ? [Number(c.use[0]), Number(c.use[1])] : [0, Number(c.sec) || 0];
export const lenOf = (c) => Math.max(0, useOf(c)[1] - useOf(c)[0]);
// dialogue in the cut's own generation (from the script's say lines); cuts made by hand count as dialogue
export const speechOf = (c) => !Array.isArray(c.say) || c.say.length > 0;

// what a sequence sets for all its cuts; a cut that changes one of these itself keeps it (cut.over)
export const COMMON = ['codes', 'orient', 'model', 'sec', 'cands', 'face', 'refineOn', 'voiceOn', 'upOn', 'mode', 'steps', 'vsa', 'refSize'];

export function freshCut(from) {
  const c = { id: null, seq: null, title: '', prompt: '', first: null, last: null, codes: [], orient: 'v', mode: 'all',
    seed: 1, steps: 2, vsa: false, refSize: 'max', face: true, refineOn: true, voiceOn: true, upOn: false,
    cand: [], saved: [], base: null, refineJob: null, upJob: null, dropped: [], over: [],
    // editing (design §12 A): use = [a, b] of the generation (null = all of it), sound = null (its own) | 'none' |
    // {cut, at} (another cut's generation from `at` s, 기능 2), src = another cut's id whose generation this cut shows
    // (기능 9, no generation of its own), brk = an episode break after this cut (기능 6), say / texts from the script,
    // fade = {in, out} seconds from / to black, picture and sound (null = none); keys = a multi-shot's later shot first
    // frames [{rel, name, at}] (H3, 기능 8); shot = with src a multi-shot generation, the shot number its use is inside
    use: null, sound: null, src: null, shot: null, brk: false, texts: [], fade: null, keys: [], voiceJob: null };
  applyModel(c, 'ltx25');
  return Object.assign(c, from || {});
}

// a model's defaults (설정 › 기본값, else the engine's) for the fields it has
// 길이: any length in seconds (boss, session #14 — the script's 6 · 7 s cuts), the model's usual lengths as quick picks;
// the server rounds to the model's frame rule (LTX 8k+1, H3 · pi-H3 17k+5)
const secOpts = (m) => MODELS[m].fields.find((f) => f.k === 'sec').opts;
export function secClamp(m, v) {
  const x = Math.round(Number(v) * 10) / 10;
  return x >= 1 ? Math.min(x, Math.max(...secOpts(m))) : null;
}
export const secInput = (m, v) => html`<span class="row" style="gap: 6px; flex-wrap: nowrap"><input class="in mono" type="number" data-k="sec" min="1" max="${Math.max(...secOpts(m))}" step="0.1" list="secs-${m}" value="${v}" style="width: 84px" aria-label="${t('vg.len')}">
  <span class="sub">${t('vg.secUnit')}</span><datalist id="secs-${m}">${secOpts(m).map((x) => html`<option value="${x}"></option>`)}</datalist></span>`;

export function applyModel(c, m) {
  const d = defaults(m);
  c.model = m;
  c.sec = d.sec;
  c.cands = d.cands;
  c.upOn = !!d.upscale;
  if (d.face !== undefined) c.face = d.face;
  if (d.steps !== undefined) c.steps = d.steps;
  if (d.vsa !== undefined) c.vsa = !!d.vsa;
  if (d.ref_size !== undefined) c.refSize = d.ref_size;
  if (d.refine !== undefined) c.refineOn = d.refine;
}

// why a model can't take this cut's inputs (null = fine) — design §11 모델 선택
export function whyNot(c, m, chars) {
  if ((c.keys || []).length && m !== 'h3') return 'keysH3';
  if (m === 'ltx25') return !c.first ? 'needFirst' : c.last ? 'noLast' : null;
  if (m === 'h3') return !c.codes.length || c.codes.some((k) => !hasSheet(chars, k)) ? 'needSheet' : null;
  return null;
}
export const hasSheet = (chars, code) => { const x = chars.find((y) => y.code === code); return !!x && x.images.some((i) => i.kind === 'sheet'); };

// everything that shapes the candidates: a change makes the made ones "옛 설정"
export function key(c) {
  return JSON.stringify([c.model, c.prompt.trim(), c.first && c.first.rel, c.last && c.last.rel, c.codes, c.orient, Number(c.sec),
    ...((c.keys || []).length ? [c.keys.map((x) => [x.rel, x.at])] : []),
    c.model === 'pih3' ? Number(c.steps) : 0, c.model === 'pih3' && c.vsa ? 1 : 0, c.model === 'h3' ? c.refSize : 0, c.model !== 'h3' && c.face && c.codes.length ? 1 : 0]);
}

export const jobOf = (id) => (id && S.jobs.find((j) => j.id === id)) || null;
const synced = (j, name) => j && j.files.find((f) => f.name === name && f.synced);

// one candidate → its job, state (pending · running · arriving · done · failed · gone), file, progress, face score
const FACES = {};
export function cand(c, x) {
  const j = jobOf(x.job);
  const name = j ? OUT[j.type] : null;
  const f = synced(j, name);
  const state = !j ? 'gone' : f ? 'done' : j.state === 'failed' ? 'failed' : j.state === 'done' ? 'arriving' : j.state;
  return { ...x, j, name, file: f, state, stopped: state === 'failed' && isStopped(j), pct: j ? Math.round(j.progress * 100) : 0, old: x.key !== key(c),
    url: f ? jobFileUrl(j, name) + '?v=' + f.sha.slice(0, 8) : null, face: FACES[x.job] };
}
// face pass scores (face.json, after the pass) of finished candidates — read once
export async function loadFaces(c) {
  for (const x of c.cand) {
    const j = jobOf(x.job);
    if (FACES[x.job] !== undefined || !synced(j, 'face.json')) continue;
    try {
      const m = await (await fetch(jobFileUrl(j, 'face.json'))).json();
      FACES[x.job] = m && m.after && m.after.median != null ? Number(m.after.median) : null;
    } catch (e) { /* next poll */ }
  }
}

// the cut's video before upscale: the refined base (H3 with 다듬기) or the base candidate's own
export function finalOf(c) {
  const bj = jobOf(c.base);
  if (!bj) return null;
  if (bj.type === 'video.r2v' && c.refineOn) {
    const rj = c.refineJob && c.refineJob.from === c.base ? jobOf(c.refineJob.job) : null;
    return synced(rj, 'shot_00.mp4') ? { job: rj.id, name: 'shot_00.mp4', j: rj } : null;
  }
  return synced(bj, OUT[bj.type]) ? { job: bj.id, name: OUT[bj.type], j: bj } : null;
}
// 목소리 (기능 17): the characters speaking in the cut's own generation (script say lines); the voice step runs when one
// of them has a registered voice — the Mac answers {skip} when none has, and the cut goes on without it
export const speakersOf = (c) => (Array.isArray(c.say) ? c.say : []).map((x) => x.code).filter((x) => /^[FM][LSE]\d/.test(x || ''));
const voiceDue = (c) => c.voiceOn !== false && speakersOf(c).length > 0;
const fkey = (f) => f && f.job + '/' + f.name;
// the cut's video with its dialogue in the speakers' own voices (the final one when the step is off or skipped)
export function voicedOf(c) {
  const f = finalOf(c);
  if (!f || !voiceDue(c)) return f;
  const v = c.voiceJob && c.voiceJob.from === fkey(f) ? c.voiceJob : null;
  if (!v) return null;
  if (v.skip) return f;
  const j = jobOf(v.job);
  return synced(j, 'voice.mp4') ? { job: j.id, name: 'voice.mp4', j } : null;
}
// the cut's result: the upscaled one when 업스케일 is on (null until it is there)
export function resultOf(c) {
  const f = voicedOf(c);
  if (!f || !c.upOn) return f;
  const uj = c.upJob && c.upJob.from === f.job + '/' + f.name ? jobOf(c.upJob.job) : null;
  return synced(uj, 'up.mp4') ? { job: uj.id, name: 'up.mp4', j: uj } : null;
}
export const urlOf = (r) => r ? jobFileUrl(r.j, r.name) + '?v=' + synced(r.j, r.name).sha.slice(0, 8) : '';

// The scene render (video.edit, design §12 A) from a sequence's cuts in order: each generation used once as a source,
// per cut the picture range, the sound source and range, the part (a new one after each break) and the screen text.
// A cut without a result yet is left out (missing). multi: the source is a multi-shot generation (기능 8 — the render
// moves cut edges onto its shot changes). → {sources [{job, name}], speech [bool], multi [bool], cuts [...], parts, endBrk, missing}
export function timeline(cuts) {
  const byId = Object.fromEntries(cuts.filter((c) => c.id).map((c) => [c.id, c]));
  const gen = (c) => (c.src && byId[c.src]) || c;   // the cut whose generation the picture comes from (기능 9)
  const res = (c) => { const g = gen(c); return resultOf(g); };
  const sources = [], speech = [], multi = [], idx = {};
  const srcOf = (r, sp, g) => {
    const k = r.job + '/' + r.name;
    if (!(k in idx)) { idx[k] = sources.length; sources.push({ job: r.job, name: r.name }); speech.push(!!sp); multi.push(!!(g && g.genOnly)); }
    else if (sp) speech[idx[k]] = true;
    return idx[k];
  };
  const out = [], missing = [];
  const last = cuts.map((c) => !c.genOnly).lastIndexOf(true);   // generation-only cuts may sit after the last placed one
  let part = 0;
  cuts.forEach((c, k) => {
    if (c.genOnly) return;   // a generation several cuts take spans of (기능 8 · 9): not placed itself
    const r = res(c);
    if (!r) missing.push(k);
    else {
      const v = srcOf(r, speechOf(gen(c)) && c.sound == null, gen(c)), a = useOf(c)[0];
      let s = v, sa = a;
      if (c.sound === 'none') s = null;
      else if (c.sound && c.sound.cut) {
        const j = byId[c.sound.cut], rj = j && res(j);
        if (rj) { s = srcOf(rj, speechOf(gen(j)), gen(j)); sa = Number(c.sound.at) || 0; } else s = null;
      }
      out.push({ v, a, n: Math.max(1, Math.round(lenOf(c) * 24)), s, sa, part, text: c.texts || [], ...(c.fade ? { fade: c.fade } : {}),
        ...(c.src && c.shot && gen(c).genOnly ? { shot: Number(c.shot) } : {}) });
    }
    if (c.brk && k < last && out.some((x) => x.part === part)) part += 1;
  });
  const placed = cuts.filter((c) => !c.genOnly);
  return { sources, speech, multi, cuts: out, parts: out.length ? part + 1 : 0, endBrk: !!(placed.length && placed[placed.length - 1].brk), missing };
}

// stage states for the progress pills and the sequence rows: wait · run · pick · done · failed · old
export function stages(c) {
  const cs = c.cand.filter((x) => x.key === key(c)).map((x) => cand(c, x));
  const live = (s) => s === 'pending' || s === 'running' || s === 'arriving';
  const gen = !cs.length ? { state: c.cand.length ? 'old' : 'wait' } :
    cs.some((x) => live(x.state)) ? { state: 'run', pct: Math.round(cs.reduce((a, x) => a + (x.state === 'done' ? 100 : x.pct), 0) / cs.length) } :
    !cs.some((x) => x.state === 'done') ? { state: 'failed', stopped: cs.filter((x) => x.state === 'failed').every((x) => x.stopped) } : c.base ? { state: 'done' } : { state: 'pick' };
  const bj = jobOf(c.base);
  const jobStage = (rec, from, on) => {
    if (!on) return { state: 'off' };
    if (!rec || rec.from !== from) return { state: c.base ? (rec ? 'old' : 'wait') : 'wait' };
    const j = jobOf(rec.job);
    return !j ? { state: 'wait' } : j.state === 'failed' ? { state: 'failed', stopped: isStopped(j) } : j.state === 'done' && j.files.every((f) => f.synced || f.tmp) ?
      { state: 'done' } : { state: 'run', pct: Math.round(j.progress * 100) };
  };
  const ref = jobStage(c.refineJob, c.base, c.refineOn && (bj ? bj.type === 'video.r2v' : c.model === 'h3'));
  const f = finalOf(c), vr = c.voiceJob && c.voiceJob.from === fkey(f) && c.voiceJob;
  const voice = vr && vr.skip ? { state: 'off', skip: vr.skip } : jobStage(c.voiceJob, fkey(f), voiceDue(c));
  const v = voicedOf(c);
  const up = jobStage(c.upJob, fkey(v), c.upOn);
  return { gen, ref, voice, up, cands: cs };
}
// one word for a whole cut (sequence rows): wait · run · pick · done · failed · stopped · old
export function cutState(c) {
  const s = stages(c);
  if (s.gen.state === 'run' || s.ref.state === 'run' || s.voice.state === 'run' || s.up.state === 'run') return 'run';
  const fs = [s.gen, s.ref, s.voice, s.up].filter((x) => x.state === 'failed');
  if (fs.length) return fs.every((x) => x.stopped) ? 'stopped' : 'failed';
  if (s.gen.state === 'old' || (c.base && c.cand.some((x) => x.job === c.base && x.key !== key(c)))) return 'old';
  if (s.gen.state === 'pick') return 'pick';
  return resultOf(c) ? 'done' : s.gen.state === 'wait' ? 'wait' : 'run';
}

// candidates with the current settings: seeds from the cut seed up, skipping ones this setting already used
export async function submitCands(c, n, label) {
  const k = key(c);
  const used = c.cand.filter((x) => x.key === k).map((x) => x.seed);
  const made = [];
  for (let i = 0; i < n; i++) {
    let seed = Number(c.seed) || 0;
    while (used.includes(seed)) seed++;
    used.push(seed);
    const r = await api('/api/video', { model: c.model, prompt: c.prompt, first: c.first && c.first.rel, last: c.last && c.last.rel,
      keys: (c.keys || []).map((x) => ({ rel: x.rel, at: x.at })),
      codes: c.codes, orient: c.orient, sec: c.sec, seed, steps: c.steps, vsa: !!c.vsa, ref_size: c.refSize, face: c.face, label: label + ' · seed ' + seed });
    made.push({ job: r.id, key: k, seed });
    c.cand.push(made[made.length - 1]);
  }
  return made;
}
// "이 후보만 다시": a new seed in the same place; the old job is closed with the cut
export async function redoCand(c, i, label) {
  const old = c.cand[i];
  const [x] = await submitCands(c, 1, label);
  c.cand.splice(c.cand.indexOf(x), 1);
  c.cand.splice(i, 1, x);
  c.dropped.push(old.job);
  if (c.base === old.job) c.base = null;
  c.saved = c.saved.filter((j) => j !== old.job);
}

// 한 번에 쭉: once this setting's candidates are in, the best face (else the first) becomes ●, then the next steps run
export function autoBase(c) {
  if (c.mode !== 'all' || c.base) return false;
  const s = stages(c);
  if (s.gen.state !== 'pick') return false;
  const done = s.cands.filter((x) => x.state === 'done');
  c.base = done.reduce((a, x) => (x.face ?? -1) > (a.face ?? -1) ? x : a, done[0]).job;
  return true;
}
// refine (H3 with 다듬기) and upscale for the ● base when they are due → true when something was submitted
export async function nextStep(c, label) {
  const bj = jobOf(c.base);
  if (bj && bj.type === 'video.r2v' && c.refineOn && (!c.refineJob || c.refineJob.from !== c.base)) {
    const r = await api('/api/refine', { picks: [{ job: c.base, shot: 0 }], label });
    c.refineJob = { job: r.id, from: c.base };
    return true;
  }
  const f = finalOf(c);
  // 목소리 before 업스케일 (it keeps the sound); a skip (no speaker had a voice) is asked again after 5 min
  if (f && voiceDue(c) && (!c.voiceJob || c.voiceJob.from !== fkey(f) || (c.voiceJob.skip && Date.now() - (c.voiceJob.at || 0) > 300000))) {
    const r = await api('/api/voice', { job: f.job, name: f.name, say: c.say, label });
    c.voiceJob = r.skip ? { skip: r.skip, from: fkey(f), at: Date.now() } : { job: r.id, from: fkey(f) };
    return true;
  }
  const v = voicedOf(c);
  if (v && c.upOn && (!c.upJob || c.upJob.from !== fkey(v))) {
    const r = await api('/api/upscale', { job: v.job, name: v.name, options: { asset_code: c.codes.join(' '), asset_kind: label } });
    c.upJob = { job: r.id, from: fkey(v) };
    return true;
  }
  return false;
}

// time for n candidates with these settings (C·D share them) + the steps after them, seconds (estimate)
export function estimate(c, n) {
  const sec = Number(c.sec) || 5;
  const per = (c.model === 'pih3' && Number(c.steps) === 4 ? PER_SEC.pih3_4 : PER_SEC[c.model]) * sec +
    (c.model !== 'h3' && c.face && c.codes.length ? FACE : 0);
  let t = LOAD[c.model] + Math.ceil(n / 2) * per;
  if (c.model === 'h3' && c.refineOn) t += REFINE * sec;
  if (voiceDue(c)) t += VOICE;
  if (c.upOn) t += UP * sec / (sec >= 2 ? 2 : 1);   // from 2 s C·D share the upscale chunks
  return Math.round(t);
}

// every job the cut made (closed with it), and what goes to the project library: ☆ candidates always (닫기 too — only
// unsaved candidates go, design §11 후보), the result only on 보관함에 저장
export const cutJobs = (c) => [...c.cand.map((x) => x.job), ...c.dropped, c.refineJob && c.refineJob.job, c.voiceJob && c.voiceJob.job, c.upJob && c.upJob.job].filter(Boolean);
export function saveItems(c, title, withResult) {
  const out = c.saved.map((id) => jobOf(id)).filter((j) => j && synced(j, OUT[j.type]))
    .map((j, i) => ({ job: j.id, name: OUT[j.type], title: title + '_cand' + (i + 1) }));
  const r = withResult && (resultOf(c) || finalOf(c));
  if (r) out.push({ job: r.job, name: r.name, title });
  return out;
}
