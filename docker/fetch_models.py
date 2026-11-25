# Download the weights shutter needs into the shared models folder (ComfyUI folder layout). Everything is pinned:
# engine/py/models.py SOURCES (group, file, Hugging Face repo, path) and weights.lock.json next to this script (repo commit,
# size and sha256 of the files shutter was tested with). A download whose sha256 differs is refused.
# Files already in the folder are kept: the pinned size counts as present (--verify also checks their sha256); another
# size is reported and left alone — a ComfyUI folder may hold another version under the same name; move it aside to
# fetch ours.
#   fetch_models.py --list                 groups, sizes, what is already there (nothing downloaded)
#   fetch_models.py [--verify] [group ...] fetch these groups (none: all), base always
# Run inside the shutter image:
#   docker run --rm --label project=shutter --user 1000:1000 -e HOME=/tmp [-e HF_TOKEN=...] \
#     -v ~/studio/models:/models shutter:<tag> python /opt/shutter/fetch_models.py image
import hashlib
import json
import os
import shutil
import sys

import models

GROUPS = {   # group → what needs it
    "base": "face check, face pass, pose check (always fetched)",
    "image": "Qwen-Image 2.1: images, characters, places, scenes; also video.swap",
    "h3": "MiniMax H3: video.r2v, video.swap, video.pih3 (pi-H3)",
    "ltx": "LTX-2.5: video.i2v (gated: accept its license on Hugging Face, pass HF_TOKEN)",
    "upscale": "SeedVR2 7B: video.upscale, video.swap with upscale",
    "voice": "voices: audio.tts, audio.voice",
}
GATED = {"Lightricks/LTX-2.5"}
LOCK = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.lock.json")))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gb(n):
    return f"{n / 1e9:.1f} GB"


def present(rel):
    p = models.path(rel)
    return os.path.exists(p) and os.path.getsize(p) == LOCK[rel]["size"]


args = sys.argv[1:]
verify, listing = "--verify" in args, "--list" in args
want = [a for a in args if not a.startswith("--")]
bad = [a for a in want if a not in GROUPS] + [a for a in args if a.startswith("--") and a not in ("--verify", "--list")]
if bad:
    sys.exit(f"unknown: {' '.join(bad)} — groups: {' '.join(GROUPS)}; options: --list, --verify")
want = set(want or GROUPS) | {"base"}
items = [(g, rel, repo, src, LOCK[rel]) for g, rel, repo, src in models.SOURCES if g in want]

if listing:
    print(f"models folder {models.ROOT}, free {gb(shutil.disk_usage(models.ROOT).free)}\n")
    for g, use in GROUPS.items():
        fs = [rel for gg, rel, _, _ in models.SOURCES if gg == g]
        have = sum(present(r) for r in fs)
        print(f"{g:8} {gb(sum(LOCK[r]['size'] for r in fs)):>9}  {have}/{len(fs)} present  {use}")
    sys.exit(0)

todo = [it for it in items if not os.path.exists(models.path(it[1]))]
need = sum(it[4]["size"] for it in todo)
free = shutil.disk_usage(models.ROOT).free
print(f"groups: {' '.join(g for g in GROUPS if g in want)} — {len(items)} files, {len(todo)} to fetch ({gb(need)}), "
      f"free {gb(free)}", flush=True)
if need > free:
    sys.exit(f"not enough space: {gb(need)} to fetch, {gb(free)} free")

from huggingface_hub import hf_hub_download   # noqa: E402 (only when something is fetched)

ok, have, fail = [], [], []
tmp = os.path.join(models.ROOT, ".dl")
for g, rel, repo, src, pin in items:
    out = models.path(rel)
    if os.path.exists(out):
        size = os.path.getsize(out)
        if size != pin["size"]:
            fail.append(f"{rel}: already there with another size ({size}, tested {pin['size']}) — move it aside to fetch ours")
        elif verify and sha256(out) != pin["sha256"]:
            fail.append(f"{rel}: already there but its sha256 is not the tested file's — move it aside to fetch ours")
        else:
            have.append(rel)
        continue
    try:
        print(f"[fetch] {g}: {repo}@{pin['revision'][:8]} :: {src} ({gb(pin['size'])})", flush=True)
        got = hf_hub_download(repo, src, revision=pin["revision"], local_dir=tmp)
        if sha256(got) != pin["sha256"]:
            os.remove(got)
            raise ValueError("sha256 differs from the tested file")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.move(got, out)
        ok.append(rel)
    except Exception as e:  # keep going; report at the end
        hint = f" — gated: accept the license on huggingface.co/{repo}, then pass -e HF_TOKEN=..." if repo in GATED else ""
        fail.append(f"{repo} :: {src} — {str(e).splitlines()[0][:200]}{hint}")
if not fail:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"\nfetched {len(ok)} / present {len(have)}{' (sha256 checked)' if verify else ''} / failed {len(fail)}")
for x in fail:
    print("  failed:", x)
sys.exit(1 if fail else 0)
