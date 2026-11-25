# Qwen-Image 2.1 text-encoder tokenizer (Qwen2 BPE), same ids as ComfyUI's QwenImage21Tokenizer.
# Needs only `tokenizers` (+ stdlib). Files: vocab.json, merges.txt, tokenizer_config.json (ComfyUI qwen25_tokenizer).
#   tok = QI21Tokenizer(dir); ids = tok(prompt, n_images)   -> list[int], -1 = one image slot
# The engine (qi21_te_encode) expands each -1 to its vision tokens, drops the system turn and the vision rows.
import json
import os

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = 151655
# Qwen2Tokenizer pre-tokenizer
PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class QI21Tokenizer:
    def __init__(self, path):
        t = Tokenizer(models.BPE.from_file(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt")))
        t.normalizer = normalizers.NFC()
        t.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PAT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        t.decoder = decoders.ByteLevel()
        added = json.load(open(os.path.join(path, "tokenizer_config.json")))["added_tokens_decoder"]
        for i in sorted(added, key=int):
            a = added[i]
            t.add_special_tokens([AddedToken(a["content"], special=a["special"], normalized=False,
                                             lstrip=a["lstrip"], rstrip=a["rstrip"], single_word=a["single_word"])])
            if t.token_to_id(a["content"]) != int(i):
                raise ValueError(f"added token {a['content']} got id {t.token_to_id(a['content'])}, expected {i}")
        self.t = t

    def __call__(self, prompt, n_images=0):
        text = prompt or " "  # prevent_empty_text
        # sd1_clip escape/unescape_important with weights disabled: "\(" -> "(", "\)" -> ")"
        text = text.replace("\\)", ")").replace("\\(", "(")
        if text.startswith("<|im_start|>"):
            full = text  # caller supplied the whole template
        else:
            template = T2I_TEMPLATE
            if n_images:
                refs = " ".join("<image{}>{}".format(i + 1, VISION_BLOCK) for i in range(n_images))
                template = template.replace("{}", refs + "{}", 1)
            full = template.format(text)
        ids = self.t.encode(full, add_special_tokens=False).ids
        out, k = [], 0
        for i in ids:
            if i == IMAGE_PAD and k < n_images:
                out.append(-1)
                k += 1
            else:
                out.append(i)
        return out
