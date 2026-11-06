"""Shared prompt generation for the parallel-tail tokenization bench/check (branch 233).

Benchmark-shaped prompts: BOS text + one ~32768-token shared prefix + a ~2048-token suffix
whose first word differs, as the frozen harness sends them. Seam-stress text is dense with
digits, CJK/kana, whitespace runs, tabs, newlines, special tokens, combining marks and emoji.
"""

from __future__ import annotations

import os
import time

T0 = time.time()


def log(msg):
    print(f"[+{time.time() - T0:6.1f}s] {msg}", flush=True)


def host_receipt():
    import socket

    try:
        aff = sorted(os.sched_getaffinity(0))
    except AttributeError:
        aff = []
    rng = f"{aff[0]}-{aff[-1]} ({len(aff)} cpus)" if aff else "?"
    node = "?"
    try:
        for d in sorted(os.listdir("/sys/devices/system/node")):
            if d.startswith("node"):
                cpus = open(f"/sys/devices/system/node/{d}/cpulist").read().strip()
                first = int(cpus.split(",")[0].split("-")[0])
                if aff and first <= aff[0] <= int(cpus.split(",")[0].split("-")[-1]):
                    node = d
    except OSError:
        pass
    return {
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "affinity": rng,
        "numa_node_of_first_cpu": node,
        "load_avg": os.getloadavg(),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }


class PromptGen:
    def __init__(self, hf, rng):
        self.hf = hf
        self.rng = rng
        self.bos = hf.bos_token or "<｜begin▁of▁sentence｜>"
        self.specials = sorted({t for t in hf.all_special_tokens if t} | {self.bos})
        vocab = hf.get_vocab()
        self.words = sorted({hf.convert_tokens_to_string([t]) for t in vocab
                             if t.startswith("Ġ") and t[1:].isalpha() and t[1:].isascii()
                             and 3 <= len(t) - 1 <= 10})
        self.caps = sorted({w for w in self.words if w[1].isupper()})
        sp = self.specials
        self.tricky_bits = [
            lambda r: f" {r.randint(0, 10**r.randint(1, 9))}",
            lambda r: f"{r.randint(0, 999)}{r.randint(0, 999)} ",
            lambda r: " " * r.randint(2, 6),
            lambda r: "\t" * r.randint(1, 3) + " ",
            lambda r: "\n" * r.randint(1, 3) + " " * r.randint(0, 3),
            lambda r: " 中文字符",
            lambda r: "日本語のテキスト ",
            lambda r: " ひらがな" + r.choice(["", " ", "。"]),
            lambda r: r.choice(sp) + r.choice(["", " ", "\n"]),
            lambda r: " été café ",
            lambda r: " \U0001f600\U0001f680 " + r.choice(["", "‍\U0001f525"]),
            lambda r: " ½ Ⅳ ٣٤ ",
            lambda r: " (x) [y] {z}; \"q\" 'r' ",
            lambda r: r.choice([" ", "　", " ", "​"]) + " ",
            lambda r: " " + r.choice(self.caps).strip() + r.choice([":", "?", "!", "..."]),
        ]

    def prose(self, n_words):
        rng, out = self.rng, []
        for _ in range(n_words):
            out.append(rng.choice(self.words))
            r = rng.random()
            if r < 0.06:
                out.append(".")
            elif r < 0.09:
                out.append(",")
            elif r < 0.10:
                out.append("\n\n")
            elif r < 0.11:
                out.append(f" {rng.randint(0, 99999)}")
        return "".join(out)

    def tricky(self, n_items):
        rng, out = self.rng, []
        for _ in range(n_items):
            if rng.random() < 0.5:
                out.append(rng.choice(self.tricky_bits)(rng))
            else:
                out.append(self.prose(rng.randint(1, 8)))
        return "".join(out)

    def fit_tokens(self, text, n):
        offs = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offs) <= n:
            return text
        return text[: offs[n][0]]

    def prefix(self):
        return self.fit_tokens(self.bos + self.prose(40000), 32768)

    def suffix(self, lead):
        return self.fit_tokens(lead + self.prose(2600), 2048)

    def tricky_prefix(self):
        return self.fit_tokens(self.bos + self.tricky(9000), 32768)

    def variant(self, tprefix, i):
        rng = self.rng
        mode = i % 4
        if mode in (0, 1):
            d = rng.randint(len(tprefix) // 2 + 600, len(tprefix) - 1)
            return tprefix[:d] + rng.choice(self.tricky_bits)(rng) + self.tricky(rng.randint(5, 400))
        if mode == 2:
            return tprefix + rng.choice(self.tricky_bits)(rng) + self.tricky(rng.randint(50, 600))
        d = rng.randint(len(tprefix) // 2 + 600, len(tprefix) - 1)
        return tprefix[:d] + rng.choice(["7", "中", " ", "\t", "́", "\n"]) \
            + self.tricky(rng.randint(5, 300))
