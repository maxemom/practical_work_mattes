#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepare_dataset.py

Create small, reproducible prompt files (1 prompt per line) from:
1) WikiText-2-like .txt dumps (with "= Title =" and "== Section ==" markers)
   -> sample N articles (seeded) and keep ONLY the first sentence per sample
2) CSV datasets (e.g., TellMeWhy)
   -> sample N rows (seeded) and concatenate narrative + question (question after narrative)

Designed for Option A:
- raw data stays local (data/raw/...) and is NOT committed
- processed prompt files (data/processed/...) are small and CAN be committed

Usage examples:
  python scripts/prepare_dataset.py wikitext \
    --input data/raw/wikitext-2/train.txt \
    --output data/processed/wikitext2_100_seed42.txt \
    --seed 42 --n-samples 100

  python scripts/prepare_dataset.py csv \
    --input data/raw/tellmewhy.csv \
    --output data/processed/tellmewhy_100_seed42.txt \
    --seed 42 --n-samples 100 \
    --narrative-col narrative --question-col question

  python scripts/prepare_dataset.py all --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


# ---------------------------
# Cleaning + first sentence
# ---------------------------

MULTISPACE_RE = re.compile(r"\s+")
# Article title lines in WikiText-2 are typically single '=' on each side:
# "= 2013 – 14 York City F.C. season ="
ARTICLE_TITLE_RE = re.compile(r"^\s*=\s*[^=].*?[^=]\s*=\s*$")
# Section titles often use multiple '=': "== Background ==" or "= = Background = ="
SECTION_TITLE_RE = re.compile(r"^\s*=+\s*=+\s*.*?\s*=+\s*=+\s*$|^\s*==+\s*.*?\s*==+\s*$")


def clean_wikitext(text: str) -> str:
    """
    Minimal normalization tailored for WikiText-2-like artifacts.
    """
    t = text.replace("\t", " ").replace("\n", " ")
    t = t.replace("@-@", "-")
    t = t.replace("<unk>", "")
    t = MULTISPACE_RE.sub(" ", t).strip()

    # light punctuation spacing fix
    t = t.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    t = t.replace(" '", "'").replace(" n't", "n't")
    return t


def clean_generic(text: str) -> str:
    """
    Generic cleanup for CSV text fields.
    """
    t = text.replace("\t", " ").replace("\n", " ")
    t = t.replace("<unk>", "")
    t = t.replace("@-@", "-")
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def first_sentence(text: str, max_chars: int = 280) -> str:
    """
    Heuristic first-sentence extraction.
    - prefers first occurrence of [.?!]
    - falls back to text start if no punctuation
    - truncates to max_chars
    """
    t = clean_wikitext(text)
    if not t:
        return ""

    m = re.search(r"(.+?[.!?])(\s|$)", t)
    sent = m.group(1).strip() if m else t

    if len(sent) > max_chars:
        sent = sent[:max_chars].rstrip()
        # avoid cutting mid-word too badly
        sent = re.sub(r"\s+\S*$", "", sent).strip() or sent

    return sent


# ---------------------------
# WikiText reader (article-based)
# ---------------------------

def iter_wikitext_articles(path: Path, drop_section_titles: bool = True) -> Iterable[str]:
    """
    Yield article blocks from a WikiText-2-like file.

    Strategy:
    - Start a new article on ARTICLE title lines: "= Title ="
    - Everything until next ARTICLE title belongs to the current article
    - Optionally skip section title lines like "== Background ==" / "= = Background = ="

    This keeps "one block per article", which is what you want for sampling.
    """
    buf: List[str] = []
    in_article = False

    def flush() -> Optional[str]:
        nonlocal buf
        if not buf:
            return None
        block = "\n".join(buf).strip()
        buf = []
        return block if block else None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # New article boundary
            if ARTICLE_TITLE_RE.match(line):
                # flush previous article (if any)
                out = flush()
                if out is not None:
                    yield out
                in_article = True
                # we do not include the title line in content (usually not needed)
                continue

            if not in_article:
                continue  # ignore preamble before first article title

            # Skip section titles if desired (they aren't content)
            if drop_section_titles and SECTION_TITLE_RE.match(line.strip()):
                continue

            buf.append(line)

    out = flush()
    if out is not None:
        yield out


# ---------------------------
# Reservoir sampling (streaming, reproducible)
# ---------------------------

def reservoir_sample(items: Iterable[str], k: int, seed: int) -> List[str]:
    """
    Reservoir sampling: sample k items uniformly from a stream without loading all items.
    """
    rng = random.Random(seed)
    res: List[str] = []
    n = 0
    for x in items:
        n += 1
        if len(res) < k:
            res.append(x)
        else:
            j = rng.randrange(n)
            if j < k:
                res[j] = x
    return res


# ---------------------------
# Prepare: WikiText -> first-sentence prompts
# ---------------------------

def prepare_wikitext(
    input_path: Path,
    output_path: Path,
    n_samples: int,
    seed: int,
    min_chars: int = 40,
    max_sentence_chars: int = 280,
    drop_section_titles: bool = True,
) -> None:
    # Stream articles -> convert each to first sentence -> filter -> sample
    def stream_sentences() -> Iterable[str]:
        for article in iter_wikitext_articles(input_path, drop_section_titles=drop_section_titles):
            s = first_sentence(article, max_chars=max_sentence_chars)
            if len(s) >= min_chars:
                yield s

    sampled = reservoir_sample(stream_sentences(), k=n_samples, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sampled) + ("\n" if sampled else ""), encoding="utf-8")
    print(f"[wikitext] wrote {len(sampled)} prompts -> {output_path} (seed={seed})")


# ---------------------------
# Prepare: CSV -> narrative + question prompts
# ---------------------------

def build_prompt(narrative: str, question: str) -> str:
    n = clean_generic(narrative)
    q = clean_generic(question)
    if n and q:
        return f"{n} {q}"
    return n or q


def prepare_csv(
    input_path: Path,
    output_path: Path,
    n_samples: int,
    seed: int,
    narrative_col: str = "narrative",
    question_col: str = "question",
    min_chars: int = 40,
    max_chars: int = 800,
) -> None:
    rng = random.Random(seed)

    reservoir: List[str] = []
    seen = 0

    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        if narrative_col not in reader.fieldnames or question_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing columns. Found: {reader.fieldnames}. "
                f"Need: '{narrative_col}' and '{question_col}'."
            )

        for row in reader:
            prompt = build_prompt(row.get(narrative_col, "") or "", row.get(question_col, "") or "")
            if not prompt:
                continue

            # length filters
            if len(prompt) < min_chars:
                continue
            if len(prompt) > max_chars:
                prompt = prompt[:max_chars].rstrip()
                prompt = re.sub(r"\s+\S*$", "", prompt).strip() or prompt

            seen += 1
            if len(reservoir) < n_samples:
                reservoir.append(prompt)
            else:
                j = rng.randrange(seen)
                if j < n_samples:
                    reservoir[j] = prompt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(reservoir) + ("\n" if reservoir else ""), encoding="utf-8")
    print(f"[csv] wrote {len(reservoir)} prompts -> {output_path} (seed={seed})")


# ---------------------------
# Config + CLI
# ---------------------------

def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return cfg


def get_cfg_path(cfg: Dict, *keys: str) -> str:
    cur: Dict = cfg
    for k in keys:
        if k not in cur:
            raise KeyError(f"Missing config key: {'.'.join(keys)} (failed at '{k}')")
        cur = cur[k]
    if not isinstance(cur, str):
        raise ValueError(f"Config value at {'.'.join(keys)} must be a string path.")
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare small prompt datasets (1 prompt per line).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common args helper
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--n-samples", type=int, default=100)

    # WikiText
    ap_w = sub.add_parser("wikitext", help="Process WikiText-2-like txt -> first sentence prompts")
    add_common(ap_w)
    ap_w.add_argument("--input", type=str, required=False, help="Path to raw wikitext train.txt")
    ap_w.add_argument("--output", type=str, required=False, help="Path to write processed prompts.txt")
    ap_w.add_argument("--min-chars", type=int, default=40)
    ap_w.add_argument("--max-sentence-chars", type=int, default=280)
    ap_w.add_argument("--keep-section-titles", action="store_true", help="Do not drop section title lines")

    # CSV
    ap_c = sub.add_parser("csv", help="Process CSV -> narrative+question prompts")
    add_common(ap_c)
    ap_c.add_argument("--input", type=str, required=False, help="Path to raw csv")
    ap_c.add_argument("--output", type=str, required=False, help="Path to write processed prompts.txt")
    ap_c.add_argument("--narrative-col", type=str, default="narrative")
    ap_c.add_argument("--question-col", type=str, default="question")
    ap_c.add_argument("--min-chars", type=int, default=40)
    ap_c.add_argument("--max-chars", type=int, default=800)

    # All (uses config)
    ap_a = sub.add_parser("all", help="Run both wikitext and csv using configs/base.yaml paths")
    add_common(ap_a)
    ap_a.add_argument("--config", type=str, default="configs/base.yaml")

    # For single commands, config is optional; if missing input/output, config will be used.
    ap.add_argument("--config", type=str, default="configs/base.yaml", help="Config used for default paths")

    args = ap.parse_args()
    cfg = load_config(Path(args.config))

    if args.cmd == "wikitext":
        input_path = Path(args.input or get_cfg_path(cfg, "paths", "raw", "wikitext2_train"))
        output_path = Path(args.output or get_cfg_path(cfg, "paths", "processed", "wikitext2_100"))
        prepare_wikitext(
            input_path=input_path,
            output_path=output_path,
            n_samples=args.n_samples,
            seed=args.seed,
            min_chars=args.min_chars,
            max_sentence_chars=args.max_sentence_chars,
            drop_section_titles=not args.keep_section_titles,
        )

    elif args.cmd == "csv":
        input_path = Path(args.input or get_cfg_path(cfg, "paths", "raw", "tellmewhy_csv"))
        output_path = Path(args.output or get_cfg_path(cfg, "paths", "processed", "tellmewhy_100"))
        prepare_csv(
            input_path=input_path,
            output_path=output_path,
            n_samples=args.n_samples,
            seed=args.seed,
            narrative_col=args.narrative_col,
            question_col=args.question_col,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

    elif args.cmd == "all":
        # WikiText
        w_in = Path(get_cfg_path(cfg, "paths", "raw", "wikitext2_train"))
        w_out = Path(get_cfg_path(cfg, "paths", "processed", "wikitext2_100"))
        prepare_wikitext(
            input_path=w_in,
            output_path=w_out,
            n_samples=args.n_samples,
            seed=args.seed,
            min_chars=40,
            max_sentence_chars=280,
            drop_section_titles=True,
        )

        # CSV
        c_in = Path(get_cfg_path(cfg, "paths", "raw", "tellmewhy_csv"))
        c_out = Path(get_cfg_path(cfg, "paths", "processed", "tellmewhy_100"))
        prepare_csv(
            input_path=c_in,
            output_path=c_out,
            n_samples=args.n_samples,
            seed=args.seed,
            narrative_col="narrative",
            question_col="question",
            min_chars=40,
            max_chars=800,
        )


if __name__ == "__main__":
    main()