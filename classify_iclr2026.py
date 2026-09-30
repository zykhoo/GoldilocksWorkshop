#!/usr/bin/env python3
"""
classify_iclr2026.py

Loads ICLR 2026 papers from Paper Copilot (github.com/papercopilot/paperlists),
then goes through them one by one, reading each title + abstract and asking a
Hugging Face model whether the paper is relevant to:

  1. SATURATION - identifying when additional investment ceases to help or
                  begins to hurt
  2. ALLOCATION - determining how a limited budget (compute, data, training,
                  human effort, other resources) should be divided across
                  interacting components
  3. SYNERGY    - identifying combinations whose joint effect exceeds what
                  their individual performance would suggest

For each paper the model first identifies the main configuration/component/
feature the paper aims to increase or decrease, checks whether the paper
explores too little vs. too much of it, and then classifies the pattern.
A paper can be tagged with any subset of the three, or none. Each paper's
submission area (primary_area), keywords, topic and status from Paper Copilot
are carried through to the output, and results are also broken down by area.

Setup:
    pip install requests transformers accelerate torch huggingface_hub

Usage:
    python classify_iclr2026.py --limit 20                    # quick test run
    python classify_iclr2026.py                               # accepted papers, local model
    python classify_iclr2026.py --scope all                   # all submissions
    python classify_iclr2026.py --model Qwen/Qwen2.5-14B-Instruct
    python classify_iclr2026.py --backend api                 # HF Inference Providers
                                                              # (needs HF_TOKEN)
    python classify_iclr2026.py --papers-file iclr2026.json   # use a local copy

Runs are resumable: already-classified papers in the results file are skipped.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict

CATEGORIES = ("saturation", "allocation", "synergy")

# Paper Copilot stores large files in Git LFS, so raw.githubusercontent.com
# only returns a pointer; media.githubusercontent.com serves the real file.
PAPERCOPILOT_URLS = [
    "https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr2026.json",
    "https://raw.githubusercontent.com/papercopilot/paperlists/main/iclr/iclr2026.json",
]

NOT_ACCEPTED = {"reject", "withdraw", "withdrawn", "desk reject", "desk rejected", "active", ""}

# --------------------------------------------------------------------------
# 1. Load papers from Paper Copilot
# --------------------------------------------------------------------------

def download_papercopilot(dest):
    import requests
    for url in PAPERCOPILOT_URLS:
        print(f"Downloading {url} ...")
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
        except Exception as e:
            print(f"  failed: {e}")
            continue
        if r.content.startswith(b"version https://git-lfs"):
            print("  got a Git LFS pointer, not the data; trying next source")
            continue
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    sys.exit(
        "Could not download iclr2026.json from Paper Copilot. Clone the repo with\n"
        "git-lfs installed (git clone https://github.com/papercopilot/paperlists)\n"
        "and pass --papers-file paperlists/iclr/iclr2026.json"
    )


def _s(x):
    return x.strip() if isinstance(x, str) else ("" if x is None else str(x))


def load_papers(papers_file, scope):
    if not os.path.exists(papers_file):
        download_papercopilot(papers_file)
    with open(papers_file, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"\nPaper Copilot records: {len(raw)}")
    print("Status counts:", dict(Counter(_s(p.get("status")) for p in raw).most_common()))

    papers = []
    for p in raw:
        status = _s(p.get("status"))
        if scope == "accepted" and status.lower() in NOT_ACCEPTED:
            continue
        title = _s(p.get("title"))
        if not title:
            continue
        papers.append({
            "id": _s(p.get("id")),
            "title": title,
            "abstract": _s(p.get("abstract")),
            "status": status,
            "track": _s(p.get("track")),
            "primary_area": _s(p.get("primary_area")) or "(none)",
            "topic": _s(p.get("topic")),
            "keywords": _s(p.get("keywords")),
            "url": _s(p.get("site")) or f"https://openreview.net/forum?id={_s(p.get('id'))}",
        })
    print(f"Papers in scope '{scope}': {len(papers)}\n")
    return papers


# --------------------------------------------------------------------------
# 2. Hugging Face classifier
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ML researcher screening ICLR 2026 papers for a workshop
on "Goldilocks" problems in AI: problems about finding the right AMOUNT,
SPLIT, or COMBINATION of something, rather than simply maximizing it.

You will get a paper's title and abstract. Use ONLY what the title and
abstract state or clearly support. Fill in the fields in the order below.
The evidence fields come first; the verdict fields must follow from them.

STEP 1 - configuration
Name the specific quantity, budget split, or combination of components that
the paper's MAIN contribution varies, studies, or optimizes. Be concrete,
e.g. "chain-of-thought length", "split of compute between pretraining and
fine-tuning", "mix of verifier rewards and reward-model scores".
Never answer with a theme name such as "saturation", "allocation",
"synergy", or "interaction".
If the main contribution does not vary or optimize any such thing, write
"N/A".

STEP 2 - goldilocks_question
Write the specific Goldilocks question the paper answers, as one question,
e.g. "At what reasoning length does accuracy stop improving?".
If you cannot write one that is grounded in the abstract, write "N/A".

STEP 3 - too_little, too_much, just_right
One sentence each on what the abstract says happens with too little, too
much, and the right amount or combination of the configuration. Write "N/A"
for any the abstract does not describe. These may be "N/A" even for relevant
papers, e.g. theory or methods papers.

STEP 4 - relevance
"direct": goldilocks_question is not "N/A" and answering it is a central
  contribution. This includes theory or methods for finding, predicting, or
  tracking the right amount, split, or combination (e.g. scaling laws that
  locate an optimum, optimal stopping, budget-allocation algorithms).
"adjacent": related and could interest the workshop audience, but the
  Goldilocks question is not central to the paper.
"not_relevant": the connection is generic or incidental.
If configuration or goldilocks_question is "N/A", relevance cannot be
"direct". Most papers are "not_relevant"; only a small minority are
"direct".

STEP 5 - themes
Set each theme to true or false independently. If relevance is
"not_relevant", all three must be false.

saturation: more of the configuration helps at first, then gains diminish,
  plateau, or reverse, or new failure modes appear.
  Examples: model size, training data or steps, test-time compute,
  reasoning length, context length, number of agents, amount of feedback,
  number of tool calls.
  NOT saturation: something is merely expensive or hard to scale.

allocation: a fixed, limited budget must be split among competing uses, so
  giving more to one leaves less for another.
  Examples: model size vs. training tokens; pretraining vs. post-training
  compute; training vs. inference compute; data-mixture proportions;
  dividing a budget across agents, experts, pipeline stages, or tasks.
  NOT allocation: the paper merely operates under a budget, or routes inputs
  for efficiency without studying how the budget should be split.

synergy: the right combination of components beats what each achieves on
  its own, and the paper studies or exploits that interaction.
  Examples: combining models, agents, modalities, data sources, reward
  signals, tools, retrieval, or human and AI input.
  NOT synergy: a method that merely has several parts, or a full method
  that beats its own ablations.

COMMON MISTAKES TO AVOID
- Do not decide from title words. "Synergizing", "hybrid", "adaptive",
  "routing", "balancing", "efficient", or "scaling" in a title are not
  evidence by themselves.
- Ordinary hyperparameter tuning, ablations, architecture search,
  efficiency gains, or combining modules are not enough.
- A new method that simply performs better is not a Goldilocks paper unless
  it answers a Goldilocks question.

STEP 6 - confidence, goldilocks_track, relevance_explanation
confidence: "high" if the abstract says it explicitly, "medium" if some
  interpretation is needed, "low" if it is mostly inferred.
goldilocks_track: "empirical_systems" if the paper empirically
  characterizes where or why these effects occur; "optimization_theory" if
  it develops theory or methods for finding, navigating, tracking, or
  shifting the right region; "both"; or "neither" (always "neither" when
  relevance is "not_relevant").
relevance_explanation: one sentence giving the concrete reason.

EXAMPLES (abstracts shortened)

Abstract: "Accuracy of reasoning models rises with chain-of-thought length up
to a task-dependent point and then falls. We characterize this point and
propose a method that stops reasoning near it."
{"configuration": "chain-of-thought length", "goldilocks_question": "At what reasoning length does accuracy peak, and how can a model stop there?", "too_little": "Short reasoning leaves hard problems unsolved.", "too_much": "Past a task-dependent point, longer reasoning lowers accuracy.", "just_right": "Stopping near the peak gives the best accuracy.", "relevance": "direct", "saturation": true, "allocation": false, "synergy": false, "confidence": "high", "goldilocks_track": "both", "relevance_explanation": "It locates the reasoning length beyond which more thinking hurts and proposes a way to stop there."}

Abstract: "Given a fixed training compute budget, we derive how it should be
split between model parameters and training tokens, and validate the
predicted optimum empirically."
{"configuration": "split of a fixed compute budget between parameters and tokens", "goldilocks_question": "For a fixed compute budget, what split between model size and data minimizes loss?", "too_little": "N/A", "too_much": "N/A", "just_right": "The derived split minimizes loss for the given budget.", "relevance": "direct", "saturation": false, "allocation": true, "synergy": false, "confidence": "high", "goldilocks_track": "both", "relevance_explanation": "Its central result is how to divide a fixed compute budget between model size and data."}

Abstract: "Verifier rewards are stable but sparse; reward-model scores are
dense but noisy. We combine them, and the hybrid outperforms either signal
alone for RL on reasoning tasks."
{"configuration": "mix of verifier rewards and reward-model scores", "goldilocks_question": "How should sparse verifier rewards and dense reward-model scores be combined to beat either alone?", "too_little": "Verifier rewards alone are stable but give sparse signal.", "too_much": "Reward-model scores alone are dense but noisy.", "just_right": "The hybrid reward outperforms either signal alone.", "relevance": "direct", "saturation": false, "allocation": false, "synergy": true, "confidence": "high", "goldilocks_track": "empirical_systems", "relevance_explanation": "It shows two complementary reward signals combined outperform each used alone."}

Abstract: "We present EgoX, which synergizes an egocentric video encoder with
a language model for human action understanding, achieving state-of-the-art
results on three benchmarks."
{"configuration": "N/A", "goldilocks_question": "N/A", "too_little": "N/A", "too_much": "N/A", "just_right": "N/A", "relevance": "not_relevant", "saturation": false, "allocation": false, "synergy": false, "confidence": "high", "goldilocks_track": "neither", "relevance_explanation": "It is a new multimodal architecture that performs well, with no study of how the combination should be chosen."}

Abstract: "We introduce a router that skips transformer layers per input,
cutting inference cost by 30% with little loss in accuracy."
{"configuration": "number of layers executed per input", "goldilocks_question": "N/A", "too_little": "N/A", "too_much": "N/A", "just_right": "N/A", "relevance": "adjacent", "saturation": false, "allocation": false, "synergy": false, "confidence": "medium", "goldilocks_track": "neither", "relevance_explanation": "It adapts compute per input for efficiency but does not study how much compute is enough or how to split a budget."}

OUTPUT
Respond with ONLY one JSON object, no markdown, with exactly these keys in
this order:
{"configuration": "<specific quantity, split, or combination, or N/A>", "goldilocks_question": "<one question, or N/A>", "too_little": "<one sentence, or N/A>", "too_much": "<one sentence, or N/A>", "just_right": "<one sentence, or N/A>", "relevance": "direct" or "adjacent" or "not_relevant", "saturation": true or false, "allocation": true or false, "synergy": true or false, "confidence": "high" or "medium" or "low", "goldilocks_track": "empirical_systems" or "optimization_theory" or "both" or "neither", "relevance_explanation": "<one sentence>"}"""


class LocalHFClassifier:
    """Runs an instruct model locally with transformers."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name} locally (first run downloads weights)...")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def __call__(self, messages):
        # Ask explicitly for a dict so this works on both older transformers
        # (which returned a bare tensor) and newer ones (which return a
        # BatchEncoding dict by default).
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
            enable_thinking=False,  # Qwen3-style models: answer directly;
                                    # ignored by templates that don't use it
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=800, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)


class APIHFClassifier:
    """Calls the model through Hugging Face Inference Providers (needs HF_TOKEN)."""

    def __init__(self, model_name):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model_name, token=os.getenv("HF_TOKEN"))

    def __call__(self, messages):
        resp = self.client.chat_completion(messages=messages, max_tokens=800, temperature=0.0)
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("model returned an empty response")
        return content


RELEVANCE = ("direct", "adjacent", "not_relevant")
CONFIDENCE = ("high", "medium", "low")
GL_TRACKS = ("empirical_systems", "optimization_theory", "both", "neither")
THEME_WORDS = {"saturation", "allocation", "synergy", "interaction", "design space"}


def parse_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in model output: {text[:200]!r}")
    obj = json.loads(m.group(0))
    # normalise keys: "too little" / "too-little" / "Too_Little" -> "too_little"
    obj = {re.sub(r"[\s\-]+", "_", str(k).strip().lower()): v for k, v in obj.items()}

    def as_bool(v):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")

    def txt(*keys):
        for k in keys:
            if obj.get(k) is not None and str(obj[k]).strip():
                return str(obj[k]).strip()
        return "N/A"

    def enum(key, allowed, default):
        v = str(obj.get(key, "")).strip().lower().replace(" ", "_")
        return v if v in allowed else default

    out = {
        "configuration": txt("configuration"),
        "goldilocks_question": txt("goldilocks_question"),
        "too_little": txt("too_little"),
        "too_much": txt("too_much"),
        "just_right": txt("just_right", "just_nice"),
        "relevance": enum("relevance", RELEVANCE, "not_relevant"),
        **{k: as_bool(obj.get(k, False)) for k in CATEGORIES},
        "confidence": enum("confidence", CONFIDENCE, "low"),
        # named goldilocks_track so it doesn't overwrite Paper Copilot's "track"
        "goldilocks_track": enum("goldilocks_track", GL_TRACKS,
                                 enum("track", GL_TRACKS, "neither")),
        "relevance_explanation": txt("relevance_explanation"),
    }

    # Enforce the prompt's own rules, and record any correction in "flag".
    flags = []
    is_na = lambda s: s.strip().upper() in ("N/A", "NA", "NONE", "")
    if out["configuration"].strip().lower() in THEME_WORDS:
        flags.append("configuration was a theme name")
        out["configuration"] = "N/A"
    if out["relevance"] == "direct" and (is_na(out["configuration"])
                                         or is_na(out["goldilocks_question"])):
        flags.append("direct without configuration/question -> adjacent")
        out["relevance"] = "adjacent"
    if out["relevance"] == "not_relevant" and any(out[k] for k in CATEGORIES):
        flags.append("themes cleared because not_relevant")
        for k in CATEGORIES:
            out[k] = False
    out["flag"] = "; ".join(flags)
    return out


def classify_paper(generate, paper, max_retries=4):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Title: {paper['title']}\n\nAbstract: {paper['abstract'] or '(no abstract)'}"},
    ]
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return parse_json(generate(messages))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1} for {paper['id']}: {e!r}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)


# --------------------------------------------------------------------------
# 3. Main loop + summary
# --------------------------------------------------------------------------

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    return done


def label_of(r):
    tags = [k for k in CATEGORIES if r.get(k)]
    return "+".join(tags) if tags else "none"


def theme_counts(rows, title):
    n = len(rows)
    print(f"\n{title}  (n={n})")
    for k in CATEGORIES:
        c = sum(r[k] for r in rows)
        print(f"  {k:<11} {c:>6}")
    combos = Counter(label_of(r) for r in rows)
    for key in ["saturation", "allocation", "synergy",
                "saturation+allocation", "saturation+synergy",
                "allocation+synergy", "saturation+allocation+synergy", "none"]:
        print(f"    {key:<32} {combos.get(key, 0):>6}")


def summarize(papers, results, csv_path, area_csv_path):
    rows = [results[p["id"]] for p in papers if p["id"] in results]
    rows = [r for r in rows if "relevance" in r]  # skip old-format records
    n = len(rows)
    print("\n" + "=" * 70)
    print(f"Total ICLR 2026 papers in scope: {len(papers)}")
    print(f"Papers classified:               {n}")
    print("=" * 70)

    print("\nRelevance:")
    for k, c in Counter(r["relevance"] for r in rows).most_common():
        print(f"  {k:<14} {c:>6}  ({100 * c / max(n, 1):.1f}%)")
    print("Direct papers by confidence:",
          dict(Counter(r["confidence"] for r in rows if r["relevance"] == "direct")))
    print("Direct papers by track:",
          dict(Counter(r["goldilocks_track"] for r in rows if r["relevance"] == "direct")))
    print("Rows corrected by consistency rules:", sum(bool(r.get("flag")) for r in rows))

    direct = [r for r in rows if r["relevance"] == "direct"]
    theme_counts(direct, "Themes among DIRECT papers")
    theme_counts([r for r in rows if r["relevance"] in ("direct", "adjacent")],
                 "Themes among DIRECT + ADJACENT papers")

    by_area = defaultdict(Counter)
    for r in rows:
        a = by_area[r["primary_area"]]
        a["total"] += 1
        a[r["relevance"]] += 1
        if r["relevance"] == "direct":
            for k in CATEGORIES:
                a[k] += r[k]
    print("\nBy primary area (total / direct / adjacent / direct: sat alloc syn):")
    for area, c in sorted(by_area.items(), key=lambda kv: -kv[1]["direct"]):
        print(f"  {area[:50]:<50} {c['total']:>5} {c['direct']:>5} {c['adjacent']:>5}"
              f"   {c['saturation']:>4} {c['allocation']:>4} {c['synergy']:>4}")

    cols = ["id", "title", "status", "track", "primary_area", "topic", "keywords",
            "relevance", "confidence", "goldilocks_track",
            "saturation", "allocation", "synergy", "label",
            "configuration", "goldilocks_question", "too_little", "too_much",
            "just_right", "relevance_explanation", "flag", "url"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row = {**r, "label": label_of(r)}
            for k in CATEGORIES:
                row[k] = int(r[k])
            w.writerow([row.get(c, "") for c in cols])

    with open(area_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["primary_area", "total", "direct", "adjacent", "not_relevant",
                    "direct_saturation", "direct_allocation", "direct_synergy"])
        for area, c in sorted(by_area.items(), key=lambda kv: -kv[1]["total"]):
            w.writerow([area, c["total"], c["direct"], c["adjacent"], c["not_relevant"],
                        c["saturation"], c["allocation"], c["synergy"]])

    print(f"\nPer-paper results: {csv_path}")
    print(f"Per-area summary:  {area_csv_path}")
    print("Tip: in pandas, read the CSV with keep_default_na=False so 'N/A' isn't shown as blank.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["accepted", "all"], default="accepted",
                    help="accepted papers only (default) or all submissions")
    ap.add_argument("--backend", choices=["local", "api"], default="local",
                    help="run the HF model locally with transformers, or via HF Inference Providers")
    ap.add_argument("--model", default="Qwen/Qwen3-14B",
                    help="any Hugging Face chat/instruct model id")
    ap.add_argument("--limit", type=int, default=None,
                    help="only classify the first N papers (for testing)")
    ap.add_argument("--papers-file", default="iclr2026.json",
                    help="local Paper Copilot JSON (downloaded if missing)")
    ap.add_argument("--results", default="iclr2026_classifications_v3.jsonl")
    ap.add_argument("--csv", default="iclr2026_classifications_v3.csv")
    ap.add_argument("--area-csv", default="iclr2026_by_area_v3.csv")
    args = ap.parse_args()

    papers = load_papers(args.papers_file, args.scope)
    if args.limit:
        papers = papers[:args.limit]

    results = load_done(args.results)
    todo = [p for p in papers if p["id"] not in results]
    print(f"{len(papers) - len(todo)} already classified, {len(todo)} to go.\n")

    if todo:
        generate = (LocalHFClassifier if args.backend == "local" else APIHFClassifier)(args.model)
        done_count = len(papers) - len(todo)
        consecutive_failures = 0
        for paper in todo:  # one by one
            try:
                cls = classify_paper(generate, paper)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"FAILED {paper['id']}: {e!r}", file=sys.stderr)
                if consecutive_failures == 1:
                    traceback.print_exc()
                if consecutive_failures >= 5:
                    sys.exit("\n5 papers in a row failed; stopping. See the traceback "
                             "above for the cause.")
                continue
            rec = {k: v for k, v in paper.items() if k != "abstract"}
            rec.update(cls)
            results[paper["id"]] = rec
            with open(args.results, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_count += 1
            flags = "".join(c[0].upper() if cls[c] else "-" for c in CATEGORIES)
            print(f"[{done_count}/{len(papers)}] {cls['relevance'][:8]:<8} {flags}  "
                  f"{paper['title'][:70]}")
            if cls["relevance"] != "not_relevant":
                print(f"      config: {cls['configuration'][:100]}")

    summarize(papers, results, args.csv, args.area_csv)


if __name__ == "__main__":
    main()