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

Two separate passes:
  1. screen        - which theme(s) each paper fits: saturation, allocation,
                     synergy, several (theme_group = "multiple"), or none.
  2. contribution  - only for papers with a theme: empirical or theoretical,
                     plus the workshop call question it answers.
    python classify_iclr2026.py --stage screen          # pass 1 only
    python classify_iclr2026.py --stage contribution    # pass 2 on pass-1 results

Runs are resumable: already-processed papers in each results file are skipped.
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

SCREEN_PROMPT = """You are an expert ML researcher screening ICLR 2026 papers for a workshop
on "Goldilocks" problems in AI. The workshop covers exactly three themes:

  SATURATION - more of an investment helps at first, then stops helping or
               starts to hurt.
  ALLOCATION - a fixed, limited budget must be split among competing uses 
               or components, with performance depending on much is assigned
               to each.
  SYNERGY    - a combination of components does better than the components
               achieve on their own.

Papers about a single trade-off dial (a setting where both extremes are bad
for different reasons, with no budget being split and no components being
combined) are OUT OF SCOPE, even if the paper finds an optimal value.

You will get a paper's title and abstract. Use ONLY what the title and
abstract state. Fill in the fields in the order below. The evidence fields
come first; the verdict fields must follow from them.

STEP 1 - configuration
Name the specific investment, budget split, or combination of components
that the paper's MAIN contribution varies or studies. Be concrete, e.g.
"chain-of-thought length", "split of compute between pretraining and
fine-tuning". Never answer with a theme name. If there is none, write "N/A".

STEP 2 - evidence
Copy or closely paraphrase the phrase from the abstract that REPORTS the
Goldilocks pattern as a finding or result, e.g. "accuracy rises with length
up to a point and then falls". A description of what the method does is not
evidence. If the abstract reports no such finding, write "N/A".

STEP 3 - goldilocks_question
The specific question the paper answers, as one question. If configuration
or evidence is "N/A", write "N/A".

STEP 4 - too_little, too_much, just_right
One sentence each on what the abstract says happens with too little, too
much, and the right amount or combination. Write "N/A" for any the abstract
does not describe.

STEP 5 - themes (each true or false, independently)
A theme is true ONLY if its evidence requirement is met in the abstract.

saturation: the configuration is an INVESTMENT of resources or effort
  (model size, data, training steps, test-time compute, reasoning length,
  context length, number of agents, feedback, tool calls), AND the abstract
  reports that gains diminish, plateau, or reverse, or that new failures
  appear as it grows.
  NOT saturation: a hyperparameter or setting that is not an investment
  (a guidance scale, a clipping bound, a temperature, a distance, a
  regularization weight), even if both extremes are bad.

allocation: the abstract names a FIXED TOTAL budget (compute, data,
  parameters, tokens, samples, annotation, time) AND studies or derives how
  splitting it among competing uses affects results.
  NOT allocation: a method that distributes something internally (e.g.
  assigns parameters, ranks, or attention to layers or samples) without a
  fixed total being traded off; or merely operating under a budget.

synergy: the abstract COMPARES the combination against the components used
  on their own and reports that the combination does better than they
  suggest, or it studies how the components interact.
  NOT synergy: a method that merely combines two or more parts, or a full
  method that beats its own ablations.

STEP 6 - relevance
"direct": at least one theme is true, and it is central to the paper.
"adjacent": related to a theme and could interest the audience, but no
  theme's evidence requirement is fully met, or it is not central.
"not_relevant": everything else, including single trade-off dials,
  hyperparameter tuning, and ordinary methods papers.
Rules: "direct" requires at least one true theme and evidence that is not
"N/A". If relevance is "not_relevant", all three themes must be false.
Most papers are "not_relevant"; only a small minority are "direct".

COMMON MISTAKES TO AVOID
- Rewriting a methods paper as a question ("How should X and Y be
  combined?") does not make it a Goldilocks paper. The abstract must report
  the pattern as a finding.
- Do not decide from words such as "synergizing", "hybrid", "adaptive",
  "allocation", "routing", "balancing", or "scaling".
- Hyperparameter tuning, ablations, architecture search, and efficiency
  gains are not enough.

STEP 7 - confidence, relevance_explanation
confidence: "high" only if the abstract states the pattern outright in the
  evidence phrase; "medium" if it needs some interpretation; "low" if it is
  mostly inferred. Use "medium" or "low" whenever in doubt.
relevance_explanation: one sentence giving the concrete reason.

EXAMPLES (abstracts shortened)

Abstract: "Accuracy of reasoning models rises with chain-of-thought length up
to a task-dependent point and then falls. We characterize this point and
propose a method that stops reasoning near it."
{"configuration": "chain-of-thought length", "evidence": "accuracy rises with chain-of-thought length up to a task-dependent point and then falls", "goldilocks_question": "At what reasoning length does accuracy peak?", "too_little": "Short reasoning leaves hard problems unsolved.", "too_much": "Past a task-dependent point, longer reasoning lowers accuracy.", "just_right": "Stopping near the peak gives the best accuracy.", "saturation": true, "allocation": false, "synergy": false, "relevance": "direct", "confidence": "high", "relevance_explanation": "It reports the reasoning length beyond which more thinking hurts."}

Abstract: "Given a fixed training compute budget, we derive how it should be
split between model parameters and training tokens, and validate the
predicted optimum empirically."
{"configuration": "split of a fixed compute budget between parameters and tokens", "evidence": "derive how a fixed compute budget should be split between parameters and tokens and validate the optimum", "goldilocks_question": "For a fixed compute budget, what split between model size and data minimizes loss?", "too_little": "N/A", "too_much": "N/A", "just_right": "The derived split minimizes loss for the given budget.", "saturation": false, "allocation": true, "synergy": false, "relevance": "direct", "confidence": "high", "relevance_explanation": "Its central result is how to divide a fixed compute budget between model size and data."}

Abstract: "Verifier rewards are stable but sparse; reward-model scores are
dense but noisy. We combine them, and the hybrid outperforms either signal
alone for RL on reasoning tasks."
{"configuration": "mix of verifier rewards and reward-model scores", "evidence": "the hybrid outperforms either signal alone", "goldilocks_question": "Does combining verifier and reward-model signals beat either alone?", "too_little": "Verifier rewards alone give sparse signal.", "too_much": "Reward-model scores alone are noisy.", "just_right": "The hybrid outperforms either signal alone.", "saturation": false, "allocation": false, "synergy": true, "relevance": "direct", "confidence": "high", "relevance_explanation": "It compares a combined reward against each signal alone and finds the combination better."}

Abstract: "We propose a token-selection method for fine-tuning that combines
a self-modulated criterion with a semantic-aware criterion, improving
performance across benchmarks."
{"configuration": "N/A", "evidence": "N/A", "goldilocks_question": "N/A", "too_little": "N/A", "too_much": "N/A", "just_right": "N/A", "saturation": false, "allocation": false, "synergy": false, "relevance": "not_relevant", "confidence": "high", "relevance_explanation": "It is a method with two parts and reports no comparison showing the combination beats each part alone."}

Abstract: "We propose an entropy-guided LoRA variant that adaptively allocates
and shares trainable parameters across layers, matching full LoRA with
fewer parameters."
{"configuration": "N/A", "evidence": "N/A", "goldilocks_question": "N/A", "too_little": "N/A", "too_much": "N/A", "just_right": "N/A", "saturation": false, "allocation": false, "synergy": false, "relevance": "not_relevant", "confidence": "high", "relevance_explanation": "It distributes parameters inside a method for efficiency, with no fixed budget being split among competing uses."}

Abstract: "Low classifier-free guidance scales produce off-prompt images while
high scales reduce diversity. We propose a method that adapts the guidance
scale at each timestep."
{"configuration": "classifier-free guidance scale", "evidence": "N/A", "goldilocks_question": "N/A", "too_little": "Low guidance produces off-prompt images.", "too_much": "High guidance reduces diversity.", "just_right": "N/A", "saturation": false, "allocation": false, "synergy": false, "relevance": "not_relevant", "confidence": "high", "relevance_explanation": "It tunes a single trade-off dial, which is not an investment, a budget split, or a combination."}

OUTPUT
Respond with ONLY one JSON object, no markdown, with exactly these keys in
this order:
{"configuration": "<specific investment, split, or combination, or N/A>", "evidence": "<phrase from the abstract reporting the pattern, or N/A>", "goldilocks_question": "<one question, or N/A>", "too_little": "<one sentence, or N/A>", "too_much": "<one sentence, or N/A>", "just_right": "<one sentence, or N/A>", "saturation": true or false, "allocation": true or false, "synergy": true or false, "relevance": "direct" or "adjacent" or "not_relevant", "confidence": "high" or "medium" or "low", "relevance_explanation": "<one sentence>"}"""


# Second, separate pass: run only on papers that the screening pass tagged
# with at least one theme. It decides whether the contribution is empirical
# or theoretical, and which workshop call question it answers.
CONTRIB_PROMPT = """You are an expert ML researcher. A screening step has already found that
the paper below fits one or more themes of a workshop on "Goldilocks"
problems in AI (saturation, allocation, synergy). Do NOT re-judge the
themes. Your only job is to decide what KIND of contribution the paper
makes, using ONLY its title and abstract.

STEP 1 - call_question
Pick the ONE workshop question below that the paper's main contribution
answers best, and give its code, e.g. "E2".

Empirical and systems questions:
E1 Saturation: along which axes do returns flatten or reverse, and how do
   the boundaries vary across configurations?
E2 Allocation: under a fixed budget, how should resources be divided across
   stages and components, and how does the empirically optimal allocation
   change with configurations?
E3 Synergy: which components interact in complementary or antagonistic
   ways, and how can genuine interaction effects be told apart from simply
   stronger individual components?
E4 What observable signals predict that a system is approaching, entering,
   or leaving a Goldilocks zone, without an exhaustive sweep?
E5 How stable are Goldilocks zones across constraints; which generalize and
   which are specific to one experimental setting?
E6 What failure modes arise on either side of a Goldilocks zone, and what
   do they reveal about the mechanisms behind saturation, poor allocation,
   or poor synergy?

Optimization and theory questions:
T1 Under what assumptions should an interior optimum or a broad
   near-optimal region exist, rather than an optimum at an extreme?
T2 How can the geometry and structure of Goldilocks zones be characterized,
   and what makes them identifiable from limited observations?
T3 How can optimization methods efficiently locate Goldilocks zones for
   expensive, noisy, high-dimensional, combinatorial, or interacting
   configurations?
T4 Allocation: when can optimal allocations be transferred across model
   scales, tasks, or budgets, and what scaling laws or invariances allow it?
T5 Synergy: how can an optimizer find useful interactions without
   evaluating all combinations, and how should complementarity or
   interference be represented in the search?
T6 Saturation: how can algorithms stay within a Goldilocks zone as
   configurations change over time?
T7 How can finite data and experimental noise be accounted for, to
   distinguish true Goldilocks zones from false optima caused by sampling
   variability?
T8 Can system or algorithm design shift, broaden, or remove the boundaries
   of a Goldilocks zone?

STEP 2 - contribution_type: "empirical" or "theoretical"
Decide from the paper's MAIN contribution, not from whether it contains any
experiments or any math.
"empirical": the main contribution is measuring, characterizing, or
  explaining the pattern through experiments, benchmarks, or system
  building.
"theoretical": the main contribution is theory, formal analysis, scaling
  laws derived as general rules, or an optimization method or algorithm for
  finding, tracking, transferring, or changing the right amount, split, or
  combination. A new optimization method counts as theoretical even if it
  is evaluated only by experiments.
Choose exactly one. If the paper does both, choose the one its central
claim rests on. It should normally match the letter of call_question.

STEP 3 - contribution_explanation
One sentence naming the main contribution and why it is empirical or
theoretical.

EXAMPLES (abstracts shortened)

Abstract: "Accuracy of reasoning models rises with chain-of-thought length up
to a task-dependent point and then falls. We measure this across 12 models
and 6 tasks and show the peak shifts with task difficulty."
{"call_question": "E1", "contribution_type": "empirical", "contribution_explanation": "Its main contribution is measuring where returns to reasoning length reverse across models and tasks."}

Abstract: "We prove that under mild smoothness assumptions the optimal split
of a fixed compute budget between parameters and tokens follows a power
law, and that this split transfers across scales."
{"call_question": "T4", "contribution_type": "theoretical", "contribution_explanation": "Its main contribution is a proof that the optimal compute split follows a transferable power law."}

Abstract: "We propose a bandit algorithm that finds strong combinations of
retrieval, tools, and prompting strategies while evaluating only a small
fraction of all combinations."
{"call_question": "T5", "contribution_type": "theoretical", "contribution_explanation": "Its main contribution is an optimization method for finding useful combinations without exhaustive search."}

OUTPUT
Respond with ONLY one JSON object, no markdown, with exactly these keys in
this order:
{"call_question": "<E1-E6 or T1-T8>", "contribution_type": "empirical" or "theoretical", "contribution_explanation": "<one sentence>"}"""


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
CONTRIB = ("empirical", "theoretical")
CALL_QS = tuple(f"E{i}" for i in range(1, 7)) + tuple(f"T{i}" for i in range(1, 9))
THEME_WORDS = {"saturation", "allocation", "synergy", "interaction", "design space"}


def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in model output: {text[:200]!r}")
    obj = json.loads(m.group(0))
    return {re.sub(r"[\s\-]+", "_", str(k).strip().lower()): v for k, v in obj.items()}


def parse_contribution(text):
    obj = extract_json(text)
    q = re.search(r"[ET][1-8]", str(obj.get("call_question", "")).upper())
    q = q.group(0) if q and q.group(0) in CALL_QS else "N/A"
    ct = str(obj.get("contribution_type", "")).strip().lower()
    flags = []
    if ct not in CONTRIB:
        if q == "N/A":
            raise ValueError(f"no usable contribution_type or call_question: {obj}")
        ct = "empirical" if q[0] == "E" else "theoretical"
        flags.append("contribution_type inferred from call_question")
    elif q != "N/A" and (q[0] == "E") != (ct == "empirical"):
        flags.append("contribution_type does not match call_question")
    return {
        "call_question": q,
        "contribution_type": ct,
        "contribution_explanation": str(obj.get("contribution_explanation", "")).strip() or "N/A",
        "contribution_flag": "; ".join(flags),
    }


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
        "evidence": txt("evidence"),
        "goldilocks_question": txt("goldilocks_question"),
        "too_little": txt("too_little"),
        "too_much": txt("too_much"),
        "just_right": txt("just_right", "just_nice"),
        "relevance": enum("relevance", RELEVANCE, "not_relevant"),
        **{k: as_bool(obj.get(k, False)) for k in CATEGORIES},
        "confidence": enum("confidence", CONFIDENCE, "low"),
        "relevance_explanation": txt("relevance_explanation"),
    }

    # Enforce the prompt's own rules, and record any correction in "flag".
    flags = []
    is_na = lambda s: s.strip().upper() in ("N/A", "NA", "NONE", "")
    if out["configuration"].strip().lower() in THEME_WORDS:
        flags.append("configuration was a theme name")
        out["configuration"] = "N/A"
    if out["relevance"] == "direct" and (is_na(out["configuration"])
                                         or is_na(out["evidence"])
                                         or is_na(out["goldilocks_question"])):
        flags.append("direct without configuration/evidence/question -> adjacent")
        out["relevance"] = "adjacent"
    if out["relevance"] == "direct" and not any(out[k] for k in CATEGORIES):
        # e.g. a single trade-off dial: out of scope for the three themes
        flags.append("direct with no theme -> adjacent")
        out["relevance"] = "adjacent"
    if out["relevance"] == "not_relevant" and any(out[k] for k in CATEGORIES):
        flags.append("themes cleared because not_relevant")
        for k in CATEGORIES:
            out[k] = False
    out["flag"] = "; ".join(flags)
    return out


def paper_text(paper):
    return f"Title: {paper['title']}\n\nAbstract: {paper['abstract'] or '(no abstract)'}"


def run_prompt(generate, system, user, parser, paper_id, max_retries=4):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return parser(generate(messages))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1} for {paper_id}: {e!r}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def screen_paper(generate, paper):
    """Pass 1: which theme(s), if any."""
    return run_prompt(generate, SCREEN_PROMPT, paper_text(paper), parse_json, paper["id"])


def contribution_paper(generate, paper, screen):
    """Pass 2: empirical vs theoretical, only for papers with a theme."""
    themes = [k for k in CATEGORIES if screen.get(k)]
    user = (paper_text(paper) +
            f"\n\nThemes found by screening: {', '.join(themes)}"
            f"\nConfiguration: {screen.get('configuration', 'N/A')}")
    return run_prompt(generate, CONTRIB_PROMPT, user, parse_contribution, paper["id"])


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


def theme_group(r):
    """saturation / allocation / synergy for one theme, multiple for 2+, none."""
    tags = [k for k in CATEGORIES if r.get(k)]
    return tags[0] if len(tags) == 1 else ("multiple" if tags else "none")


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


def summarize(papers, results, csv_path, area_csv_path, contrib=None):
    contrib = contrib or {}
    rows = [{**results[p["id"]], **contrib.get(p["id"], {})}
            for p in papers if p["id"] in results]
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
    print("Rows corrected by consistency rules:", sum(bool(r.get("flag")) for r in rows))

    direct = [r for r in rows if r["relevance"] == "direct"]
    print("\nDIRECT papers by theme group:")
    for g in ["saturation", "allocation", "synergy", "multiple"]:
        print(f"  {g:<11} {sum(theme_group(r) == g for r in direct):>6}")
    theme_counts(direct, "Themes among DIRECT papers")
    theme_counts([r for r in rows if r["relevance"] in ("direct", "adjacent")],
                 "Themes among DIRECT + ADJACENT papers")

    if any(r.get("contribution_type") for r in direct):
        print("\nDIRECT papers: theme group x contribution type (pass 2)")
        print(f"  {'':<11} {'empirical':>10} {'theoretical':>12} {'not run':>8}")
        for g in ["saturation", "allocation", "synergy", "multiple"]:
            grp = [r for r in direct if theme_group(r) == g]
            e = sum(r.get("contribution_type") == "empirical" for r in grp)
            t = sum(r.get("contribution_type") == "theoretical" for r in grp)
            print(f"  {g:<11} {e:>10} {t:>12} {len(grp) - e - t:>8}")
        print("  by call question:", dict(sorted(Counter(
            r.get("call_question", "not run") for r in direct).items())))

    by_area = defaultdict(Counter)
    for r in rows:
        a = by_area[r["primary_area"]]
        a["total"] += 1
        a[r["relevance"]] += 1
        if r["relevance"] == "direct":
            for k in CATEGORIES:
                a[k] += r[k]
            a["d_" + r.get("contribution_type", "not_run")] += 1
    print("\nBy primary area (total / direct / adjacent / direct: sat alloc syn):")
    for area, c in sorted(by_area.items(), key=lambda kv: -kv[1]["direct"]):
        print(f"  {area[:50]:<50} {c['total']:>5} {c['direct']:>5} {c['adjacent']:>5}"
              f"   {c['saturation']:>4} {c['allocation']:>4} {c['synergy']:>4}")

    cols = ["id", "title", "status", "track", "primary_area", "topic", "keywords",
            "relevance", "theme_group", "confidence",
            "saturation", "allocation", "synergy", "label",
            "configuration", "evidence", "goldilocks_question", "too_little", "too_much",
            "just_right", "relevance_explanation", "flag",
            "contribution_type", "call_question", "contribution_explanation",
            "contribution_flag", "url"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row = {**r, "label": label_of(r), "theme_group": theme_group(r)}
            for k in CATEGORIES:
                row[k] = int(r[k])
            w.writerow([row.get(c, "") for c in cols])

    with open(area_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["primary_area", "total", "direct", "adjacent", "not_relevant",
                    "direct_saturation", "direct_allocation", "direct_synergy",
                    "direct_empirical", "direct_theoretical"])
        for area, c in sorted(by_area.items(), key=lambda kv: -kv[1]["total"]):
            w.writerow([area, c["total"], c["direct"], c["adjacent"], c["not_relevant"],
                        c["saturation"], c["allocation"], c["synergy"],
                        c["d_empirical"], c["d_theoretical"]])

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
    ap.add_argument("--stage", choices=["screen", "contribution", "both"], default="both",
                    help="screen = themes only; contribution = empirical/theoretical pass "
                         "on already-screened papers; both = run both in order")
    ap.add_argument("--contrib-on", choices=["direct", "any"], default="direct",
                    help="run pass 2 on direct papers with a theme (default) or on any "
                         "paper with a theme, including adjacent ones")
    ap.add_argument("--results", default="iclr2026_classifications_v7.jsonl")
    ap.add_argument("--contrib-results", default="iclr2026_contribution_v7.jsonl")
    ap.add_argument("--csv", default="iclr2026_classifications_v7.csv")
    ap.add_argument("--area-csv", default="iclr2026_by_area_v7.csv")
    args = ap.parse_args()

    papers = load_papers(args.papers_file, args.scope)
    if args.limit:
        papers = papers[:args.limit]

    generate = None

    def get_generator():
        nonlocal generate
        if generate is None:
            cls = LocalHFClassifier if args.backend == "local" else APIHFClassifier
            generate = cls(args.model)
        return generate

    def run_pass(name, todo, fn, out_path, store, show):
        done_count, total = len(store), len(store) + len(todo)
        consecutive_failures = 0
        for paper in todo:  # one by one
            try:
                res = fn(paper)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"FAILED {paper['id']}: {e!r}", file=sys.stderr)
                if consecutive_failures == 1:
                    traceback.print_exc()
                if consecutive_failures >= 5:
                    sys.exit(f"\n5 papers in a row failed in the {name} pass; stopping.")
                continue
            rec = {"id": paper["id"], **res} if name == "contribution" else \
                  {**{k: v for k, v in paper.items() if k != "abstract"}, **res}
            store[paper["id"]] = rec
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_count += 1
            print(f"[{name} {done_count}/{total}] {show(res)}  {paper['title'][:70]}")

    # ---- Pass 1: themes
    results = load_done(args.results)
    if args.stage in ("screen", "both"):
        todo = [p for p in papers if p["id"] not in results]
        print(f"PASS 1 (themes): {len(papers) - len(todo)} done, {len(todo)} to go.\n")
        if todo:
            gen = get_generator()
            run_pass("screen", todo, lambda p: screen_paper(gen, p), args.results, results,
                     lambda r: f"{r['relevance'][:8]:<8} {theme_group(r):<10}")

    # ---- Pass 2: empirical vs theoretical, only papers with a theme
    contrib = load_done(args.contrib_results)
    if args.stage in ("contribution", "both"):
        themed = [p for p in papers if p["id"] in results
                  and any(results[p["id"]].get(k) for k in CATEGORIES)
                  and (args.contrib_on == "any" or results[p["id"]]["relevance"] == "direct")]
        todo = [p for p in themed if p["id"] not in contrib]
        print(f"\nPASS 2 (contribution type): {len(themed)} papers with a theme, "
              f"{len(todo)} to go.\n")
        if todo:
            gen = get_generator()
            run_pass("contribution", todo,
                     lambda p: contribution_paper(gen, p, results[p["id"]]),
                     args.contrib_results, contrib,
                     lambda r: f"{r['contribution_type']:<11} {r['call_question']:<3}")

    summarize(papers, results, args.csv, args.area_csv, contrib)


if __name__ == "__main__":
    main()