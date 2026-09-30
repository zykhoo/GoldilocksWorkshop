# Goldilocks' Workshop 

Some mini projects to support the workshop. 


## 1. Counting Goldilocks Papers at ICLR 2026

> [!IMPORTANT]
> **The counts this script produces are a very conservative underestimate of the number of Goldilocks papers at ICLR 2026.** Treat them as a strict lower bound, not an estimate of the true number.
>
> The pipeline is designed to miss papers rather than include doubtful ones. It reads only each paper's title and abstract. It counts a paper only if the abstract itself *reports* a saturation, allocation, or synergy pattern as a finding. It excludes single trade-off dials, papers where the Goldilocks question is secondary, and anything the model rates as merely "adjacent". Many papers establish these patterns in ablations, appendices, or discussion sections that never reach the abstract, and all of those are missed. The real number of ICLR 2026 papers engaging with Goldilocks problems is almost certainly much higher.

### What this is

`classify_iclr2026.py` screens every accepted ICLR 2026 paper for relevance to a workshop on **Goldilocks problems in AI**: problems about finding the right *amount* (saturation), *split* (allocation), or *combination* (synergy) of something, rather than simply maximizing it. It uses an open-weight language model from Hugging Face, running locally, and reads each paper's title and abstract one at a time.

Each paper is sorted into the workshop's three themes:

| Theme | Definition | Example |
|---|---|---|
| **Saturation** | Identifying when additional investment ceases to help or begins to hurt. | e.g Accuracy rises with chain-of-thought length up to a point, then falls. |
| **Allocation** | Determining how a limited budget of compute, data, training, human effort, or other resources should be divided across interacting components. | e.g. Splitting a fixed compute budget between model parameters and training tokens. |
| **Synergy** | Identifying combinations whose joint effect exceeds what their individual performance would suggest. | e.g. A hybrid of verifier rewards and reward-model scores that beats either alone. |

A paper can fit one theme, several (`multiple`), or none.

Papers with themes are then classified by contribution type, **empirical** or **theoretical**, and matched to the workshop call question they answer best.

### How it works

The script runs two separate passes, each with its own prompt, so the model only makes one kind of decision at a time.

#### Pass 1: Themes (`--stage screen`)

For each paper, the model fills in evidence fields before any verdict:

1. **Configuration.** The specific investment, budget split, or combination of components the paper's main contribution varies or studies.
2. **Evidence.** A phrase from the abstract that *reports* the Goldilocks pattern as a finding. A description of what a method does does not count.
3. **Goldilocks question.** The specific question the paper answers.
4. **Too little / too much / just right.** What the abstract says happens at each point.
5. **Themes.** Each theme is true only if its evidence requirement is met in the abstract:
   - *Saturation:* the configuration is an investment of resources or effort, and the abstract reports that gains diminish, plateau, or reverse.
   - *Allocation:* the abstract names a fixed total budget and studies how splitting it affects results.
   - *Synergy:* the abstract compares the combination against the components used alone.
6. **Relevance.** One of:
   - `direct`: at least one theme is true and central to the paper.
   - `adjacent`: related, but no theme's evidence requirement is fully met.
   - `not_relevant`: everything else.

The code enforces the prompt's rules after every response and records any correction in the `flag` column. For example, a paper cannot be `direct` without a theme and supporting evidence, and a `not_relevant` paper cannot have themes.

**Explicitly out of scope:** single trade-off dials, meaning a setting where both extremes are bad for different reasons but no budget is split and no components are combined. Examples are guidance scales, clipping bounds, temperatures, and regularization weights. Hyperparameter tuning, ablations, architecture search, efficiency gains, and methods that merely combine several parts are also excluded.

#### Pass 2: Contribution type (`--stage contribution`)

This pass runs only on papers that Pass 1 tagged with a theme (by default, `direct` papers only). It does not re-judge the themes. The model picks the workshop call question the paper answers best, then labels the main contribution:

- **Empirical** (questions E1–E6): the main contribution is measuring, characterizing, or explaining a Goldilocks pattern through experiments, benchmarks, or systems.
- **Theoretical** (questions T1–T8): the main contribution is theory, formal analysis, general scaling laws, or an optimization method for finding, tracking, transferring, or shifting a Goldilocks region. New optimization methods count as theoretical even when they are evaluated only experimentally.

The full list of call questions is in `CONTRIB_PROMPT` in the script.

### Data source

Paper metadata comes from [Paper Copilot](https://github.com/papercopilot/paperlists) (`iclr/iclr2026.json`), which provides each paper's title, abstract, decision status, primary area, keywords, and OpenReview link. By default only accepted papers (Poster, Spotlight, Oral) are screened; use `--scope all` to include every submission.

Paper Copilot stores this file in Git LFS. The script downloads it from `media.githubusercontent.com` automatically. If that fails, clone the repository with `git-lfs` installed and pass the file path:

```bash
git lfs install
git clone https://github.com/papercopilot/paperlists
python classify_iclr2026.py --papers-file paperlists/iclr/iclr2026.json
```

### Installation

```bash
pip install requests transformers accelerate torch huggingface_hub
```

For the API backend, also set a Hugging Face token:

```bash
export HF_TOKEN=hf_...
```

### Usage

Always start with a small test run and read the results before a full run:

```bash
python classify_iclr2026_v2.py --stage screen --model Qwen/Qwen3-14B --limit 50
```

Then run everything:

```bash
python classify_iclr2026_v2.py --model Qwen/Qwen3-14B                      # both passes, in order
python classify_iclr2026_v2.py --stage screen --model Qwen/Qwen3-14B       # Pass 1 only
python classify_iclr2026_v2.py --stage contribution --model Qwen/Qwen3-14B # Pass 2 on Pass 1 results
```

| Option | Default | Description |
|---|---|---|
| `--stage` | `both` | `screen`, `contribution`, or `both`. |
| `--model` | `Qwen/Qwen3.8-27B` | Any Hugging Face chat/instruct model ID. |
| `--backend` | `local` | `local` runs the model with `transformers`; `api` uses Hugging Face Inference Providers. |
| `--scope` | `accepted` | `accepted` papers only, or `all` submissions. |
| `--contrib-on` | `direct` | Run Pass 2 on `direct` papers with a theme, or on `any` paper with a theme. |
| `--limit` | none | Only process the first N papers. |
| `--papers-file` | `iclr2026.json` | Local Paper Copilot JSON; downloaded if missing. |

Both passes are **resumable**: papers already in a results file are skipped, so an interrupted run continues where it stopped. If you change the prompt or model, delete the old results files (or use new file names) so old results are not mixed in.

#### Hardware notes

The default local backend loads the model in its native precision. As a rough guide, a 14B model needs about 30 GB of GPU memory and a 27B model about 54 GB. On a 48 GB GPU such as an NVIDIA L40, a 27B model needs a quantized checkpoint (for example FP8). Thinking mode is turned off for Qwen3-style models so that the output budget is spent on the JSON answer.

The screening prompt asks for several steps in a fixed order, and smaller models tend to skip steps. If you see many flagged rows or vague `configuration` values, try a larger model.

### Output

| File | Contents |
|---|---|
| `iclr2026_classifications_v7.jsonl` | Pass 1 results, one JSON record per paper. |
| `iclr2026_contribution_v7.jsonl` | Pass 2 results, one JSON record per themed paper. |
| `iclr2026_classifications_v7.csv` | Both passes merged, one row per paper. |
| `iclr2026_by_area_v7.csv` | Counts per ICLR primary area. |

The script also prints a summary: relevance counts, direct papers by theme group, the full breakdown of theme combinations, theme group by contribution type, and counts by primary area.

#### Main CSV columns

| Column | Description |
|---|---|
| `id`, `title`, `status`, `track`, `primary_area`, `topic`, `keywords`, `url` | Paper metadata from Paper Copilot. |
| `relevance` | `direct`, `adjacent`, or `not_relevant`. |
| `theme_group` | `saturation`, `allocation`, `synergy`, `multiple`, or `none`. |
| `saturation`, `allocation`, `synergy` | 1 or 0 for each theme. |
| `label` | The exact combination, e.g. `allocation+synergy`. |
| `configuration` | What the paper varies or studies. |
| `evidence` | The phrase from the abstract that reports the pattern. |
| `goldilocks_question` | The question the paper answers. |
| `too_little`, `too_much`, `just_right` | What happens at each point, if the abstract says. |
| `confidence` | `high`, `medium`, or `low`. |
| `relevance_explanation` | One-sentence reason for the relevance decision. |
| `flag` | Any correction the code made to the model's answer. |
| `contribution_type` | `empirical` or `theoretical` (Pass 2). |
| `call_question` | Workshop call question code, E1–E6 or T1–T8 (Pass 2). |
| `contribution_explanation` | One-sentence reason for the contribution type (Pass 2). |
| `contribution_flag` | Any correction the code made in Pass 2. |

> [!TIP]
> Many cells contain the text `N/A`. pandas and Excel treat `N/A` as missing and show it as blank. In pandas, read the CSV with `pd.read_csv(path, keep_default_na=False)` to see the actual values.

### Limitations

**The counts are a lower bound.** As described at the top, the screening criteria are strict by design, and abstract-only screening misses any paper whose Goldilocks finding is not stated in its abstract. For a count closer to the true number, the pipeline would need to read introductions or full papers.

**The classifier is an LLM and makes mistakes in both directions.** The strict criteria mainly cause false negatives, but spot checks of earlier versions also found false positives. The most common were methods papers rephrased as Goldilocks questions, and methods that "allocate" something internally without a fixed budget being split. The `evidence` column is the quickest check: if a direct paper's evidence reads like a method description rather than a finding, it is likely a false positive.

**Theme boundaries involve judgment calls.** The exclusion of single trade-off dials, and the line between saturation (an investment) and a tunable setting, are deliberate choices for this workshop's scope. Other reasonable definitions would give different counts.

**Results depend on the model.** Different models, and different versions of the prompt, give different counts. Report the model and prompt version alongside any numbers.

**Validate before reporting numbers.** Before using the counts in a paper or proposal, hand-label a random sample of papers, including both positives and negatives, and report the agreement between the classifier and the human labels.

### Acknowledgements

Paper metadata is provided by [Paper Copilot](https://github.com/papercopilot/paperlists). Original paper records are hosted on [OpenReview](https://openreview.net/group?id=ICLR.cc/2026/Conference).
