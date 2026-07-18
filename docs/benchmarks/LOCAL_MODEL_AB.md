# Local Model A/B

Date: 2026-07-18

## Verdict

The initial experiment did not support using ContextIR v0.3.0 as a
general-purpose semantic compressor. After correcting task scoring and adding
query-aware routing in v0.4.0, the same seven-case subset shows no aggregate
quality loss on any tested backend. Qwen3 1.7B quality increased from `0.5570`
raw to `0.6595` auto while model input fell by 69.0%.

This confirms the new algorithm on the bounded experiment, not in general.
The sample is small, one QA case still regressed, and 7-8B, labelled privacy,
and production agent-history evaluations remain missing.

## Initial v0.3.0 Result

The experiment does support two narrower claims:

- the local privacy boundary and provider-independent model adapter work with
  both Ollama and LM Studio;
- routing to raw input preserves quality when the policy recognizes that
  compression is unsafe.

The v0.3.0 routing heuristic did not reliably recognize retrieval, passage
counting, or arbitrary document QA. It selected `hybrid` for four of seven
cases, including contexts whose answer-bearing evidence was then omitted.

## Setup

- hardware: Apple M1 Pro, 16 GB unified memory;
- Ollama 0.32.0 with Qwen3 0.6B Q4 and Qwen3 1.7B Q4;
- LM Studio with Qwen3 0.6B Q8 and a 32K context window;
- deterministic temperature 0 and seed 42 where supported;
- 64 output-token limit;
- five official [LongBench v1](https://github.com/THUDM/LongBench) examples: two `multifieldqa_en`, two
  `passage_retrieval_en`, and one `passage_count`;
- one [RULER](https://github.com/NVIDIA/RULER)-style needle diagnostic and one repeated operational/privacy case.

The five LongBench examples are an official-data subset, not a LongBench
leaderboard run. The synthetic needle follows the RULER task shape but is not
an official RULER score.

## Results

| Backend and model | Mode | Quality | Model input tokens | Prompt ratio | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Ollama Qwen3 0.6B Q4 | raw | 0.2244 | 57,972 | 1.000 | 9.98 s |
| Ollama Qwen3 0.6B Q4 | auto | 0.1625 | 31,345 | 0.541 | 3.61 s |
| LM Studio Qwen3 0.6B Q8 | raw | 0.0383 | 57,916 | 1.000 | 11.24 s |
| LM Studio Qwen3 0.6B Q8 | auto | 0.0275 | 31,289 | 0.540 | 4.66 s |
| Ollama Qwen3 1.7B Q4 | raw | 0.5570 | 57,972 | 1.000 | 16.41 s |
| Ollama Qwen3 1.7B Q4 | auto | 0.3262 | 31,345 | 0.541 | 6.36 s |

Quality uses normalized token F1 for QA and the operational diagnostic,
LongBench-style extraction scores for retrieval and counting, and normalized
exact match for the needle task. It is averaged over seven cases, so it should
be read as a directional A/B measure rather than a population estimate. The
v0.3.0 report values above were recalculated after aligning the harness with
the retrieval and count task semantics.

On Qwen3 1.7B, the LongBench-only mean fell from `0.4731` raw to `0.1500` auto.
The two diagnostics were unchanged: auto selected raw for both. The 1.7B model
also scored `1.0` on the raw needle case, showing that the zero scores from the
0.6B model were not caused only by a broken harness or truncated context.

Forced modes were worse on Qwen3 0.6B:

| Backend | Hybrid quality | Semantic quality |
| --- | ---: | ---: |
| Ollama | 0.0087 | 0.0110 |
| LM Studio | 0.0000 | 0.0057 |

For short factual QA and repeated PII, the v0.3.0 intermediate representation
could also be larger than raw text. Repeated occurrences of one sensitive
value received distinct placeholders, which defeated deduplication.

## Query-Aware v0.4.0 Follow-Up

The follow-up keeps exhaustive counting raw, retrieves query-relevant evidence
verbatim, preserves paragraph ownership and output instructions, and reuses a
placeholder for repeated PII. The internal contract is no longer rendered into
the model prompt when selected source spans already cover its events.

| Backend and model | Raw quality | Query-aware auto | Auto input tokens | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Ollama Qwen3 0.6B Q4 | 0.2244 | 0.3791 | 17,961 | 69.0% |
| LM Studio Qwen3 0.6B Q8 | 0.0383 | 0.2052 | 17,905 | 69.1% |
| Ollama Qwen3 1.7B Q4 | 0.5570 | 0.6595 | 17,961 | 69.0% |

For Qwen3 1.7B, the LongBench-only mean increased from `0.4731` to `0.6167`.
Both passage retrieval cases and the needle diagnostic were answered exactly.
The operational/privacy diagnostic retained its raw score while using about 3%
of the estimated source tokens. The biomedical QA case fell from `0.6154` to
`0.3333`, so the aggregate improvement does not yet justify a no-regression
claim across task families.

## Cursor Control

Cursor Agent CLI was verified with:

```bash
agent --print --mode ask --trust --model gpt-5.3-codex-low 'Reply with only READY.'
```

It returned `READY`. A two-case raw/auto/semantic sanity check scored `0.7750`,
`0.7500`, and `0.2650`. Cursor Agent is a remote model control, not a local
small-model backend, so these results are reported separately and are not
included in the local comparison.

## Reproduce

Download the three named LongBench JSONL files into a local directory, then run:

```bash
python3 scripts/evaluate_model_ab.py \
  --backend ollama \
  --model qwen3:1.7b \
  --longbench-dir /tmp/contextir-longbench/data
```

The manifest stores only dataset names and row indices. LongBench source text
is not copied into this repository. Machine-readable outputs are in
`reports/model_ab/`.

## Remaining Work

- run the full official task sets with confidence intervals;
- add a conventional summary and embedding-retrieval baseline;
- test a 7-8B local model and production-shaped agent/tool histories;
- measure per-task regression gates rather than relying only on aggregate mean;
- evaluate PII precision, recall, and leakage on a labelled corpus;
- replace English marker heuristics with explicit application-supplied task and
  query fields where integrations can provide them.
