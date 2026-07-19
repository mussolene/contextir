# Local Model A/B

Date: 2026-07-19

## Verdict

The initial experiment did not support using ContextIR v0.3.0 as a
general-purpose semantic compressor. After correcting task scoring and adding
query-aware routing in v0.4.0, the same seven-case subset shows no aggregate
quality loss on any tested backend. The v0.5.0 nine-case Qwen3 8B run preserves
every raw score, reduces model input by 70.0%, and cuts mean latency from
`53.8 s` to `2.51 s`.

This confirms the new algorithm on the bounded experiment, not in general.
The sample is still small. External supported-class privacy and synthetic agent
diagnostics are now present, but broad official and production-owned
evaluations remain missing.

The v1.3 constrained-context follow-up adds paired bootstrap intervals over ten
official QA/retrieval examples. It supports a positive quality-and-input result
for Qwen3 8B on that subset. The smaller 0.6B result remains inconclusive.

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
- Ollama with Qwen3 8B Q4 and a 32K context window;
- LM Studio with Qwen3 0.6B Q8 and a 32K context window;
- deterministic temperature 0 and seed 42 where supported;
- 64 output-token limit;
- five official [LongBench v1](https://github.com/THUDM/LongBench) examples: two `multifieldqa_en`, two
  `passage_retrieval_en`, and one `passage_count`;
- one [RULER](https://github.com/NVIDIA/RULER)-style needle diagnostic and one repeated operational/privacy case.
- two strict synthetic agent tool-routing and state-retrieval diagnostics.

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

## Qwen3 8B and Agent Diagnostics

The v0.5.0 run compares raw and auto over the seven earlier cases plus two
strict agent diagnostics. Auto selected hybrid for eight cases and correctly
kept exhaustive passage counting raw.

| Mode | Mean quality | Model input tokens | Prompt ratio | Mean latency |
|---|---:|---:|---:|---:|
| raw | 0.7272 | 61,081 | 1.0000 | 53.80 s |
| auto | 0.7272 | 18,193 | 0.2995 | 2.51 s |

Every case has identical raw and auto quality. Both tool routing and agent-state
retrieval score `1.0` in both modes; on those two cases auto uses 232 instead of
3,176 input tokens and reduces mean latency from `9.07 s` to `1.03 s`.
Passage counting scores zero in both modes, so the current failure is not caused
by compression. The suite is too small for a general no-regression claim.

## v1.3 Constrained Retrieval

The v1.3 follow-up uses the first five `multifieldqa_en` and first five
`passage_retrieval_en` LongBench examples with the same 2,048-token Ollama
context, 64 output tokens, and 32 reserved prompt-overhead tokens in both arms.
The raw arm is allowed to undergo the backend's normal context truncation. The
runtime arm calls `ContextPipeline.run(..., chunked_retrieval=True)`, records
every model call, and enforces a 1,952-token prompt budget before invocation.

| Model | Mode | Mean quality | 95% quality CI | Backend input | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4 | raw | 0.1889 | [0.0548, 0.3296] | 10,936 | 0.502 s |
| Qwen3 0.6B Q4 | runtime | 0.3121 | [0.1106, 0.5349] | 4,267 | 0.350 s |
| Qwen3 8B Q4 | raw | 0.1668 | [0.0405, 0.3213] | 10,936 | 1.446 s |
| Qwen3 8B Q4 | runtime | 0.6983 | [0.4700, 0.9026] | 4,267 | 1.002 s |

The paired comparison is more informative than the independent intervals:

| Model | Mean quality delta | Paired 95% CI | Better / tied / worse | Input ratio |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4 | +0.1232 | [-0.0154, 0.3392] | 4 / 5 / 1 | 0.3902 |
| Qwen3 8B Q4 | +0.5315 | [0.2086, 0.8225] | 7 / 2 / 1 | 0.3902 |

Latency was measured with locally resident, warmed models and is useful only
for the within-run comparison on this machine.

The 8B interval excludes zero on this subset. The 0.6B interval does not, so
its observed improvement is directional rather than established. Both models
regress on `multifieldqa_en_2`; deployment gates should therefore remain
per-task and must not rely only on aggregate quality.

All ten official runtime cases used one `direct` model call after query-aware
evidence packing. None needed map/reduce. A separate synthetic diagnostic puts
the answer near the start of one 3,000-word evidence segment so raw truncation
loses it. That case activated exactly one `map` call and changed extraction
success from 0 to 1 on both models. It is route coverage, not an official
quality estimate; on the 0.6B run the map prompt also used more backend tokens
than the truncated raw control.

Reproduce the official subset:

```bash
python3 scripts/evaluate_model_ab.py \
  --backend ollama \
  --model qwen3:8b \
  --modes raw,chunked \
  --case-ids multifieldqa_en_0,multifieldqa_en_1,multifieldqa_en_2,multifieldqa_en_3,multifieldqa_en_4,passage_retrieval_en_0,passage_retrieval_en_1,passage_retrieval_en_2,passage_retrieval_en_3,passage_retrieval_en_4 \
  --context-length 2048 \
  --max-output-tokens 64 \
  --longbench-dir /tmp/contextir-longbench/data
```

Machine-readable reports use the `_constrained_retrieval.json` and
`_chunk_map.json` suffixes under `reports/model_ab/`.

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
  --model qwen3:8b \
  --modes raw,auto \
  --longbench-dir /tmp/contextir-longbench/data
```

The manifest stores only dataset names and row indices. LongBench source text
is not copied into this repository. Machine-readable outputs are in
`reports/model_ab/`.

## Remaining Work

- run the full official task sets with confidence intervals;
- add a conventional summary and embedding-retrieval baseline;
- replace synthetic agent diagnostics with application-owned histories;
- measure per-task regression gates rather than relying only on aggregate mean;
- evaluate all required PII classes on deployment-shaped RU/EN corpora;
- replace English marker heuristics with explicit application-supplied task and
  query fields where integrations can provide them.
