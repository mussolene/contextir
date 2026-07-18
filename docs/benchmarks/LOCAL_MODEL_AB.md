# Local Model A/B

Date: 2026-07-18

## Verdict

The experiment does not support using ContextIR v0.3.0 as a general-purpose
semantic compressor for arbitrary document QA. `auto` reduced model-reported
input tokens by about 46%, but reduced aggregate quality by 21-28% relative to
masked raw input on the tested local models. This misses the project's maximum
3% quality-loss exit criterion.

The experiment does support two narrower claims:

- the local privacy boundary and provider-independent model adapter work with
  both Ollama and LM Studio;
- routing to raw input preserves quality when the policy recognizes that
  compression is unsafe.

The current routing heuristic does not reliably recognize retrieval, passage
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
| Ollama Qwen3 0.6B Q4 | raw | 0.2101 | 57,972 | 1.000 | 9.98 s |
| Ollama Qwen3 0.6B Q4 | auto | 0.1625 | 31,345 | 0.541 | 3.61 s |
| LM Studio Qwen3 0.6B Q8 | raw | 0.0383 | 57,916 | 1.000 | 11.24 s |
| LM Studio Qwen3 0.6B Q8 | auto | 0.0275 | 31,289 | 0.540 | 4.66 s |
| Ollama Qwen3 1.7B Q4 | raw | 0.4141 | 57,972 | 1.000 | 16.41 s |
| Ollama Qwen3 1.7B Q4 | auto | 0.3262 | 31,345 | 0.541 | 6.36 s |

Quality is task-appropriate normalized token F1 for QA and the operational
diagnostic, and normalized exact match for retrieval, counting, and needle
tasks. It is averaged over seven cases, so it should be read as a directional
A/B measure rather than a population estimate.

On Qwen3 1.7B, the LongBench-only mean fell from `0.2731` raw to `0.1500` auto.
The two diagnostics were unchanged: auto selected raw for both. The 1.7B model
also scored `1.0` on the raw needle case, showing that the zero scores from the
0.6B model were not caused only by a broken harness or truncated context.

Forced modes were worse on Qwen3 0.6B:

| Backend | Hybrid quality | Semantic quality |
| --- | ---: | ---: |
| Ollama | 0.0087 | 0.0110 |
| LM Studio | 0.0000 | 0.0057 |

For short factual QA and repeated PII, the intermediate representation can
also be larger than raw text. Repeated occurrences of one sensitive value
currently receive distinct placeholders, which defeats deduplication in the
operational case.

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

## Next Experiment

The next implementation should be retrieval-aware rather than expanding the
current event heuristic:

1. keep the task instruction and query verbatim;
2. classify document QA, retrieval, counting, transformation, and operational
   history before selecting a compression strategy;
3. chunk long evidence and retrieve query-relevant source spans locally;
4. apply ContextIR events only to repeated operational history and constraints;
5. pass selected evidence verbatim with source references;
6. reuse one placeholder for repeated occurrences of the same protected value;
7. fall back to masked raw whenever evidence coverage cannot be verified.

That design keeps the useful privacy and policy layers while removing the
unsupported assumption that a fixed five-token event summary can preserve
arbitrary facts.
