# VMAVul

VMAVul generates C/C++ vulnerability samples for training vulnerability
detectors. The implementation follows the paper's Vulnerability Manifestation Atlas
(VMA), coverage-guided generation, tool-supported validation, refinement loop, and
downstream detector evaluation.

This release contains:

- the generation pipeline described in Section 3;
- [`vmavul_samples.jsonl`](vmavul_samples.jsonl), synthetic vulnerable and patched functions with the paper's five-dimensional descriptors;
- [`rq2_results.csv`](rq2_results.csv), the Section 4.3 augmentation results.

## Method

### Vulnerability Manifestation Atlas

VMAVul represents each sample with five measured dimensions:

| Dimension | Meaning | Values |
|---|---|---|
| D1 | Weakness family | Buffer Overflow, NULL Dereference, Use After Free, Integer Overflow, Command Injection, SQL Injection, Path Traversal, Resource Leak, Division by Zero, Format String |
| D2 | Intra-procedural taint length | Zero, Short, Medium, Long |
| D3 | Control-flow form at the sink | Linear, Conditional, Loop, Nested, Exception |
| D4 | Source-referencing guard | Guarded, Unguarded |
| D5 | Size-complexity tier | Small, Medium, Large |

The Cartesian product contains 1,200 cells. The paper's three admissibility constraints
exclude 264 cells, leaving 936 admissible cells. Run `vmavul space` to inspect the count.

The ten families cover the CWE sets used in the paper:

| Family | CWEs |
|---|---|
| Buffer Overflow | CWE-119, CWE-125, CWE-787 |
| NULL Dereference | CWE-476 |
| Use After Free | CWE-415, CWE-416 |
| Integer Overflow | CWE-190, CWE-191 |
| Command Injection | CWE-77, CWE-78 |
| SQL Injection | CWE-89 |
| Path Traversal | CWE-22 |
| Resource Leak | CWE-401, CWE-404 |
| Division by Zero | CWE-369 |
| Format String | CWE-134 |

### Generation and scheduling

Each generation request contains a target VMA cell, examples from the corresponding
weakness family in the training split, and relevant successful repairs from refinement
memory. The model returns a vulnerable function, its patched counterpart, and a shared
calling context.

The scheduler has three stages:

1. **Bootstrapping** processes each weakness family until accepted samples cover three
   distinct cells, or three consecutive attempts add no new cell.
2. **Coverage expansion** prioritizes uncovered cells with the most covered neighbors.
   An uncovered cell is exhausted after `k = 10` failed attempts.
3. **Densification** fills covered cells below `tau = 50`, then varies archived samples
   toward uncovered neighboring cells or diversifies samples within their current cell.

### Validation and refinement

Candidates pass four checks in increasing order of cost:

1. both vulnerable and patched programs compile with Clang;
2. CodeQL or a sanitizer reports the target weakness in the vulnerable function, the
   patched version removes it, and no contradictory or unrelated runtime error appears;
3. Joern and Lizard recover a complete, admissible D1-D5 descriptor;
4. the minimum UniXcoder cosine distance to samples already in the measured cell is at
   least `delta = 0.2`.

On failure, the diagnostic from the failed check is sent to a refiner from a different
model family. A candidate receives at most `r = 3` refinement rounds. Accepted samples
are archived under their measured cell, not merely the requested cell. Runs persist the
archive, scheduler state, refinement memory, and per-iteration evidence and can resume.

## Repository layout

```text
configs/                 VMAVul, model-pool, sink/source, and tool-version configuration
prompts/                 generation and diagnostic-driven refinement templates
src/vmavul/              VMA, scheduler, agent, validators, archive, and LLM clients
scripts/data/            training-split reference-pool builder
scripts/baselines/       adapter for public baseline outputs
scripts/rq2/             downstream LineVul and LoRA detector experiments
scripts/run_synthesis.sh end-to-end VMAVul generation and 15,000-sample export
docker/Dockerfile        pinned validation-tool environment
```

## Installation

### Docker

The image installs the validation tools and the Python environment:

```bash
docker build -f docker/Dockerfile -t vmavul .
docker run --rm -it \
  -v "$PWD/data:/vmavul/data" \
  -v "$PWD/runs:/vmavul/runs" \
  vmavul bash
```

### Manual setup

The versions used in the paper are recorded in `configs/toolchain.json`:

| Component | Version or revision | Purpose |
|---|---|---|
| Python | 3.10 or newer | pipeline and experiments |
| Clang and compiler-rt | 17.0.6 | syntax and ASan/UBSan/LSan checks |
| CodeQL CLI | 2.25.4 | static weakness checks |
| `codeql/cpp-queries` | 1.6.2 | C/C++ queries |
| Joern | 4.0.547 | D2-D4 characterization |
| Lizard | 1.23.0 | D5 characterization |
| UniXcoder | `microsoft/unixcoder-base` | within-cell diversity |
| LineVul | commit `9401ec6e60b762a9308645961268203244bee048` | downstream detector |

```bash
uv sync --extra ml
codeql pack download codeql/cpp-queries@1.6.2 --dir <qlpacks>

export VMAVUL_CLANG=clang-17
export VMAVUL_CODEQL=<codeql-dir>/codeql
export VMAVUL_CODEQL_CPP_QUERIES=<qlpacks>/codeql/cpp-queries/1.6.2
export VMAVUL_JOERN_HOME=<joern-cli-dir>
export VMAVUL_EMBED_MODEL=microsoft/unixcoder-base
export JAVA_HOME=<jdk-17-dir>
export PATH="$JAVA_HOME/bin:$PATH"

uv run vmavul check-tools --config configs/vmavul.yaml
```

Generated programs are untrusted. Runtime checks default to a fail-closed bubblewrap
sandbox with no network, a read-only host, masked user/data directories, and one writable
temporary directory. If bubblewrap is unavailable, runtime validation is skipped rather
than run on the host. Use `VMAVUL_SANDBOX=none` only inside a disposable container.

## Models

Bootstrapping uses the two API models in the paper:

- GPT-5.6;
- Claude Opus 4.8.

Coverage expansion and densification use four locally served models:

- Qwen3.6-35B-A3B;
- Gemma-4-31B-IT;
- DeepSeek-R1-Distill-Llama-8B;
- GLM-4.7-Flash.

Local models are accessed through OpenAI-compatible endpoints. Start the vLLM servers
and set the URLs used by `configs/models.yaml`, for example:

```bash
vllm serve /models/GLM-4.7-Flash \
  --served-model-name zai-org/GLM-4.7-Flash \
  --port 8004

export OPENAI_API_KEY=<key>
export ANTHROPIC_API_KEY=<key>
export VMAVUL_QWEN_URL=http://localhost:8001/v1
export VMAVUL_GEMMA_URL=http://localhost:8002/v1
export VMAVUL_DEEPSEEK_URL=http://localhost:8003/v1
export VMAVUL_GLM_URL=http://localhost:8004/v1
```

`VMAVUL_LOCAL_API_KEY` is optional for endpoints that do not require authentication.

## Datasets

Download the four datasets from their official releases:

| Dataset | Official source |
|---|---|
| MegaVul | <https://github.com/Icyrockton/MegaVul> |
| PrimeVul | <https://github.com/DLVulDet/PrimeVul> |
| ICVul | <https://github.com/Chaomeng-Lu/ICVul> |
| TitanVul | <https://huggingface.co/datasets/yikun-li/TitanVul> |

Place normalized splits at:

```text
data/datasets/<dataset>/train.jsonl
data/datasets/<dataset>/valid.jsonl
data/datasets/<dataset>/test.jsonl
```

Each JSONL record must contain:

```json
{"id":"...","func":"int f(...) {...}","target":1,"cwe":"CWE-787","split":"train"}
```

`target` is `1` for vulnerable and `0` for non-vulnerable. Training positives used as
generation references also require a CWE from one of the ten families. Use the official
train, validation, and test splits released with each dataset. The repository does not
download or redistribute those releases.

Build a reference pool exclusively from vulnerable functions in a training split:

```bash
uv run python scripts/data/build_reference_pool.py --train-set megavul
```

The script excludes normalized functions that occur in any of the four test splits.

## Generate samples

Run the complete pipeline for one training dataset:

```bash
scripts/run_synthesis.sh megavul
```

Equivalent explicit commands are:

```bash
uv run python scripts/data/build_reference_pool.py --train-set megavul
VMAVUL_TRAIN_SET=megavul uv run vmavul synthesize \
  --config configs/vmavul.yaml \
  --reference-pool data/reference_pool/megavul.jsonl
uv run vmavul export \
  --run runs/megavul/vmavul \
  --out runs/megavul/vmavul/samples.jsonl
uv run python scripts/sample_augmentation.py \
  --input runs/megavul/vmavul/samples.jsonl \
  --out data/augmentation/megavul/vmavul.jsonl \
  --n 15000 --seed 0
```

A synthesis run writes `archive.jsonl`, `events.jsonl`, `refinement_memory.jsonl`,
`scheduler_state.json`, `run_config.json`, `space.json`, and `summary.json` beneath its
run directory.

## Downstream experiments

The paper evaluates augmentation with LineVul, Qwen3.5-4B, and
Llama3.1-8B-Instruct across the four datasets and five detector seeds. Checkpoint
selection uses only the validation split; each selected checkpoint is then evaluated on
all four test sets.

Install the pinned LineVul implementation:

```bash
scripts/rq2/setup_linevul.sh
```

The adapter converts JSONL inputs to LineVul's CSV format, runs the pinned implementation, and collects its metrics.

List or run the full downstream matrix:

```bash
uv run python scripts/rq2/run_matrix.py --out results/rq2
uv run python scripts/rq2/run_matrix.py --out results/rq2 --run --gpus 0,1,2,3
uv run python scripts/rq2/aggregate.py --results results/rq2 > results/rq2/tables.json
```

LineVul uses the official CodeBERT-based implementation. The two LLM detectors are
fine-tuned with LoRA. Fixed training defaults are declared by the two adapters and are
held constant across augmentation settings.

## License

The repository code is released under the MIT license. Third-party tools, datasets, and
model weights remain subject to their providers' licenses and are not redistributed.
