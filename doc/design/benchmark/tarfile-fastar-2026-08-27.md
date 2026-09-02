# `tarfile` versus `fastar` benchmark

Date: 2026-08-27

## Result

`fastar` was faster than Python's `tarfile` for archive creation and extraction
for every tested model. The largest creation improvement was 26.8% for Qwen3-4B;
the largest extraction improvement was 21.0% for SmolLM2-135M.

## Measurements

Times are wall-clock medians in milliseconds. The archive size is shown in GiB.

| Model | Source | Files | Archive | Backend | Repetitions | Create | Extract |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| BERT-tiny | 0.017 GiB | 21 | 0.015 GiB | `tarfile` | 3 | 622.77 | 65.62 |
| BERT-tiny | 0.017 GiB | 21 | 0.015 GiB | `fastar` | 3 | 470.27 | 55.43 |
| BERT-base-uncased | 0.411 GiB | 18 | 0.380 GiB | `tarfile` | 3 | 14,812.56 | 1,759.93 |
| BERT-base-uncased | 0.411 GiB | 18 | 0.380 GiB | `fastar` | 3 | 11,641.99 | 1,407.94 |
| SmolLM2-135M | 0.254 GiB | 27 | 0.200 GiB | `tarfile` | 3 | 20,282.59 | 1,555.28 |
| SmolLM2-135M | 0.254 GiB | 27 | 0.201 GiB | `fastar` | 3 | 17,748.32 | 1,229.10 |
| Falcon3-1B | 3.119 GiB | 18 | 2.427 GiB | `tarfile` | 1 | 279,253.88 | 19,212.79 |
| Falcon3-1B | 3.119 GiB | 18 | 2.430 GiB | `fastar` | 1 | 210,178.39 | 16,774.36 |
| Qwen3-4B | 8.855 GiB | 35 | 7.041 GiB | `tarfile` | 1 | 716,155.48 | 49,757.72 |
| Qwen3-4B | 8.855 GiB | 35 | 7.049 GiB | `fastar` | 1 | 523,890.82 | 45,352.64 |

## Relative speedup

Speedup is calculated as `tarfile time / fastar time`; values above 1.0 mean
that `fastar` is faster.

| Model | Creation speedup | Extraction speedup |
| --- | ---: | ---: |
| BERT-tiny | 1.32x (24.5%) | 1.18x (15.5%) |
| BERT-base-uncased | 1.27x (21.4%) | 1.25x (20.0%) |
| SmolLM2-135M | 1.14x (12.5%) | 1.27x (21.0%) |
| Falcon3-1B | 1.33x (24.7%) | 1.15x (12.7%) |
| Qwen3-4B | 1.37x (26.8%) | 1.10x (8.9%) |

## Methodology

- Models were downloaded before timing and were not loaded by Transformers.
- Each snapshot contained the canonical safetensors model files plus the
  tokenizer and configuration files; duplicate ONNX, Flax, TensorFlow, and
  other export formats were excluded.
- Both backends created gzip-compressed archives from the same model directory
  and extracted their own archive into a fresh directory.
- Every extraction was validated against the source file manifest and byte
  total.
- Smaller models used three repetitions. Falcon3-1B and Qwen3-4B used one
  repetition because their source sizes were 3.119 GiB and 8.855 GiB.
- Archive work used persistent storage under
  `/home/torsten.kilias/benchmark-work`; model snapshots were stored under
  `/home/torsten.kilias/benchmark-models`.

The raw benchmark data is available at
[tarfile-fastar-results.json](tarfile-fastar-results.json).

## Uncompressed archives

The benchmark was repeated without gzip compression, using `tarfile` mode `w`
and `fastar` mode `w`. Times are wall-clock medians in milliseconds.

| Model | Source | Files | Archive | Backend | Repetitions | Create | Extract |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| BERT-tiny | 0.017 GiB | 21 | 0.017 GiB | `tarfile` | 3 | 21.54 | 7.96 |
| BERT-tiny | 0.017 GiB | 21 | 0.017 GiB | `fastar` | 3 | 6.61 | 6.32 |
| BERT-base-uncased | 0.411 GiB | 18 | 0.411 GiB | `tarfile` | 3 | 157.42 | 132.65 |
| BERT-base-uncased | 0.411 GiB | 18 | 0.411 GiB | `fastar` | 3 | 149.86 | 124.60 |
| SmolLM2-135M | 0.254 GiB | 27 | 0.254 GiB | `tarfile` | 3 | 103.87 | 83.86 |
| SmolLM2-135M | 0.254 GiB | 27 | 0.254 GiB | `fastar` | 3 | 94.67 | 79.54 |
| Falcon3-1B | 3.119 GiB | 18 | 3.119 GiB | `tarfile` | 1 | 2,601.34 | 2,692.42 |
| Falcon3-1B | 3.119 GiB | 18 | 3.119 GiB | `fastar` | 1 | 2,657.36 | 2,543.84 |
| Qwen3-4B | 8.855 GiB | 35 | 8.855 GiB | `tarfile` | 1 | 18,535.65 | 17,807.44 |
| Qwen3-4B | 8.855 GiB | 35 | 8.855 GiB | `fastar` | 1 | 7,546.18 | 15,444.91 |

### Uncompressed relative speedup

| Model | Creation speedup | Extraction speedup |
| --- | ---: | ---: |
| BERT-tiny | 3.26x (69.3%) | 1.26x (20.6%) |
| BERT-base-uncased | 1.05x (4.8%) | 1.07x (6.1%) |
| SmolLM2-135M | 1.10x (8.9%) | 1.05x (5.2%) |
| Falcon3-1B | 0.98x (-2.2%) | 1.06x (5.5%) |
| Qwen3-4B | 2.46x (59.3%) | 1.15x (13.3%) |

The uncompressed raw benchmark data is available at
[tarfile-fastar-results-no-gzip.json](tarfile-fastar-results-no-gzip.json).

## Compressed versus uncompressed

Compression reduced archive size by 7.5% to 22.2%, while substantially
increasing creation and extraction time. The percentages below describe the
reduction from the uncompressed archive size.

| Model | Backend | Compressed | Uncompressed | Size reduction | Create: compressed / uncompressed | Extract: compressed / uncompressed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BERT-tiny | `tarfile` | 0.015 GiB | 0.017 GiB | 11.8% | 28.9x | 8.2x |
| BERT-tiny | `fastar` | 0.015 GiB | 0.017 GiB | 11.8% | 71.0x | 8.8x |
| BERT-base-uncased | `tarfile` | 0.380 GiB | 0.411 GiB | 7.5% | 94.2x | 13.3x |
| BERT-base-uncased | `fastar` | 0.380 GiB | 0.411 GiB | 7.5% | 77.7x | 11.3x |
| SmolLM2-135M | `tarfile` | 0.200 GiB | 0.254 GiB | 21.3% | 195.3x | 18.5x |
| SmolLM2-135M | `fastar` | 0.201 GiB | 0.254 GiB | 20.9% | 187.7x | 15.5x |
| Falcon3-1B | `tarfile` | 2.427 GiB | 3.119 GiB | 22.2% | 107.4x | 7.1x |
| Falcon3-1B | `fastar` | 2.430 GiB | 3.119 GiB | 22.1% | 79.1x | 6.6x |
| Qwen3-4B | `tarfile` | 7.041 GiB | 8.855 GiB | 20.5% | 38.6x | 2.8x |
| Qwen3-4B | `fastar` | 7.049 GiB | 8.855 GiB | 20.4% | 69.4x | 2.9x |

The ratios use the median timings from the compressed and uncompressed runs.
The large-model results use one repetition, so they should be treated as
indicative rather than statistically robust.

## Reproducing the benchmark

The benchmark runner is available at
[tarfile_fastar.py](tarfile_fastar.py). Pass each
model as `--model NAME=PATH`, use a persistent `--work-dir`, and select the
JSON output path with `--output`:

```shell
poetry run python doc/design/benchmark/tarfile_fastar.py \
  --model bert-tiny=/path/to/bert-tiny \
  --model qwen3-4b=/path/to/qwen3-4b \
  --output benchmark-results.json \
  --work-dir /path/to/persistent/benchmark-work
```

Use `--uncompressed-only` to benchmark only `.tar` archives.
