# CSE 151B Kaggle Competition

## Hardware & Runtime

| GPU | Approximate inference time |
|-----|---------------------------|
| A30 | ~2 hr 53 min |
| L40S | ~2 hr 53 min |

## Model Weights Setup

The notebook uses **Qwen/Qwen3-4B-Thinking-2507** loaded via vLLM with INT8 quantization (BitsAndBytes). Weights are downloaded automatically from Hugging Face on first run — no manual download is required.

If you want to pre-download or use a local copy, place the model directory at a path of your choice and update `MODEL_ID` in the notebook's configuration cell to that path:

```python
MODEL_ID = "/path/to/Qwen3-4B-Thinking-2507"
```

The default Hugging Face cache is `~/.cache/huggingface/hub/`. To redirect it, set the environment variable before launching the notebook:

```bash
export HF_HOME=/your/storage/path
jupyter notebook cse151b_comp.ipynb
```

## Reproducing Results

### 1. Install dependencies

```bash
uv venv .venv --seed --clear
.venv/bin/python -m pip install --no-cache-dir \
    sympy numpy transformers vllm tqdm bitsandbytes \
    antlr4-python3-runtime==4.11.1 ipykernel jupyter
```

### 2. Place the dataset

Put the private test file at:

```
data/private.jsonl
```

### 3. Run inference

Open `cse151b_comp.ipynb`, run the setup and helper cells, then execute the final cell:

```python
results = run_inference()
```

This single call will:
1. Load `data/private.jsonl`
2. Download and initialize the model (Qwen3-4B-Thinking-2507, INT8)
3. Generate responses in batches of 20, saving progress to `response_cache_9.json` after each batch
4. Write `results/results.jsonl` (JSONL) and `results/results.csv` (CSV, submission-ready)

## Model Parameters

### vLLM Engine

| Parameter | Value | Description |
|-----------|-------|-------------|
| `quantization` | `bitsandbytes` | INT8 weight quantization |
| `load_format` | `bitsandbytes` | On-the-fly BnB weight loading |
| `enable_prefix_caching` | `True` | Cache shared prompt prefixes across requests |
| `gpu_memory_utilization` | `0.8` | Fraction of GPU VRAM reserved for model + KV cache |
| `max_model_len` | `8192` | Maximum sequence length (prompt + generation) |
| `max_num_seqs` | `32` | Maximum sequences processed in parallel |
| `max_num_batched_tokens` | `8192` | Maximum total tokens per forward pass |

### Sampling

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | `0.1` | Low temperature for near-deterministic output |
| `top_p` | `0.95` | Nucleus sampling cutoff |
| `top_k` | `20` | Top-k token candidates |
| `min_p` | `0.0` | Minimum probability threshold (disabled) |
| `presence_penalty` | `0.0` | No presence penalty |
| `repetition_penalty` | `1.0` | No repetition penalty |
| `max_tokens` | `8192` | Maximum new tokens per response |

---

### Configuration

All tunable parameters are at the top of the configuration cell:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-4B-Thinking-2507` | Model name or local path |
| `GPU_ID` | `0` | `CUDA_VISIBLE_DEVICES` |
| `DATA_PATH` | `data/private.jsonl` | Input dataset |
| `OUTPUT_PATH` | `results/results.jsonl` | JSONL output (CSV uses same stem) |
| `CACHE_FILE` | `response_cache_9.json` | Incremental response cache |
| `MAX_TOKENS` | `8192` | Max tokens generated per response |
| `BATCH_SIZE` | `20` | Sequences per vLLM batch |
