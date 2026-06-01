#!/usr/bin/env python
# coding: utf-8

# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM (INT8 quantized)
# 4. Scoring responses against ground-truth answers
# 5. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# ### Comment Out the cell below after first installation.

# In[ ]:


# # Install uv
# !wget -qO- https://astral.sh/uv/install.sh | sh

# # Create a virtual environment
# !uv venv .venv --seed --clear

# # Install dependencies — this is fast thanks to uv's parallel resolver
# !.venv/bin/python -m pip install --no-cache-dir sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter
# #!~/.venv/bin/python -m pip install --no-cache-dir sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter

# # Install Jupyter Kernel
# !.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"

# print("Done. Restart the kernel before proceeding.")
# print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")


# ### Run the cell below every time to activate the installed environment. 

# In[1]:


# activate venv after installation. This needs to be run everytime.
get_ipython().system('source ./.venv/bin/activate')


# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# In[2]:


import json
import csv
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/results.jsonl"
CACHE_FILE  = "cache/response_cache.json"
MAX_TOKENS  = 8192
BATCH_SIZE  = 20

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# In[ ]:


# Dataset loading is handled inside run_inference().


# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# In[3]:


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Solve the problem carefully step-by-step. "
    "Keep the reasoning concise and focused. "
    "Double-check important calculations. "
    "Put the final answer inside \\boxed{}."
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Solve the problem carefully and choose the single best answer. "
    "Keep the reasoning concise and focused. "
    "Pay close attention to arithmetic and sign errors. "
    "Output only the final answer inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ## 5. Load Model with vLLM (for general case, vLLM is faster)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
# Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# In[ ]:


# Model loading is handled inside run_inference().


# ## 5. Load Model with Transformers (alternative to vLLM for DataHub)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT4 quantization** via BitsAndBytes.  
# 
# Key parameters:
# - `load_in_4bit` — quantization strategy of INT4

# In[ ]:


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

# llm = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     device_map="auto",
# )


# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# ### Generate with vLLM

# In[ ]:


# # Build prompts for first 5 entries
# prompts = []
# for item in data:
#     system, user = build_prompt(item["question"], item.get("options"))
#     prompt_text = tokenizer.apply_chat_template(
#         [{"role": "system", "content": system},
#          {"role": "user",   "content": user}],
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     prompts.append(prompt_text)

# # Generate
# print(f"Generating responses for {len(prompts)} questions...")
# outputs = llm.generate(prompts, sampling_params=sampling_params)

# responses = [out.outputs[0].text.strip() for out in outputs]

# # Preview first 3
# for i in range(min(3, len(responses))):
#     print(f"\n── Response {i} (id={data[i].get('id')}) ──")
#     print(responses[i][:400], "..." if len(responses[i]) > 400 else "")


# In[4]:


def run_inference(
    data_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    cache_file=CACHE_FILE,
    save_eval=False,
):
    """Full ML inference pipeline: load data → load model → generate → save outputs."""

    # 1. Load dataset
    data = [json.loads(line) for line in open(data_path)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # 2. Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        trust_remote_code=True,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    print("Model loaded.")

    # 3. Generate responses (with incremental cache)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached responses")
    else:
        cache = {}

    pending = [item for item in data if str(item["id"]) not in cache]
    print(f"{len(pending)} items remaining, {len(data) - len(pending)} already cached")

    prompts = []
    for item in pending:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_items   = pending[i:i + BATCH_SIZE]
        print(f"Generating batch {i // BATCH_SIZE + 1} ({i}–{i + len(batch_prompts)})...")
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        for item, out in zip(batch_items, outputs):
            cache[str(item["id"])] = out.outputs[0].text.strip()
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        print(f"  Saved cache ({len(cache)} total entries)")

    responses = [cache[str(item["id"])] for item in data]

    # 4. Process results
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Processing"):
        is_mcq    = bool(item.get("options"))
        extracted = extract_letter(response) if is_mcq else None
        results.append({
            "id":        item.get("id"),
            "is_mcq":    is_mcq,
            "extracted": extracted,
            "response":  response,
            "correct":   None,
        })
    print(f"Processing complete: {len(results)} results")

    # 5. Save JSONL
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            if save_eval:
                record.update({"gold": r.get("gold"), "correct": r["correct"]})
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(results)} records → {out_path}")

    # 6. Save CSV
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])
    print(f"Saved CSV → {csv_path}")

    return results


# ### Generate with Transformers (for Datahub)

# In[ ]:


# # Build prompts for first 5 entries
# prompts = []
# for item in data[:5]:
#     system, user = build_prompt(item["question"], item.get("options"))
#     prompt_text = tokenizer.apply_chat_template(
#         [{"role": "system", "content": system},
#          {"role": "user",   "content": user}],
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     prompts.append(prompt_text)

# # Tokenize (padded batch)
# print(f"Generating responses for {len(prompts)} questions...")
# inputs = tokenizer(
#     prompts,
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
#     max_length=16384,
# ).to(llm.device)

# # Generate
# with torch.no_grad():
#     output_ids = llm.generate(
#         **inputs,
#         max_new_tokens=MAX_TOKENS,
#         temperature=0.6,
#         top_p=0.95,
#         top_k=20,
#         repetition_penalty=1.0,
#         do_sample=True,
#     )

# # Decode only the new tokens (strip the prompt)
# responses = []
# for i, out in enumerate(output_ids):
#     new_tokens = out[inputs["input_ids"].shape[1]:]
#     responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

# # Preview first 3
# for i in range(min(3, len(responses))):
#     print(f"\n── Response {i} (id={data[i].get('id')}) ──")
#     print(responses[i][:400], "..." if len(responses[i]) > 400 else "")


# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# In[ ]:


# Response processing is handled inside run_inference().


# ## 8. Summary
# 
# Print accuracy broken down by question type.

# In[ ]:


# Summary is printed inside run_inference().


# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# In[ ]:


# JSONL output is saved inside run_inference().


# In[5]:


results = run_inference()


# ## Next Steps
# 
# This notebook gives you a working baseline. Here are directions to improve your score:
# 
# - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
# - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
# - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
# 
# Good luck!
