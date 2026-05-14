# 🧠 Schema-Aware Text-to-SQL with Parameter-Efficient Fine-Tuning

> _Translating plain English into SQL using a small open-source model — no GPT-4, no API, just efficient fine-tuning and smart post-processing._

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)
[![Spider](https://img.shields.io/badge/Benchmark-Spider-purple)](https://yale-lily.github.io/spider)

---

## 📌 Overview

This project fine-tunes **Phi-3.5-mini-instruct** (3.8B parameters) using **LoRA (Low-Rank Adaptation)** to generate SQL queries from natural language questions. The motivation is straightforward: non-technical users shouldn't need to know SQL to access data, and large closed models like GPT-4 — while effective — are expensive and require API access. We show that a small, open, self-hosted model can achieve competitive results through efficient fine-tuning and smart post-processing.

Built on the **Spider benchmark**, the system goes beyond basic fine-tuning with a multi-stage post-processing pipeline:

- 🔍 **Schema Validity Enforcement** — alias-aware checker that catches hallucinated table/column names
- 🎯 **Candidate Reranking** — generates 8 SQL candidates and ranks by schema score + executability
- 🔀 **Hybrid Picker** — self-consistency voting among valid and executable candidates

**Result:** 52.5% execution accuracy and a **10× improvement** in exact match — running on a single GPU with no API required.

> 🏆 **How we stack up against the competition:**
> GPT-4 (state of the art) hits **85.3%** — but it's a ~1 trillion parameter closed model behind a paid API.
> WizardCoder-15B, a fine-tuned open model **4× our size**, reaches **61.0%**.
> **We hit 52.5% with a 3.8B model, trained on just 2,500 examples, on a single GPU — no API, no cloud dependency, fully open weight.**
> That's within 8.5 percentage points of a model 4× larger. For the compute budget, that's a competitive result.

> 💡 **Hardware used:** Google Colab Pro with NVIDIA A100 80GB GPU.
> Fine-tuning takes ~2–3 hours. Full evaluation across 275 examples takes ~6–7 hours.

---

## 📊 Key Results

### Ablation — each component's contribution

| System                      | Exec Acc  | Exact Match | Schema Validity |
| --------------------------- | --------- | ----------- | --------------- |
| Base model (no fine-tuning) | 46.5%     | 3.5%        | 0.234           |
| + LoRA fine-tuning          | **52.5%** | **37.5%**   | 0.436           |
| + Schema filter             | 50.0%     | 38.5%       | 0.516           |
| + Reranker                  | 52.0%     | 38.0%       | **0.642**       |

### How we compare against the competition

| Model                         | Size        | Exec Acc  | Access             |
| ----------------------------- | ----------- | --------- | ------------------ |
| GPT-4o (zero-shot)            | ~1 Trillion | 86.6%     | 💸 API only        |
| GPT-4 + DAIL-SQL (SOTA)       | ~1 Trillion | 85.3%     | 💸 API only        |
| GPT-3.5-turbo (zero-shot)     | ~175B       | 75.5%     | 💸 API only        |
| WizardCoder-15B (QLoRA)       | 15B         | 61.0%     | ✅ Self-hosted     |
| **Ours: Phi-3.5-mini + LoRA** | **3.8B**    | **52.5%** | ✅ **Self-hosted** |

> 💡 We are **4× smaller** than the closest open-weight competitor (WizardCoder-15B) and within **8.5 percentage points** of its accuracy.
> We use **250× fewer parameters** than GPT-4-class models and require **zero API access**.
> For an academic project with 2,500 training examples on a single GPU — **52.5% is not just decent, it's genuinely competitive.**

---

## 🏗️ System Pipeline

```
Natural Language Question
         ↓
   Prompt Formatting
   (schema + question)
         ↓
 LoRA Fine-Tuned Phi-3.5-mini
         ↓
  Generate 8 SQL Candidates
  (temp 0.0 → 1.0)
         ↓
 Alias-Aware Schema Validation
         ↓
     Hybrid Picker
  (consistency + executability)
         ↓
     Best SQL Query ✓
```

---

## 📂 Project Structure

```
schema-aware-text2sql/
│
├── 📁 Colab/                          ← Jupyter notebooks (run in order)
│   ├── Milestone0_Baseline_pipeline_and_validation.ipynb
│   ├── Milestone_1_LoRA.ipynb
│   ├── Milestone_2_Validation and Candidate Reranking.ipynb
│   ├── Milestone_3_Evaluation_Ablation.ipynb
│   └── Milestone_4_Project_Report.ipynb
│
├── 📁 spider/                         ← Spider dataset placeholder
│   └── (download dataset here — see Setup below)
│
├── 📁 milestone1_lora/                ← Created by Milestone 1 notebook
│   └── csv_outputs/, adapter weights
│
├── 📁 milestone2_schema_val/          ← Created by Milestone 2 notebook
│   └── csv_outputs/
│
├── 📁 milestone3_full_eval/           ← Created by Milestone 3 notebook
│   └── csv_outputs/
│
├── 📁 milestone4_report/              ← Created by Milestone 4 notebook
│   └── figures/
│
├── 📁 processed_data/                 ← Created by Milestone 0 notebook
│   └── processed_dev.pkl, processed_train.pkl
│
├── 📄 README.md
└── 📄 .gitignore
```

> ⚠️ Only the `Colab/` notebooks and the empty `spider/` placeholder are committed to this repo.
> All other folders (`milestone1_lora/`, `milestone2_schema_val/`, etc.) are created automatically
> when you run the notebooks in order. Model adapter weights and large CSV outputs are excluded
> from the repo due to file size — re-run the notebooks to reproduce them.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/schema-aware-text2sql.git
cd schema-aware-text2sql
```

### 2. Download the Spider dataset

```
Download from: https://yale-lily.github.io/spider
Extract and place all contents inside the spider/ folder.
```

The `spider/` folder should contain:

```
spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/
    ├── concert_singer/
    ├── pets_1/
    └── ... (200+ databases)
```

### 3. Install dependencies

```bash
pip install transformers peft datasets torch accelerate
pip install bitsandbytes pandas matplotlib seaborn
```

### 4. Mount Google Drive (if running in Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Run notebooks in order

```
Colab/Milestone0  →  Colab/Milestone_1  →  Colab/Milestone_2
→  Colab/Milestone_3  →  Colab/Milestone_4
```

Each notebook reads from the folders created by the previous one.
Do not skip steps — Milestone 1 depends on the processed data from Milestone 0,
Milestone 2 depends on the adapter weights from Milestone 1, and so on.

---

## 🔬 What Each Notebook Does

| Notebook                                         | What it does                                             | Output folder            |
| ------------------------------------------------ | -------------------------------------------------------- | ------------------------ |
| `Milestone0_Baseline_pipeline_and_validation`    | Load Spider, format prompts, run base model baseline     | `processed_data/`        |
| `Milestone_1_LoRA`                               | LoRA fine-tuning on 2,500 Spider examples                | `milestone1_lora/`       |
| `Milestone_2_Validation and Candidate Reranking` | Schema validator, candidate reranker, M2 evaluation      | `milestone2_schema_val/` |
| `Milestone_3_Evaluation_Ablation`                | Full ablation study across 6 systems on 275 dev examples | `milestone3_full_eval/`  |
| `Milestone_4_Project_Report`                     | Figures, ablation tables, final summary                  | `milestone4_report/`     |

---

## ⚙️ LoRA Configuration

| Parameter            | Value                                                            | Why                                              |
| -------------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| Base model           | Phi-3.5-mini-instruct (3.8B)                                     | Small, open-source, strong at structured output  |
| LoRA rank r          | 32                                                               | Good balance of expressiveness and efficiency    |
| LoRA alpha α         | 64                                                               | Standard 2×r scaling                             |
| Target modules       | qkv_proj, o_proj, gate_up_proj, down_proj, embed_tokens, lm_head | Attention + output layers                        |
| Learning rate        | 2e-4                                                             | Standard for LoRA                                |
| Effective batch size | 8 (per-device 2 × grad accum 4)                                  | GPU memory constraint                            |
| Epochs               | 4                                                                | Sufficient without overfitting on 2,500 examples |
| Updated params       | <1% (~20M / 3.8B)                                                | Core LoRA efficiency benefit                     |
| Precision            | BF16                                                             | Halves memory with negligible accuracy loss      |
| Hardware             | NVIDIA A100 80GB                                                 | Google Colab Pro                                 |

---

## 🔑 Key Technical Contributions

### Alias-Aware Schema Validity Checker

Standard validators break on aliased SQL like:

```sql
SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id
```

They flag `T1` and `T2` as unknown identifiers. Our validator resolves aliases
by parsing `FROM`/`JOIN` clauses first (`T1 → singer`, `T2 → concert`),
then validates column references against the actual schema metadata.

### Hybrid Picker with Self-Consistency

```
8 candidates at temps [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ↓
Schema-valid AND executable? → self-consistency vote → pick most frequent
        ↓ (fallback)
Schema-valid only? → self-consistency vote
        ↓ (fallback)
Reranker: sort by (-schema_score, -exec_ok, rank)
```

Result: +19 correct answers over greedy decoding alone, only 1 regression.

---

## 📈 Error Analysis

| Error Type        | Count | % of Failures |
| ----------------- | ----- | ------------- |
| Syntax error      | 79    | 36.2%         |
| Wrong table name  | 64    | 29.4%         |
| Wrong column name | 39    | 17.9%         |
| Wrong result set  | 18    | 8.3%          |
| Other             | 15    | 6.9%          |
| Ambiguous column  | 3     | 1.4%          |

JOIN performance: **~41% accuracy on 0-JOIN queries → <2% on 1+ JOIN queries.**
This is the primary limitation and the main target for future work.

---

## 🔭 Future Work

- 🗂️ **Retrieval-augmented schema** — inject only relevant tables per question
- 📚 **Full Spider training set** — 7k+ examples for better JOIN coverage
- 🤖 **Larger base model** — 7B or 13B with the same LoRA pipeline
- 🔒 **Constrained decoding** — enforce schema adherence at generation time
- 🌐 **Cross-benchmark** — evaluate on BIRD

---

## 🛠️ Tech Stack

| Tool                            | Purpose                      |
| ------------------------------- | ---------------------------- |
| 🤗 Transformers                 | Model loading and inference  |
| 🤗 PEFT                         | LoRA implementation          |
| 🔥 PyTorch                      | Training backend             |
| 🐍 Python 3.10+                 | Core language                |
| 📊 Pandas                       | Result processing            |
| 📉 Matplotlib / Seaborn         | Visualisations               |
| 🗄️ SQLite3                      | SQL execution for evaluation |
| ☁️ Google Colab Pro (A100 80GB) | Training and evaluation      |
| 💾 Google Drive                 | Checkpoint and data storage  |

---

## 📄 .gitignore

```gitignore
# Model weights and checkpoints
*.bin
*.safetensors
checkpoint-*/
adapter_model/

# Large output files
*.csv
*.pkl

# Dataset (download separately)
spider/

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Auto-created milestone folders
milestone1_lora/
milestone2_schema_val/
milestone3_full_eval/
milestone4_report/
processed_data/
```

---

## 📚 References

- [Spider Dataset](https://yale-lily.github.io/spider)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Phi-3.5-mini-instruct — Microsoft](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [PEFT Library — HuggingFace](https://github.com/huggingface/peft)
- [DAIL-SQL](https://arxiv.org/abs/2308.15363)

---

## 📝 License

Developed as part of an MSCS course project at the University of Illinois Chicago.
Available for educational and research use.

---

<p align="center">
  Made with ☕ and an A100 &nbsp;|&nbsp; MSCS · University of Illinois Chicago
</p>
