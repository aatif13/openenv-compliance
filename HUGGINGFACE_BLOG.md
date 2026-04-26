# Teaching LLMs to audit contracts, invoices, and privacy policies with OpenEnv

**TL;DR**  
We built [**Document Compliance Auditing**](https://huggingface.co/spaces/aakama/openenv-compliance), an [OpenEnv](https://huggingface.co/docs/hub/main/en/spaces) environment where agents read real business-style documents, take structured compliance actions, and receive dense step rewards. A companion [**Google Colab notebook**](https://colab.research.google.com/) (the training notebook in this project: `document(2) (1).ipynb`) shows how to connect to the live Space, collect trajectories, run **Unsloth**-accelerated QLoRA, **TRL** SFT warm-start for JSON formatting, and **TRL GRPO** with custom reward functions—then plot reward and loss curves and compare before/after scores.

---

## Why this problem matters

Enterprises process employment contracts, invoices, and privacy policies every day. Reviews are slow, expensive, and inconsistent. Large language models can *read* these documents, but systematic auditing—field checks, cross-document matching, and regulatory clause coverage with severity—needs **training signal** and **verifiable** metrics. We designed an environment that turns compliance auditing into a well-defined control problem: clear observations, structured actions, and rewards that punish hallucination and reward coverage.

## What the environment does

**Three tasks (easy → hard):**

1. **Employment contract** — confirm required fields or mark them missing.  
2. **Invoice vs purchase order** — flag real discrepancies (e.g. tax, quantity, references).  
3. **GDPR-style privacy policy** — flag missing clauses and assign severities.

The agent sees task metadata, the document (or a truncated view for long texts), what it has already flagged, step limits, and available actions. Actions are JSON objects such as `flag_violation`, `mark_field_present` / `mark_field_missing`, and `submit`. **Rewards are dense** (per step), so training algorithms get feedback often, not only at episode end.

**Beyond single-agent evaluation**, the system also supports **auditor + verifier** flows and **curriculum-style** difficulty in the product surface—so researchers can study oversight and progression, not just one-shot scores.

## How we train (judge-friendly recipe)

We **do not** re-implement a full training stack from scratch. The workflow is:

1. **Point at the live environment**  
   The Space exposes standard HTTP endpoints (OpenEnv-style) for `reset`, `step`, `grader`, and more—see the Space README and Swagger UI on the same deployment.

2. **Collect trajectories**  
   The notebook builds prompts from observations and stores actions with rewards (rule-based or model rollouts) for later training.

3. **SFT warm-start (TRL `SFTTrainer`)**  
   A short phase on *positive* steps teaches **valid JSON** and the action schema so later GRPO rollouts are not garbage.

4. **GRPO (TRL `GRPOTrainer`)**  
   Group-relative policy optimization with multiple reward components—format validity, field validity, and severity heuristics—on top of **Unsloth**-wrapped QLoRA for efficient GPU use (e.g. T4 in Colab).

5. **Evidence**  
   The notebook **plots training rewards** (and related curves) and runs **before/after** evaluation tables by task. Export those images and link them from the project README so judges can verify a real run without downloading large video files (use a public YouTube or short Loom link instead, per competition guidance).

## What a strong submission looks like

- **OpenEnv** as the contract between environment and client—built on `openenv-core` and a clear `openenv.yaml` spec.  
- **A runnable training path** (Colab or local) using **Unsloth** and/or **TRL**—as in our notebook.  
- **Public Space** so evaluators can hit `/health` and `openenv validate <url>` (with the correct CLI version).  
- **A short video or a mini-writeup** (this post, or a README section) with **links only** to external media—**no** huge video binaries in the Hub repo.

## Try it and reproduce

- **Hugging Face Space (environment):** [https://huggingface.co/spaces/aakama/openenv-compliance](https://huggingface.co/spaces/aakama/openenv-compliance)  
- **Source / notebook:** use the training notebook in the repository, open it in Colab, set your Hugging Face token for model download, and run on a **GPU** runtime.  
- **What judges look for (official criteria):** [Google Doc (bookmark)](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit?tab=t.0#bookmark=kix.2dz0x0nie3me)  

## Closing

**Document compliance auditing** is a practical, under-served testbed for document-grounded control and long-context reasoning. If you fine-tune with our trajectories and reward design, you get **measurable** before/after scores and plots—and a story that is easy to explain in under two minutes to a non-specialist. We hope this environment makes it simpler to build and compare agents that do real enterprise work, not only toy games.

*If you build on this Space, consider linking your models, runs, and discussion threads in the community tab so others can find follow-up work easily.*
