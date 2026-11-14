# ES Evaluation Framework

This repository contains code for the paper:

**Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers**

Paper: [ACL Anthology](https://aclanthology.org/2026.findings-acl.1146/) | [arXiv](https://arxiv.org/abs/2601.07698)

Former title: **Emotional Support Evaluation Framework via Controllable and Diverse Seeker Simulator**

## Overview

This work introduces a controllable help-seeker simulator for stress-testing emotional support models. The simulator is conditioned on multi-dimensional seeker profiles, including psychological and linguistic features, so that supporter models can be evaluated against diverse help-seeking behaviors rather than only cooperative or homogeneous users.

The core model uses supervised fine-tuning as a backbone and a feature-conditioned MoE/MoLA-style routing mechanism to specialize behavior across different seeker profiles. This repository includes code for training the seeker simulator, preparing MoE/MoLA inputs, and generating multi-turn emotional-support dialogues.

## Repository Structure

- `src/training/`: SFT backbone training and dataset utility scripts.
- `src/moe/`: ordinal seeker-feature vector construction, SFT merge, and MoE/MoLA training scripts.
- `src/simulation/`: multi-turn dialogue generation scripts for GPT and SFT supporter settings.
- `src/annotation/`: prompt-based seeker-feature annotation tools and reusable labeling prompts.

## Main Components

- SFT seeker simulator training
- MoE/MoLA training with ordinal feature-vector-conditioned routing
- Dialogue generation between seeker simulators and supporter models
- Prompt-based annotation of seeker profiles from dialogue data
- Utility scripts for data checks and model preparation

## Configuration

Experiment-specific paths and model identifiers are intentionally left blank in the scripts. Before running the code, set the relevant input data paths, checkpoint paths, output directories, and model IDs for your local environment.

## Prompt-Based Seeker Feature Annotation

`src/annotation/` provides prompt-based annotation for the seeker features used by the framework. It tags full dialogues for coping strategy, engagement, resistance, and utterance style; it additionally tags counselor-seeker pairs for self-disclosure and client reaction. Verbosity, dialogue length, and profanity are computed as rule-based features.

Install the optional annotation dependencies and set an API key:

```bash
pip install -r requirements-annotation.txt
export OPENAI_API_KEY=...
```

Input is JSONL with a `messages` field and an optional `op_summary`. By default, `user` messages are treated as counselor turns and `assistant` messages as seeker turns.

```bash
python src/annotation/tag_features.py \
  --input data.jsonl \
  --output data_tagged.jsonl \
  --model gpt-4o-mini \
  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --embed-in-system
```

The command preserves each record, adds normalized `seeker_features` and pair-level annotations, and—with `--embed-in-system`—writes the `<seeker_features>` block consumed by `src/moe/make_mola_data_ordinal.py`. The reusable templates are under `src/annotation/prompts/`.

The utterance-style template intentionally uses the four labels supported by the current MoLA vectorizer: `plain`, `upset`, `verbose`, and `pleasing`.

## Citation

If you use this repository, code, prompts, processed resources, or any derivatives in your research, you should cite our paper:

```bibtex
@inproceedings{heo-etal-2026-stress,
    title = "Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers",
    author = "Heo, Chaewon  and
      Jin, Cheyon  and
      Jo, Yohan",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1146/",
    doi = "10.18653/v1/2026.findings-acl.1146",
    pages = "22842--22869",
    ISBN = "979-8-89176-395-1",
    abstract = "As emotional support chatbots have recently gained significant traction across both research and industry, a common evaluation strategy has emerged: use help-seeker simulators to interact with supporter chatbots. However, current simulators suffer from two critical limitations: (1) they fail to capture the behavioral diversity of real-world seekers, often portraying them as overly cooperative, and (2) they lack the controllability required to simulate specific seeker profiles. To address these challenges, we present a controllable seeker simulator driven by nine psychological and linguistic features that underpin seeker behavior. Using authentic Reddit conversations, we train our model via a Mixture-of-Experts (MoE) architecture, which effectively differentiates diverse seeker behaviors into specialized parameter subspaces, thereby enhancing fine-grained controllability. Our simulator achieves superior profile adherence and behavioral diversity compared to existing approaches. Furthermore, evaluating 7 prominent supporter models with our system uncovers previously obscured performance degradations. These findings underscore the utility of our framework in providing a more faithful and stress-tested evaluation for emotional support chatbots."
}
```

Any use of the code, prompts, processed resources, or derivatives from this repository should retain appropriate attribution to the original paper and this repository.
