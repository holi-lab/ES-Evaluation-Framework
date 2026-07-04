#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
# data_path = ""
# save_path = ""
data_path = ""
save_path = ""


def require_config(name: str, value):
    if not value:
        raise ValueError(f"Please set `{name}` before running this script.")
    return value

# ===============================
# FEATURE DEFINITIONS
# ===============================
CATEGORICAL = {
    "coping_strategy": [
        "emotion_processing",
        "problem_focused",
        "avoidant",
        "maladaptive_behavior",
    ],
    "utterance_style": [
        "plain",
        "upset",
        "verbose",
        "pleasing",
    ],
}

LEVEL_MAP = {
    "engagement_level": {
        "low": 1,
        "medium": 2,
        "high": 3,
    },
    "resistance_level": {
        "low": 1,
        "medium": 2,
        "high": 3,
    },
    "verbosity_level": {
        "very_short": 1,
        "short": 2,
        "medium": 3,
        "long": 4,
        "very_long": 5,
    },
    "total_dialogue_turns_level": {
        "short": 1,
        "medium": 2,
        "long": 3,
    },
}

# ===============================
# UTIL
# ===============================
def extract_seeker_features(system_content: str) -> dict:
    start, end = system_content.find("<seeker_features>"), system_content.find("</seeker_features>")
    if start == -1 or end == -1:
        return {}

    inner = system_content[start + len("<seeker_features>"): end]
    feats = {}
    for line in inner.split("\n"):
        line = line.strip("- ").strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        feats[k.strip()] = v.strip()
    return feats


def scale(level, min_l, max_l):
    mid = (min_l + max_l) / 2
    return (level - mid) / (max_l - min_l)


def build_feature_vec(feats: dict):
    vec = []

    # categorical
    for k, values in CATEGORICAL.items():
        one = [0.0] * len(values)
        if k in feats and feats[k] in values:
            one[values.index(feats[k])] = 1.0
        vec.extend(one)

    # ordinal / metric
    for k, mapping in LEVEL_MAP.items():
        if k in feats:
            lvl = mapping.get(feats[k], None)
            if lvl is not None:
                vec.append(scale(lvl, min(mapping.values()), max(mapping.values())))
            else:
                vec.append(0.0)
        else:
            vec.append(0.0)

    # numeric level
    if "self_disclosure_level" in feats:
        lvl = int(feats["self_disclosure_level"])
        vec.append(scale(lvl, 1, 4))
    else:
        vec.append(0.0)

    # binary
    vec.append(1.0 if feats.get("user_profanity_flag") == "True" else 0.0)

    return vec

def main():
    require_config("data_path", data_path)
    require_config("save_path", save_path)

    with open(data_path, "r", encoding="utf-8") as f, open(save_path, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="Building feature vectors"):
            item = json.loads(line)
            system_msg = next((m for m in item["messages"] if m["role"] == "system"), None)
            if system_msg is None:
                continue

            feats = extract_seeker_features(system_msg["content"])
            item["feature_emb"] = build_feature_vec(feats)

            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved to:", save_path)
    print("Feature dim =", len(build_feature_vec({})))


if __name__ == "__main__":
    main()
