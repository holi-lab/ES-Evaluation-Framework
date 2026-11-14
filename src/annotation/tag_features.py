#!/usr/bin/env python3
"""Create normalized seeker features from JSONL dialogues using the bundled prompts."""
import argparse, json, os, re
from collections import Counter
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).parent
LABELS = {
    "coping_strategy": ["problem_focused", "emotion_processing", "avoidant", "maladaptive_behavior"],
    "engagement_level": ["high", "medium", "low"], "resistance_level": ["high", "medium", "low"],
    "utterance_style": ["plain", "upset", "verbose", "pleasing"],
    "self_disclosure_level": ["1", "2", "3", "4"], "client_reaction": ["positive", "neutral", "negative"],
}

def prompts(kind):
    return {f.stem: f.read_text(encoding="utf-8") for f in (HERE / "prompts" / kind).glob("*.txt")}

def parse_label(answer, allowed):
    for value in allowed:
        if re.search(rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])", answer.lower()): return value
    raise ValueError(f"Unexpected annotation {answer!r}; expected one of {allowed}")

def ask(client, model, template, values):
    answer = client.chat.completions.create(model=model, messages=[{"role":"user","content":template}], temperature=0).choices[0].message.content or ""
    return parse_label(answer, values)

def make_turns(messages, client_role, counselor_role):
    return [("Client" if m["role"] == client_role else "Counselor", m["content"])
            for m in messages if m["role"] in (client_role, counselor_role)]

def rule_features(turns, tokenizer_name):
    client_text = [t for s, t in turns if s == "Client"]
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name); lengths = [len(tok.encode(t, add_special_tokens=False)) for t in client_text]
    else: lengths = [len(t.split()) for t in client_text]
    avg = sum(lengths) / max(1, len(lengths))
    verbosity = "very_short" if avg < 15 else "short" if avg < 30 else "medium" if avg < 60 else "long" if avg < 100 else "very_long"
    try:
        from profanity_check import predict; profanity = bool(predict(client_text).sum()) if client_text else False
    except ImportError: profanity = False
    n = len(turns)
    return {"verbosity_level": verbosity, "total_dialogue_turns_level": "long" if n >= 9 else "medium" if n >= 6 else "short", "user_profanity_flag": profanity}

def add_system_block(messages, features):
    block = "<seeker_features>\n" + "\n".join(f"- {k}: {v}" for k,v in features.items()) + "\n</seeker_features>"
    result = [dict(m) for m in messages]; system = next((m for m in result if m["role"] == "system"), None)
    if system: system["content"] = re.sub(r"\n?<seeker_features>.*?</seeker_features>", "", system["content"], flags=re.S).rstrip()+"\n\n"+block
    else: result.insert(0, {"role":"system", "content":block})
    return result

def annotate(item, client, args, dialogue_prompts, pair_prompts):
    turns = make_turns(item["messages"], args.client_role, args.counselor_role)
    if not turns: raise ValueError("No relevant turns; set --client-role / --counselor-role")
    summary = str(item.get(args.op_summary_key, "")); dialogue = "\n".join(f"{s}: {t}" for s,t in turns)
    features = {name: ask(client,args.model,p.replace("{op_summary}",summary).replace("{dialogue}",dialogue),LABELS[name]) for name,p in dialogue_prompts.items()}
    pairs=[]; disclosure=[]; reactions=[]
    for (s1,t1),(s2,t2) in zip(turns,turns[1:]):
        if (s1,s2) != ("Counselor","Client"): continue
        tags={name:ask(client,args.model,p.replace("{op_summary}",summary).replace("{commenter_text}",t1).replace("{client_text}",t2),LABELS[name]) for name,p in pair_prompts.items()}
        pairs.append({"counselor_text":t1,"client_text":t2,"tags":tags}); disclosure.append(int(tags["self_disclosure_level"])); reactions.append(tags["client_reaction"])
    features["self_disclosure_level"] = round(sum(disclosure)/len(disclosure)) if disclosure else 1; features.update(rule_features(turns,args.tokenizer))
    output=dict(item); output.update({"seeker_features":features,"pair_features":pairs,"client_reaction_counts":dict(Counter(reactions))})
    if args.embed_in_system: output["messages"] = add_system_block(item["messages"],features)
    return output

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--model",required=True); p.add_argument("--client-role",default="assistant"); p.add_argument("--counselor-role",default="user"); p.add_argument("--op-summary-key",default="op_summary"); p.add_argument("--tokenizer"); p.add_argument("--embed-in-system",action="store_true"); args=p.parse_args()
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]); dp,pp=prompts("dialogue"),prompts("pair")
    with open(args.input,encoding="utf-8") as src,open(args.output,"w",encoding="utf-8") as dst:
        for line in tqdm(src,desc="Annotating"):
            if line.strip(): dst.write(json.dumps(annotate(json.loads(line),client,args,dp,pp),ensure_ascii=False)+"\n")
if __name__ == "__main__": main()
