from __future__ import annotations
ARMS = ["cot_token","posthoc_summary","output_only"]

def enrich_items(items, cfg):
    out=[]
    for i,row in enumerate(items):
        r=dict(row)
        r["reward_arm"]=ARMS[i%3]
        r["cheated"]=int(row.get("label",0))
        out.append(r)
    return out

