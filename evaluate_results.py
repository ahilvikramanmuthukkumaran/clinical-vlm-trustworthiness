import json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINE = RESULTS / "test_baseline.jsonl"
ADAPTER = RESULTS / "test_adapter_train.jsonl"
ABSTAIN_TERMS = ("cannot determine from this image", "cannot determine",
                 "unable to determine", "cannot tell from this image",
                 "unable to tell", "i don't know", "not enough information")

def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f"{path.name}:{n}: {e}")
    return rows

def norm(s):
    s = str(s or "").lower().replace("**", "").replace("__", "").replace("`", "")
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s).split())

def abstained(r):
    if r.get("abstained_by_phrase") is True: return True
    text = norm(r.get("answer", "")) + " " + norm(r.get("raw_text", ""))
    return any(t in text for t in ABSTAIN_TERMS)

def exact(a, b): return norm(a) == norm(b)

def token_f1(a, b):
    x, y = norm(a).split(), norm(b).split()
    if not x or not y: return float(not x and not y)
    overlap = sum((Counter(x) & Counter(y)).values())
    if not overlap: return 0.0
    precision, recall = overlap / len(x), overlap / len(y)
    return 2 * precision * recall / (precision + recall)

def summarize(rows):
    valid = [r for r in rows if not r.get("error") and not r.get("is_stub", False)]
    gold = [r for r in valid if str(r.get("gold_answer", "")).strip()]
    matches = [exact(r.get("answer", ""), r.get("gold_answer", "")) for r in gold]
    f1s = [token_f1(r.get("answer", ""), r.get("gold_answer", "")) for r in gold]
    conf = [r["confidence"] for r in valid if isinstance(r.get("confidence"), (int, float))]
    times = [r["seconds"] for r in valid if isinstance(r.get("seconds"), (int, float))]
    abst = [abstained(r) for r in valid]
    return {
        "rows_total": len(rows), "valid_non_stub_rows": len(valid),
        "error_rows": sum(bool(r.get("error")) for r in rows),
        "stub_rows": sum(bool(r.get("is_stub", False)) for r in rows),
        "rows_with_gold_answer": len(gold),
        "exact_match_count": sum(matches),
        "exact_match_rate": sum(matches) / len(matches) if matches else None,
        "mean_token_f1": sum(f1s) / len(f1s) if f1s else None,
        "abstention_count": sum(abst),
        "abstention_rate": sum(abst) / len(valid) if valid else None,
        "mean_reported_confidence": sum(conf) / len(conf) if conf else None,
        "mean_generation_seconds": sum(times) / len(times) if times else None,
    }

def pct(x): return "N/A" if x is None else f"{x*100:.2f}%"
def num(x): return "N/A" if x is None else f"{x:.4f}"

def main():
    br, ar = read_jsonl(BASELINE), read_jsonl(ADAPTER)
    def index(rows, label):
        d = {}
        for r in rows:
            if r.get("id") in d: print(f"WARNING: duplicate id in {label}: {r.get('id')}")
            d[r.get("id")] = r
        return d
    b, a = index(br, "baseline"), index(ar, "adapter")
    ids = sorted(set(b) & set(a), key=str)
    b_rows, a_rows = [b[i] for i in ids], [a[i] for i in ids]
    bs, ads = summarize(b_rows), summarize(a_rows)
    pairs = []
    for i in ids:
        gold = b[i].get("gold_answer", "")
        pairs.append({
            "id": i, "question": b[i].get("question", ""), "gold_answer": gold,
            "baseline_answer": b[i].get("answer", ""), "adapter_answer": a[i].get("answer", ""),
            "baseline_exact_match": exact(b[i].get("answer", ""), gold),
            "adapter_exact_match": exact(a[i].get("answer", ""), gold),
            "baseline_abstained": abstained(b[i]), "adapter_abstained": abstained(a[i]),
            "baseline_confidence": b[i].get("confidence"), "adapter_confidence": a[i].get("confidence")
        })
    improved = sum(not x["baseline_exact_match"] and x["adapter_exact_match"] for x in pairs)
    regressed = sum(x["baseline_exact_match"] and not x["adapter_exact_match"] for x in pairs)
    delta = None if bs["exact_match_rate"] is None or ads["exact_match_rate"] is None else ads["exact_match_rate"] - bs["exact_match_rate"]
    caveat = ("Exact match and token F1 are lexical reference-answer metrics, not clinically validated accuracy "
              "or a hallucination rate. Abstention is estimated from saved flags/text. Reported confidence is not "
              "necessarily a calibrated probability. This script analyses saved predictions and does not run inference.")
    summary = {
        "evaluation_note": caveat,
        "dataset_alignment": {"baseline_rows": len(br), "adapter_rows": len(ar),
                              "matched_unique_ids": len(ids), "baseline_only_ids": sorted(set(b)-set(a), key=str),
                              "adapter_only_ids": sorted(set(a)-set(b), key=str)},
        "baseline": bs, "adapter_train": ads,
        "paired_comparison": {"adapter_minus_baseline_exact_match_rate": delta,
                              "baseline_wrong_adapter_exact": improved,
                              "baseline_exact_adapter_not": regressed},
        "paired_examples": pairs
    }
    json_path, md_path = RESULTS / "evaluation_summary.json", RESULTS / "evaluation_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Baseline vs. Adapter Evaluation", "", "## Dataset alignment",
        f"- Baseline rows: **{len(br)}**", f"- Adapter rows: **{len(ar)}**",
        f"- Matched unique IDs: **{len(ids)}**",
        f"- IDs only in baseline: **{len(set(b)-set(a))}**",
        f"- IDs only in adapter: **{len(set(a)-set(b))}**", "",
        "## Metrics on matched IDs", "",
        "| Metric | Baseline | Adapter (train prompt) |", "|---|---:|---:|",
        f"| Exact match | {pct(bs['exact_match_rate'])} ({bs['exact_match_count']}/{bs['rows_with_gold_answer']}) | {pct(ads['exact_match_rate'])} ({ads['exact_match_count']}/{ads['rows_with_gold_answer']}) |",
        f"| Mean token F1 | {num(bs['mean_token_f1'])} | {num(ads['mean_token_f1'])} |",
        f"| Abstention rate | {pct(bs['abstention_rate'])} | {pct(ads['abstention_rate'])} |",
        f"| Mean reported confidence | {num(bs['mean_reported_confidence'])} | {num(ads['mean_reported_confidence'])} |",
        f"| Mean generation time (s) | {num(bs['mean_generation_seconds'])} | {num(ads['mean_generation_seconds'])} |",
        "", "## Paired comparison", f"- Adapter minus baseline exact-match rate: **{pct(delta)}**",
        f"- Baseline not exact-match, adapter exact-match: **{improved}**",
        f"- Baseline exact-match, adapter not exact-match: **{regressed}**", "",
        "## Interpretation caveat", caveat, "",
        "Metrics are computed from saved JSONL predictions; no model inference is performed."
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {json_path}\nSaved: {md_path}")

if __name__ == "__main__": main()
