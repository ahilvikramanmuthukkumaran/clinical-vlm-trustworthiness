"""Report metrics for the clinical-VLM project (test split, 337 questions).

Reads results/test_baseline.jsonl and results/test_adapter_train.jsonl and writes
results/evaluation_summary.md and results/evaluation_summary.json.
Usage:  python model/scripts/report_metrics.py [results_dir]
Standard library only; no GPU, torch or Hugging Face token needed.
"""
import json, re, collections, string, sys
from pathlib import Path

TAU = 0.8
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

def load(name): return [json.loads(l) for l in open(RES / name, encoding="utf-8") if l.strip()]
NEG = re.compile(r"\b(no|not|without|absent|negative|cannot|none|unremarkable|free of|ruled out|isn't|doesn't|aren't|don't)\b|n't\b")
def pol_strict(t):
    m = re.match(r"^\W*(yes|no)\b", t.strip().lower()); return m.group(1) if m else None
HEDGE = re.compile(r"(difficult|hard|impossible|not possible|unable|cannot|can't|not enough|unclear|inconclusive)\b[^.]{0,40}(determin|confirm|say|assess|tell|evaluat|rule|conclu|identif)")
def pol_lenient(t, q=""):
    p = pol_strict(t)
    if p: return p
    t = t.lower(); q = q.lower()
    if "cannot determine" in t or HEDGE.search(t): return None
    first = " ".join(re.split(r"(?<=[.!?])\s", t.strip())[:2])
    if re.search(r"\b(normal|typical)\b", q):
        if re.search(r"abnormal|hypodense|lesion", first): return "no"
        if re.search(r"normal|typical", first) and not NEG.search(first): return "yes"
    return "no" if NEG.search(first) else "yes"
def norm(s): return " ".join(s.lower().translate(str.maketrans("", "", string.punctuation.replace("-", ""))).replace("-", " ").split())
def f1(a, b):
    a, b = a.split(), b.split()
    if not a or not b: return 0.0
    c = sum((collections.Counter(a) & collections.Counter(b)).values())
    if c == 0: return 0.0
    p, r = c / len(b), c / len(a); return 2 * p * r / (p + r)
def prep(r):
    r = dict(r); g = norm(r["gold_answer"]); t = r["raw_text"]
    r["closed"] = r["question_type"] == "closed"
    r["phrase"] = bool(r.get("abstained_by_phrase"))
    if r["closed"]:
        r["ps"], r["pl"] = pol_strict(t), pol_lenient(t, r["question"])
        r["ok_s"] = r["ps"] == g; r["ok_l"] = r["pl"] == g
    else:
        p = norm(t); r["ps"] = r["pl"] = p
        r["ok_s"] = p == g
        r["ok_l"] = p == g or (g in p and len(g) > 1) or (p in g and len(p) > 2) or f1(g, p) >= 0.5
    return r
def metrics(rows, tau, lenient=True):
    ok = "ok_l" if lenient else "ok_s"; pk = "pl" if lenient else "ps"
    N = len(rows)
    ab = lambda r: r["phrase"] or (tau is not None and r["confidence"] < tau)
    ans = [r for r in rows if not ab(r)]
    cor = sum(r[ok] for r in ans)
    cl = [r for r in rows if r["closed"]]; cla = [r for r in cl if not ab(r)]
    cn = [r for r in rows if r["category"] == "closed_no"]
    op = [r for r in rows if not r["closed"]]; opa = [r for r in op if not ab(r)]
    return dict(n=N, abstain=1 - len(ans) / N, coverage=len(ans) / N, acc_all=cor / N,
        selective_acc=cor / len(ans) if ans else float("nan"), committed_err=(len(ans) - cor) / N,
        closed_acc=sum(r[ok] for r in cla) / len(cl), closed_cov=len(cla) / len(cl),
        closed_selacc=(sum(r[ok] for r in cla) / len(cla)) if cla else float("nan"),
        open_acc=sum(r[ok] for r in opa) / len(op),
        false_yes=sum(1 for r in cn if not ab(r) and r[pk] == "yes") / len(cn), n_closed_no=len(cn))

def main():
    base = [prep(r) for r in load("test_baseline.jsonl")]
    adap = [prep(r) for r in load("test_adapter_train.jsonl")]
    assert len(base) == len(adap) == 337 and {r["id"] for r in base} == {r["id"] for r in adap}
    cfgs = [("Baseline", metrics(base, None)), ("Adapter only", metrics(adap, None)),
            ("Adapter + abstention (tau=%.1f)" % TAU, metrics(adap, TAU))]
    rows = [("Closed-question accuracy (n=165)", "closed_acc"), ("Open-question accuracy (n=172)", "open_acc"),
            ('False "yes" on closed "no" (n=79)', "false_yes"), ("Abstention rate", "abstain"),
            ("Selective accuracy (answered only)", "selective_acc"), ("Committed-error rate", "committed_err"),
            ("Overall accuracy (abstain = wrong)", "acc_all")]
    pct = lambda x: "%.1f%%" % (100 * x)
    md = ["# Evaluation summary (test split, n = 337)", "",
          "Computed from the saved Kaggle predictions in results/*.jsonl (not live inference).", "",
          "| Metric | " + " | ".join(n for n, _ in cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for label, k in rows: md.append("| %s | %s |" % (label, " | ".join(pct(m[k]) for _, m in cfgs)))
    md += ["", "Mean generation time per question: baseline %.2f s, adapter %.2f s." % (
        sum(r["seconds"] for r in base) / len(base), sum(r["seconds"] for r in adap) / len(adap))]
    open(RES / "evaluation_summary.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({n: m for n, m in cfgs}, open(RES / "evaluation_summary.json", "w", encoding="utf-8"), indent=1)
    print("\n".join(md))

if __name__ == "__main__":
    main()