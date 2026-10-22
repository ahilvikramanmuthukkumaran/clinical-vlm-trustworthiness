"""Run predict() in mitigated mode over the validation split, resumably.

Writes one JSON object per line to results/val_mitigated.jsonl, flushed and
fsynced after every example, so killing the process mid-run loses at most the
example in flight. Re-running skips ids already present in the output file.

Usage:
    python model/scripts/run_val_mitigated.py
    python model/scripts/run_val_mitigated.py --limit 5          # quick check
    python model/scripts/run_val_mitigated.py --retry-errors     # redo failed rows
    python model/scripts/run_val_mitigated.py --mode baseline    # other mode

Validation split only. This script never reads data/splits/test.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.src.predict import cuda_available, load_config, predict  # noqa: E402

DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "val.json"
DEFAULT_ADAPTER = REPO_ROOT / "model" / "checkpoints" / "medgemma_vqarad_lora"
DEFAULT_OUT = REPO_ROOT / "results" / "val_mitigated.jsonl"

# image_path values in the split JSON are relative to the dataset root (data/),
# not to the splits/ directory the JSON itself lives in.
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="base directory that image_path values resolve against")
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mode", default="mitigated", choices=["mitigated", "baseline"])
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N not-yet-done examples")
    p.add_argument("--retry-errors", action="store_true",
                   help="reprocess rows previously written with an error field")
    p.add_argument("--allow-stub", action="store_true",
                   help="write stub rows when there is no usable GPU. Off by default: "
                        "a silent stub run would fill the output with is_stub=True "
                        "placeholders that look like real predictions")
    return p.parse_args()


def show(path: Path) -> str:
    """Repo-relative path for display, falling back to absolute when outside it."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_done(path: Path, retry_errors: bool) -> set:
    """Ids already written. Tolerates a truncated final line from a hard crash."""
    if not path.exists():
        return set()
    done, dropped = set(), 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1          # partial last line, will be rewritten
                continue
            if retry_errors and row.get("error"):
                continue
            done.add(row.get("id"))
    if dropped:
        print(f"note: ignored {dropped} unparsable line(s) in {path.name} "
              f"(expected after a hard crash; they will be re-run)")
    return done


def main():
    args = parse_args()

    if args.split.name == "test.json":
        raise SystemExit("refusing to run on the test split")

    rows = json.loads(args.split.read_text(encoding="utf-8"))
    if not args.adapter.exists():
        raise SystemExit(f"adapter not found: {args.adapter}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out, args.retry_errors)

    todo = [r for r in rows if r["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    config = load_config()
    print(f"split                : {show(args.split)} ({len(rows)} examples)")
    print(f"data root            : {show(args.data_root)}")
    print(f"adapter              : {show(args.adapter)}")
    print(f"output               : {show(args.out)}")
    print(f"mode                 : {args.mode}")
    print(f"abstention_threshold : {config['abstention_threshold']}")
    print(f"min_yes_no_mass      : {config['min_yes_no_mass']}")
    have_gpu = cuda_available()
    print(f"CUDA available       : {have_gpu}")
    if not have_gpu and not args.allow_stub:
        raise SystemExit(
            "\nNo usable GPU stack, so predict() would return stub answers for every\n"
            "example and this file would look like a completed evaluation.\n"
            "Refusing to write stub rows. Pass --allow-stub only if you genuinely\n"
            "want is_stub=True placeholders.\n"
            "If torch is installed but blocked by Windows Smart App Control\n"
            "(OSError WinError 4551 on torch_python.dll), see model/docs/FIT_TEST.md."
        )
    print(f"already done         : {len(done)}   to do now: {len(todo)}")
    if not todo:
        print("nothing to do")
        return
    print()

    t_start = time.time()
    n_ok = n_err = n_abstained = 0

    # Append mode: every completed example is durable before the next starts.
    with open(args.out, "a", encoding="utf-8") as out_fh:
        for i, row in enumerate(todo, 1):
            image_file = args.data_root / row["image_path"]
            record = {
                "id": row["id"],
                "image_id": row.get("image_id"),
                "image_path": row["image_path"],
                "question": row["question"],
                "question_type": row.get("question_type"),
                "category": row.get("category"),
                "gold_answer": row["answer"],
                "mode": args.mode,
                "adapter": show(args.adapter),
            }

            t0 = time.time()
            try:
                result = predict(image_file, row["question"], mode=args.mode,
                                 adapter_path=str(args.adapter))
                record.update({
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "abstained": result.abstained,
                    "raw_text": result.raw_text,
                    "is_stub": result.is_stub,
                })
                n_ok += 1
                n_abstained += int(result.abstained)
            except Exception as exc:  # noqa: BLE001 - one bad example must not end the run
                record["error"] = f"{type(exc).__name__}: {exc}"
                n_err += 1

            record["seconds"] = round(time.time() - t0, 2)

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            os.fsync(out_fh.fileno())      # survive a power loss, not just a crash

            if "error" in record:
                print(f"[{i}/{len(todo)}] {row['id']}  ERROR  {record['error'][:70]}")
            else:
                print(f"[{i}/{len(todo)}] {row['id']}  conf={record['confidence']:.3f}  "
                      f"abst={record['abstained']!s:<5} "
                      f"gold={row['answer'][:18]!r:<20} "
                      f"ans={record['answer'][:34]!r}")

    elapsed = time.time() - t_start
    print()
    print(f"done: {n_ok} ok, {n_err} errors, {n_abstained} abstained "
          f"({n_abstained / n_ok:.1%} of ok)" if n_ok else f"done: {n_err} errors")
    print(f"elapsed {elapsed/60:.1f} min for {len(todo)} examples "
          f"({elapsed/max(len(todo),1):.1f} s each)")
    print(f"total rows now in {args.out.name}: "
          f"{sum(1 for _ in open(args.out, encoding='utf-8'))}")


if __name__ == "__main__":
    main()
