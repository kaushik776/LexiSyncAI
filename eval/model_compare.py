"""Compare LLMs (and thinking mode) on the same questions: quality, time and tokens.

    python -m eval.model_compare                         # default setups and questions
    python -m eval.model_compare --questions F04 M01 U03 M06

Each setup unloads the previously used model first (so two models never share the GPU),
warms the new model up with one small call (not timed), then runs the full pipeline.
Run it with the laptop on mains power: on battery the GPU is throttled ~3x.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src import config
from src.indexer import load_index
from src.llm import LLM
from src.pipeline import Pipeline

from eval.answer_eval import key_fact_recall

SETUPS = [   # (label, model, thinking)
    ("qwen2.5:7b (current)", "qwen2.5:7b", "auto"),
    ("qwen3:8b, thinking off", "qwen3:8b", "off"),
    ("qwen3:8b, thinking on", "qwen3:8b", "on"),
]
OUT = config.ROOT / "eval" / "results"


def ollama_cli() -> str | None:
    found = shutil.which("ollama")
    default = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
    return found or (str(default) if default.exists() else None)


def unload_all() -> None:
    cli = ollama_cli()
    if not cli:
        return
    loaded = subprocess.run([cli, "ps"], capture_output=True, text=True).stdout.splitlines()[1:]
    for line in loaded:
        if line.strip():
            subprocess.run([cli, "stop", line.split()[0]], capture_output=True)


def run(question_ids: list[str]) -> dict:
    questions = {q["id"]: q for q in json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))}
    index = load_index("default")
    results = {}
    for label, model, think in SETUPS:
        unload_all()
        config.LLM_MODEL, config.LLM_THINK = model, think
        llm = LLM()
        llm.chat_json("Reply with JSON.", "Say ok.", {"type": "object", "properties": {"ok": {"type": "string"}},
                                                       "required": ["ok"]})          # load the model (not timed)
        pipe = Pipeline(index, llm)
        rows = []
        for qid in question_ids:
            q = questions[qid]
            t = time.time()
            a = pipe.ask(q["question"])
            rows.append({
                "id": qid, "type": q["type"], "badge": a.badge, "abstained": a.abstained,
                "correct_abstention": (q["type"] == "unanswerable") == a.abstained,
                "key_facts": key_fact_recall(a.answer if not a.abstained else "", q.get("key_facts", [])),
                "seconds": round(time.time() - t, 1), "llm_seconds": a.timings.get("llm_s"),
                "llm_calls": a.usage["llm_calls"], "tokens_out": a.usage["completion_tokens"],
                "answer": a.answer,
            })
            r = rows[-1]
            print(f"{label:<24} {qid} [{a.badge:<21}] {r['seconds']:6.1f}s llm {r['llm_seconds']:6.1f}s "
                  f"calls {r['llm_calls']} tokens out {r['tokens_out']:5} kf={r['key_facts']}  {a.answer[:70]!r}",
                  flush=True)
        results[label] = rows
    unload_all()
    return results


def to_markdown(results: dict) -> str:
    lines = ["| Setup | Question | Badge | Key facts | Total s | LLM s | LLM calls | Tokens out |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rows in results.items():
        for r in rows:
            kf = "-" if r["key_facts"] is None else f"{r['key_facts']:.2f}"
            lines.append(f"| {label} | {r['id']} ({r['type']}) | {r['badge']} | {kf} | {r['seconds']} | "
                         f"{r['llm_seconds']} | {r['llm_calls']} | {r['tokens_out']} |")
    lines += ["", "| Setup | Mean total s | Mean tokens out | Correct abstention |", "|---|---|---|---|"]
    for label, rows in results.items():
        n = len(rows)
        lines.append(f"| {label} | {sum(r['seconds'] for r in rows) / n:.1f} | "
                     f"{sum(r['tokens_out'] for r in rows) / n:.0f} | "
                     f"{sum(r['correct_abstention'] for r in rows)}/{n} |")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", default=["F04", "M01", "U03"])
    args = ap.parse_args()
    results = run(args.questions)
    md = to_markdown(results)
    print("\n" + md)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_compare.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "model_compare.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
