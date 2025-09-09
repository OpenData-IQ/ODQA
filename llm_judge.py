#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch LLM-as-Judge over many files + metrics.

Input dir (e.g., results/):
- *.json  -> each file is a single record with fields:
    thread_id, run_at, question_id, question, answer, question_type, source, remark, llm_final
- *.jsonl -> each line is one such record

Outputs:
- evaluations/evaluation_<thread_id>.json  (judged item with `judgement`)
- evaluations/summary.json                 (metrics + counts per category)
- evaluations/summary.csv                  (flat summary row for spreadsheets)

Metrics:
- overall_accuracy = (perfect + acceptable) / total
- found_accuracy   = (perfect + acceptable) / found_total
  where found_total excludes items judged as {"category":"problem_answers","problem_type":"not found"}.

Requires:
- `openai` (Chat Completions API; adjust if you use the new responses API)
- `langfuse` (set up your SDK env vars)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe, get_client, Langfuse
import openai
from collections import Counter
import csv
import os
from langchain_openai import ChatOpenAI
langfuse = get_client()
load_dotenv()
import re

# ---------- Prompt template (uses your exact fields) ----------
PROMPT_TEMPLATE = """
You are an impartial LLM-as-Judge. Evaluate a single record (input json) describing a question/answer pair. 
Determine the question from the field "question" in the input json document and 
determine the gold standard answer from the field "answer"  
and assign EXACTLY ONE category for the llm answer in the field "llm_final" from:

- "perfect": The answer is sound and extremely complete.
- "acceptable": Correct but could use minor extra explanation or detail.
- "partially": Contains some correct aspects, but is not entirely correct or is significantly incomplete.
- "problem_answers": Use ONLY if one of these applies, and set problem_type accordingly:
    - "not found": dataset(s) were not found.
    - "token limit": token/context length limit reached.
    - "recursion limit": recursion depth/limit reached.
    - "answer incorrect": the answer is wrong.
    - "no answer": no answer given. agent likely timed out.

Decision rubric (apply in order):
1) If the content clearly indicates token/context limit, recursion limit, explicit wrongness, or that data was not found → category="problem_answers" with the right problem_type.
2) Else if the answer includes the primary resource (e.g., exact URL from `answer`) OR is clearly actionable and comprehensive (multiple relevant links/resources + concise guidance) → "perfect".
3) Else if the answer is correct and useful but a bit thin on detail or explanation → "acceptable".
4) Else if there are some correct signals but it’s vague, incomplete, or partially off → "partially".

Language: If the record is in German, write the rationale in German; otherwise match the input language.

INPUT (STRICT JSON):
{input_json}

RESPONSE FORMAT (STRICT JSON, no extra text):
{{
  "thread_id": "<copy from input>",
  "run_at": "<copy from input>",
  "question_id": "<copy from input>",
  "question": "<copy from input>",
  "answer": "<copy from input>",
  "question_type": "<copy from input>",
  "source": "<copy from input>",
  "remark": "<copy from input>",
  "llm_final": "<copy from input>",
  "judgement": {{
    "category": "perfect" | "acceptable" | "partially" | "problem_answers",
    "problem_type": "not found" | "token limit" | "recursion limit" | "answer incorrect" | "no answer" | null,
    "rationale": "<1–3 short sentences explaining the decision>"
  }}
}}
Rules:
- Return valid JSON only.
- If category != "problem_answers", set "problem_type" to null.
- Do not invent new fields or categories.
- Do not include markdown or commentary outside the JSON.
"""

# --- optional: toleranter Fallback-Parser für "schmutzigen" Output ---
def _parse_json_loose(s: str):
    # 1) ```json ... ``` extrahieren
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", s, re.S | re.I)
    if m:
        s = m.group(1)
    else:
        # 2) erstes Objekt/Array greifen
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            s = m.group(1)
    # 3) häufige Fehler: trailing commas killen
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)

# ---------- Langfuse-observed single-record judge ----------
@observe()
def judge_record(item: Dict[str, Any], model: str = "gpt-4o-mini", temperature: float = 0.0) -> Dict[str, Any]:
    """
    item must contain:
      thread_id, run_at, question_id, question, answer, question_type, source, remark, llm_final
    Returns parsed JSON dict that echoes the item and adds a "judgement" object.
    """
    langfuse.update_current_trace(
        name=f"Judge '{item.get('thread_id','unknown')}'",
        tags=["ext_eval_pipelines", "llm_as_judge"]
    )

    Langfuse()

    input_json = json.dumps(item, ensure_ascii=False, separators=(",", ": "))
    prompt_body = PROMPT_TEMPLATE.format(input_json=input_json)
    #prompt = PROMPT_TEMPLATE.format(input_json=input_json)

    #resp = openai.chat.completions.create(
    #    model=model,
    #    temperature=temperature,
    #    messages=[{"role": "user", "content": prompt}]
    #)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        # api_key = os.getenv('OPENAI_API_KEY'),
        # model="anthropic/claude-sonnet-4",
        # model = 'google/gemini-2.5-flash',
        model=model,
        # model = 'meta-llama/llama-3.1-70b-instruct',
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
        # optional but recommended by OpenRouter:
    )

    # LCEL-Chain: System zwingt JSON-Only, Parser validiert strikt
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "You are a JSON API. Reply with ONLY a single valid minified JSON object. No markdown, no code fences, no commentary."),
        ("user", "{prompt_body}")
    ])
    parser = JsonOutputParser()
    chain = prompt_tmpl | llm | parser

    try:
        parsed = chain.invoke({"prompt_body": prompt_body})  # -> dict
        return parsed
    except Exception as _:
        # Strikter Parser hat aufgegeben – wir versuchen einen toleranten Fallback:
        try:
            resp_text = llm.invoke(prompt_body).content  # korrekter LangChain-Aufruf
            parsed = _parse_json_loose(resp_text)
            return parsed
        except Exception as e2:
            # Letzter Fallback: dein bisheriges "problem_answers"-Objekt
            return {
                **item,
                "judgement": {
                    "category": "problem_answers",
                    "problem_type": "answer incorrect",
                    "rationale": f"Ungültiges JSON vom Bewertungsmodell: {e2}"
                }
            }


# ---------- File loaders ----------
def iter_records_from_file(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            print(f)
            obj = json.load(f)
            # Allow either a single dict or a list of dicts
            if isinstance(obj, list):
                for rec in obj:
                    yield rec
            else:
                yield obj
    else:
        return  # ignore other extensions

def collect_records(indir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    items = []
    for p in sorted(indir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
            for rec in iter_records_from_file(p):
                items.append((p, rec))
    return items

# ---------- Metrics ----------
def compute_metrics(judged_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns counts & accuracies:
      - total
      - counts per category
      - counts per problem_type (for problem_answers)
      - overall_accuracy = (perfect + acceptable) / total
      - found_total      = total - count(problem_answers with problem_type == 'not found')
      - found_accuracy   = (perfect + acceptable) / found_total (if found_total > 0)
    """
    total = len(judged_items)
    cat_counts = Counter()
    prob_counts = Counter()

    for it in judged_items:
        j = (it.get("judgement") or {})
        cat = j.get("category")
        cat_counts[cat] += 1
        if cat == "problem_answers":
            prob_counts[j.get("problem_type")] += 1

    correct = cat_counts.get("perfect", 0) + cat_counts.get("acceptable", 0)
    overall_accuracy = (correct / total) if total else 0.0

    not_found = prob_counts.get("not found", 0)
    found_total = total - not_found
    found_accuracy = (correct / found_total) if found_total else 0.0

    return {
        "total": total,
        "counts_by_category": dict(cat_counts),
        "problem_type_counts": dict(prob_counts),
        "correct_total": correct,
        "overall_accuracy": overall_accuracy,
        "found_total": found_total,
        "found_accuracy": found_accuracy
    }

# ---------- Batch main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch judge results dir and compute accuracies.")
    ap.add_argument("--indir", type=str, required=True, help="Directory of raw LLM response files (json/jsonl).")
    ap.add_argument("--outdir", type=str, default="evaluations", help="Directory to write judged items & summary.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load all records
    pairs = collect_records(indir)
    if not pairs:
        print(f"No .json or .jsonl records found in: {indir}")
        return

    judged_items = []
    for _, rec in pairs:
        # minimal field guard; you can relax or enforce as needed
        required = ["thread_id","run_at","question_id","question","answer","question_type","source","remark","llm_final"]
        missing = [k for k in required if k not in rec]
        if missing:
            # Write a problem judgement to keep accounting consistent
            judged = {
                **rec,
                "judgement": {
                    "category": "problem_answers",
                    "problem_type": "answer incorrect",
                    "rationale": f"Missing required fields: {', '.join(missing)}"
                }
            }
        else:
            judged = judge_record(rec, model=args.model, temperature=args.temperature)

        # Write per-item file
        tid = judged.get("thread_id", "unknown")
        out_path = outdir / f"evaluation_{tid}.json"
        out_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        judged_items.append(judged)

    # Compute metrics
    metrics = compute_metrics(judged_items)

    # Save JSON summary
    summary_json = outdir / "summary.json"
    summary_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save CSV summary (one row)
    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "total",
            "perfect",
            "acceptable",
            "partially",
            "problem_answers",
            "problem_not_found",
            "problem_token_limit",
            "problem_recursion_limit",
            "problem_answer_incorrect",
            "correct_total",
            "overall_accuracy",
            "found_total",
            "found_accuracy",
        ])
        cat = metrics["counts_by_category"]
        prob = metrics["problem_type_counts"]
        w.writerow([
            metrics["total"],
            cat.get("perfect", 0),
            cat.get("acceptable", 0),
            cat.get("partially", 0),
            cat.get("problem_answers", 0),
            prob.get("not found", 0),
            prob.get("token limit", 0),
            prob.get("recursion limit", 0),
            prob.get("answer incorrect", 0),
            metrics["correct_total"],
            f"{metrics['overall_accuracy']:.4f}",
            metrics["found_total"],
            f"{metrics['found_accuracy']:.4f}",
        ])

    print(f"Wrote {len(judged_items)} evaluations to: {outdir}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV : {summary_csv}")


if __name__ == "__main__":
    main()
