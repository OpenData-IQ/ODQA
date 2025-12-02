import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe, get_client, Langfuse
import os
from langchain_openai import ChatOpenAI
import re
langfuse = get_client()
load_dotenv()


# Optional JSON-Parser as fallback in case LLM returns invalid output
def _parse_json_loose(s: str):
    # find the json section
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", s, re.S | re.I)
    if m:
        s = m.group(1)
    else:
        # get the first object, array
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            s = m.group(1)
    # remove unnecessary commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)


# Judging of agent or LLM responses
# Langfuse-enabled observations
@observe()
def judge_record(item: Dict[str, Any], prompt_template: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
    langfuse.update_current_trace(
        name=f"Judge '{item.get('thread_id')}'",
        tags=["llm_as_judge"]
    )
    Langfuse()
    input_json = json.dumps(item, ensure_ascii=False, separators=(",", ": "))
    prompt_body = prompt_template.format(input_json=input_json)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    # Chain according to Langchain Expression Language (LCEL)
    # Output is validated with the help of a JSON parser
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "You are a JSON API. Reply ONLY with a single valid JSON object. "
         "No markdown, no code, no commentary."),
        ("user", "{prompt_body}")
    ])
    parser = JsonOutputParser()
    chain = prompt_tmpl | llm | parser

    try:
        parsed = chain.invoke({"prompt_body": prompt_body})
        return parsed
    except Exception as _:
        # Invalid JSON, try a fallback, invoke llm (without parser) and
        # process with custom JSON parser
        try:
            resp_text = llm.invoke(prompt_body).content
            parsed = _parse_json_loose(resp_text)
            return parsed
        # In case that fails again,
        # Unpack the input json item and return an error message to the user
        except Exception as e2:
            return {
                **item,
                "judgement": {
                    "category": "problem_answers",
                    "problem_type": "answer incorrect",
                    "rationale": f"Invalid JSON from LLM-as-judge: {e2}"
                }}


# JSON file loading
def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            print(f)
            yield json.load(f)
    else:
        return


def collect_records(in_dir: Path) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(in_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".json":
            for rec in iter_records(p):
                items.append(rec)
    return items


# Trigger LLM-as-judge pipeline
def main():
    ap = argparse.ArgumentParser(description="Executes an LLM-as-judge pipeline for a directory with JSON results")
    ap.add_argument("--in_dir", type=str, default="test", required=True, help="Directory of raw LLM response files (json).")
    ap.add_argument("--out_dir", type=str, default="test", help="Directory to write judged items.")
    ap.add_argument("--model", type=str, default="openai/gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    # Process CLI commands
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load all records
    pairs = collect_records(in_dir)
    if not pairs:
        print(f"No .json records found in: {in_dir}")
        return
    # open prompt template
    with open("prompts/llm-judge.txt", "r", encoding="utf-8") as prompt_file:
        prompt_template = ChatPromptTemplate(prompt_file)
        for record in pairs:
            judged = judge_record(record, prompt_template=prompt_template, model=args.model, temperature=args.temperature)
            tid = judged.get("thread_id")
            out_path = out_dir / f"evaluation_{tid}.json"
            out_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
