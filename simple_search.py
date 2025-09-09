from dotenv import load_dotenv
from openai import OpenAI
import csv
import os
from datetime import datetime
import json

from urllib.parse import urlparse

def pretty_print_with_sources(choice):
    """
    choice: one element of completion.choices (i.e. completion.choices[0])
    Returns a Markdown string with the assistant content + a 'Quellen' list.
    """

    # Get the message content (works for both pydantic objects and dicts)
    msg = getattr(choice, "message", choice.get("message") if isinstance(choice, dict) else choice)
    content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else "")
    annotations = getattr(msg, "annotations", msg.get("annotations") if isinstance(msg, dict) else []) or []

    # Normalize annotations to simple dicts we care about
    norm = []
    for ann in annotations:
        # attribute-style or dict-style access
        ann_type = getattr(ann, "type", ann.get("type") if isinstance(ann, dict) else None)
        if ann_type != "url_citation":
            continue
        url_citation = getattr(ann, "url_citation", ann.get("url_citation") if isinstance(ann, dict) else None)
        if not url_citation:
            continue
        url = getattr(url_citation, "url", url_citation.get("url") if isinstance(url_citation, dict) else None)
        title = getattr(url_citation, "title", url_citation.get("title") if isinstance(url_citation, dict) else None)
        if url:
            norm.append({"title": title or url, "url": url})

    # Deduplicate by URL but preserve order
    seen = set()
    dedup = []
    for item in norm:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        dedup.append(item)

    # Determine how many numeric markers appear in the text: [1], [2], ...
    import re
    markers = sorted({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", content)})

    # Build index → source mapping.
    # Strategy: map 1..N in order to the first N deduped citations.
    N = min(len(markers), len(dedup)) if markers else len(dedup)
    index_to_source = {i+1: dedup[i] for i in range(N)}

    # Render the “Quellen” block (Markdown)
    if not index_to_source:
        return content  # nothing to add

    lines = ["", "### Quellen"]
    for i in range(1, N+1):
        src = index_to_source[i]
        host = urlparse(src["url"]).netloc or ""
        # Example: [1] Title (host) – URL
        lines.append(f"- **[{i}]** {src['title']} ({host}) – {src['url']}")

    # If there are more annotations than markers, append the extras too
    if len(dedup) > N:
        lines.append("- –")
        for extra in dedup[N:]:
            host = urlparse(extra["url"]).netloc or ""
            lines.append(f"- {extra['title']} ({host}) – {extra['url']}")

    return content + "\n" + "\n".join(lines)


load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

#model = "openai/gpt-4o-search-preview"
#model_str = "gpt-4o-search-preview"
model = "perplexity/sonar"
model_str = "sonar"
#model = "openai/gpt-5"
#model_str = "gpt5"
results_dir = "results/simple_search"
start_index = 1
limit = 204
files_written = []

# Open input CSV
with open("open-data-benchmark/de-questions.csv", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    # Skip rows up to start_index - 1
    for _ in range(start_index - 1):
            next(reader, None)

    for i, row in enumerate(reader, start=start_index):
        if limit is not None and (i - start_index + 1) > limit:
            break

        thread_id = f"{model_str}-{i:04d}"
        question_id = i
        question = row.get("frage", "") or ""
        answer = row.get("antwort", "") or ""
        question_type = row.get("frage_typ", "") or ""
        source = row.get("datengrundlage", "") or ""
        remark = row.get("bemerkungen", "") or ""

        completion = client.chat.completions.create(

            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Open Data QA agent for Germany. "
                               "You receive questions and should provide answers "
                               "with the corresponding sources. "
                               "When possible, search GovData and other suitable "
                               "German Open Data portals or publicly available information on administrative data."
                               #"Do not use online search. 
                                "Do not ask clarifying questions; simply presume the most likely information need given the user’s question."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            #extra_body={
            #    "search_domain_filter": ["*.de"]
            #}
        )
        print(completion.choices[0].message)
        #llm_text = completion.choices[0].message.content + " "+ completion.choices[0].search_results
        llm_text = pretty_print_with_sources(completion.choices[0])

        # Build JSON object for this row
        data = {
            "thread_id": thread_id,
            "run_at": datetime.utcnow().isoformat() + "Z",
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "source": source,
            "remark": remark,
            "llm_final": llm_text
        }

        # Write each row as its own JSON file, e.g., results/qa-0001.json
        out_path = os.path.join(results_dir, f"{thread_id}.json")
        with open(out_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        files_written.append(out_path)
        print(f"Wrote {out_path}")

print(f"Done. Wrote {len(files_written)} JSON files to {results_dir}.")


