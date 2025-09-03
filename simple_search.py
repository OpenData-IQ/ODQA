from dotenv import load_dotenv
from openai import OpenAI
import csv
import os
from datetime import datetime
import json

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

model = "openai/gpt-4o-search-preview"
model_str = "gpt-4o-search-preview"
results_dir = "simple_search"
start_index = 1
limit = 10
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

            model="openai/gpt-4o-search-preview",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Open Data QA agent. "
                               "You receive questions and should provide "
                               "answers with the corresponding source."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        llm_text = completion.choices[0].message.content

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


