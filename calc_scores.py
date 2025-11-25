import os
import json
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from typing import List, Dict, DefaultDict
import pandas as pd
import logging

# 8 question types
QUESTION_TYPES = [
    "simple",
    "simple with restriction",
    "multi hop",
    "post processing heavy",
    "set",
    "false premise",
    "aggregation",
    "comparison",
]

# Initialize a Counter for each question type
total_type_counters = defaultdict(int)

# Category counts with a nested dict
cat_counts = DefaultDict[str, DefaultDict[str, int]]
category_type_counts: cat_counts = defaultdict(lambda: defaultdict(int))

# Problem counts with a nested dict
prob_counts = DefaultDict[str, DefaultDict[str, int]]
problem_type_counts: prob_counts = defaultdict(lambda: defaultdict(int))


def calculate_metrics(total: int, category_counts: Counter, problem_counts: Counter) -> List[float]:
    skipped = problem_counts.get("token limit", 0) + problem_counts.get("recursion limit", 0) + problem_counts.get(
        "no answer", 0)
    success_count = category_counts.get("perfect", 0) + category_counts.get("acceptable", 0)
    success_rate = success_count / total if total else 0
    completion_rate = (total - skipped) / total if total else 0
    conditional_accuracy = success_count / (total - skipped) if total - skipped > 0 else 0
    print("Completion Rate", completion_rate)
    print("Success Rate", success_rate)
    print("Conditional Accuracy", conditional_accuracy)
    return [total, completion_rate, success_rate, conditional_accuracy]


#def calculate_all(path: str, excluded: int, by_type: bool):
def calculate_all(path: str, excluded: int):
    category_counts = Counter()
    problem_counts = Counter()
    total = 0
    for file in os.scandir(path):
        print(file)
        with open(file, 'r', encoding="utf-8") as f:
            json_doc = json.loads(f.read())
            logging.info(f"[INFO] Processing file {f}...")
            if json_doc.get("question_id") not in excluded:
                total += 1
                total_type_counters[json_doc.get("question_type")] += 1
                judgement = json_doc.get("judgement")
                category = judgement.get("category")
                category_counts[category] += 1
                category_type_counts[json_doc.get("question_type")][category] +=1
                if category == "problem_answers":
                    problem = judgement.get("problem_type")
                    problem_counts[problem] += 1
                    problem_type_counts[json_doc.get("question_type")][problem] +=1
    metrics = calculate_metrics(total,category_counts, problem_counts)
    # initialize data of lists.
    data = {'metric': ['total', 'completion_rate', 'success_rate', 'conditional_accuracy'],
            'all': metrics}
    nu_list = data["all"]
    print(f"Completion {nu_list[1]}")
    print(f"Success {nu_list[2]}")
    print(f"Cond {nu_list[3]}")

    for item in QUESTION_TYPES:
        new_metrics = calculate_metrics(total_type_counters[item], category_type_counts[item], problem_type_counts[item])
        data[item] = new_metrics
    df = pd.DataFrame(data)

    # Create scores dir in case it's not there
    scores = "scores"
    if not os.path.exists(scores):
        os.makedirs(scores)
        logging.info(f"[INFO] {scores} directory created.")
    else:
        logging.info(f"[INFO] {scores} already exists.")
        model = path.split("/")[-1].split("human-")[-1]
    df.to_csv(Path(scores, f"{model}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Calculates accuracy scores.")
    ap.add_argument("--eval_dir", type=str, required=True, help="Directory of judged LLM files")
    ap.add_argument("--exclude", nargs="*", type=int, default=[16, 54, 77, 90])
    #ap.add_argument("--by_type", type=bool, default=False)
    args = ap.parse_args()
    calculate_all(args.eval_dir, args.exclude)


if __name__ == "__main__":
    main()