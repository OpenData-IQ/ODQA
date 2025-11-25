import os
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

OUTPUT_DIR = "scores"
DPI = 220


def radar(ax, types, series_dict: dict[str, list[float]], title: str):
    # Draw radar plot for multiple series
    n = len(types)
    if n == 0:
        return
    angles = np.linspace(0, 2*math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    objects = []

    for name, values in series_dict.items():
        vals = (values + values[:1]) if values else []
        line_object, = ax.plot(angles, vals, linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.15)
        objects.append(line_object)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(types, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20","40","60","80"])
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")

    return objects


def generate_radar(metric: str, models: List[str]):
    question_types = [
        "simple",
        "simple with restriction",
        "multi hop",
        "post processing heavy",
        "set",
        "false premise",
        "aggregation",
        "comparison"
    ]
    print(metric)

    list_of_raw_series = [
        pd.read_csv(Path(OUTPUT_DIR, f"{model}.csv"), index_col="metric")
        for model in models
    ]


    # get the total values as the number of questions
    totals = list_of_raw_series[0].loc["total"].drop("all")

    # build labels like "simple\n(n=22)" or "simple (22)"
    question_types_with_n = [
            f"{qt}\n(N={int(totals[qt])})"
            for qt in question_types
    ]

    series_dict = {
        model_name: df.drop("all", axis=1).loc[metric].tolist()
        for model_name, df in zip(models, list_of_raw_series)
    }

    fig1, ax1 = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))

    if metric == 'completion_rate':
        line_object = radar(
            ax1,
            question_types_with_n,
            series_dict,
            f"{metric.replace('_', ' ').upper()} (%) — by question type"
        )
    else:
        line_object = radar(
            ax1,
            question_types,
            series_dict,
            f"{metric.replace('_', ' ').upper()} (%) — by question type"
        )


    ax1.legend(
        line_object,
        models,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.10)
    )

    fig1.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, f"radar_{metric}.png")
    fig1.savefig(out1, dpi=DPI)
    plt.close(fig1)


def main():
    ap = argparse.ArgumentParser(description="Calculates accuracy scores.")
    ap.add_argument("--metric", type=str, required=True, help="Metric for radar chart")
    ap.add_argument("--models", nargs="*", type=str, default=["gpt5", "gpt5-mini", "gpt5-mini-40"],
                    help="Models for radar chart generation")
    #ap.add_argument("--by_type", type=bool, default=False)
    args = ap.parse_args()
    generate_radar(args.metric, args.models)


if __name__ == "__main__":
    main()