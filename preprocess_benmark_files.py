import pandas as pd
import re
import unicodedata
from rdflib import Graph

questions_df = pd.DataFrame()
questions_df["frage_id"] = None
questions_df["frage"] = None
questions_df["antwort"] = None
questions_df["frage_typ"] = None
questions_df["datengrundlage"] = None
questions_df["bemerkungen"] = None

sources_df = pd.DataFrame()
sources_df["quell_id"] = None
sources_df["frage_id"] = None
sources_df["dateiname"] = None
sources_df["metadaten"] = None

benchmark_dir = "open-data-benchmark"
df = pd.read_csv(f"{benchmark_dir}/cleaned_questions_dataset.csv")
with open("sparql/extract.sparql", "r", encoding="utf-8") as f:
    extract_query = f.read()

for row in df.itertuples(index=True, name="Row"):
    print(row.Index)
    print(row.Index + 1)
    print(row.frage)
    questions_df.loc[len(questions_df)] = [row.Index + 1, row.frage, row.antwort, row.frage_typ,
                                           str(row.datengrundlage), row.datenquelle_schwierigkeit]
    source_part = 1
    metadata_file = ""
    if (row.frage_typ != "multi hop") or (" " not in row.dateiname):
        # Write information to questions_df
        if pd.notna(row.metadaten):
            metadata_file = row.metadaten.replace(" ", "").replace("\u00a0", "")
        sources_df.loc[len(sources_df)] = [f"{row.Index + 1}-{str(source_part)}", row.Index + 1, row.dateiname,
                                           metadata_file]
    else:
        files = row.dateiname.split(" ")
        metadata = row.metadaten.split(" ")
        for idx, file in enumerate(files):
            if idx > 0:
                source_part = source_part + 1
                #print(f"source_part {str(source_part)}")
            if len(metadata) == 1:
                idx_metadata = 0
            else:
                idx_metadata = idx
            metadata_file = metadata[idx_metadata].replace(" ", "").replace("\u00a0", "")
            sources_df.loc[len(sources_df)] = [f"{row.Index + 1}-{str(source_part)}", row.Index + 1, files[idx],
                                               metadata_file]

    if row.datengrundlage == 1.0:
        remark = row.antwort
        metadata_files = row.metadaten
        print(metadata_files)
        if pd.notna(metadata_files):
            datasets = []
            metadata_files = row.metadaten.split(" ")
            for idx, metadata_file in enumerate(metadata_files):
                with open(f"{benchmark_dir}/metadaten/{metadata_file}.rdf", "r", encoding="utf-8") as f:
                    dcat_snippet = f.read()
                    g = Graph()
                    g.parse(data=dcat_snippet, format="xml")
                    # g.print()
                    results = g.query(extract_query)
                    first = next(iter(results), None)
                    if first is not None:
                        datasets.append(str(first["dataset"]))
                # print(results)
            questions_df.at[row.Index, "antwort"] = datasets[0] if len(datasets) == 1 else " ".join(datasets)
        if not pd.isna(row.datenquelle_schwierigkeit):
            questions_df.at[row.Index, "bemerkungen"] = f"{remark} - {row.datenquelle_schwierigkeit}"
        else:
            questions_df.at[row.Index, "bemerkungen"] = remark
    # write questions to questions_df
    # write sources to sources_df
    if row.datengrundlage == 2.0:
        if not pd.isna(row.datenquelle_schwierigkeit):
            questions_df.at[row.Index, "bemerkungen"] = row.datenquelle_schwierigkeit
    questions_df.to_csv(f"{benchmark_dir}/de-questions.csv.", index=False)
    sources_df.to_csv(f"{benchmark_dir}/sources_raw.csv.", index=False)
