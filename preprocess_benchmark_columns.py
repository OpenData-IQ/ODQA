import pandas as pd
import os
from rdflib import Graph
from rdflib import URIRef

benchmark_dir = "open-data-benchmark"
df = pd.read_csv(f"{benchmark_dir}/sources_raw.csv")

df["dataset"] = None
# df["distribution"] = None
df["format"] = None

with open("sparql/extract.sparql", "r", encoding="utf-8") as f:
    extract_query = f.read()

for row in df.itertuples(index=True, name="Row"):
    # Save file format
    if pd.notna(row.dateiname):
        name, ext = os.path.splitext(row.dateiname)
        ext = ext.replace(",","").replace('"','')
        # Remove the leading dot if present
        ext = ext.lstrip(".")
        df.at[row.Index, "format"] = ext
        print(row.metadaten)
        if pd.notna(row.metadaten):
            df.at[row.Index, "format"] = ext
            with open(f"{benchmark_dir}/metadaten/{row.metadaten}.rdf", "r", encoding="utf-8") as f:
                dcat_snippet = f.read()
                g = Graph()
                g.parse(data=dcat_snippet, format="xml")
                #g.print()
                results = g.query(extract_query)
                #print(results)
                for result in results:
                    print(str(result.dataset))
                    df.at[row.Index, "dataset"] = str(result.dataset)

df.to_csv(f"{benchmark_dir}/sources.csv.", index=False)






