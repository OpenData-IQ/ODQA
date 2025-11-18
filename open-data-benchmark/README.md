# ODQA Open Data Benchmark

Diese Datei beschreibt die Struktur des Repository und den Aufbau des Benchmark.
___

## Projektbeschreibung

Das ODQA open data benchmark ist ein Fragenset, um automatische Systeme zu testen: Wie gut beantworten sie Fragen über öffentlich verfügbare statistische Daten?

Es orientiert sich am Design des „CRAG - Comprehensive RAG Benchmark“ [^1] bildet die Diversität und Komplexität realistischer Bürger*innenfragen ab.

[^1]: Yang, Xiao, et al. „CRAG - Comprehensive RAG Benchmark“. arXiv:2406.04744v1 (2024).

___

## 📂 Repository Structure

- **`open-data-benchmark/`**  
  - `benchmarks.csv`: 123
  - `cleaned_questions_dataset.csv`: 123
  - `en-questions.csv`, `de-questions.csv`: Question–answer pairs in English and German with task and question type labels  
  - `generate_questions.ipynb`: 123
  - `sources.csv`: 123 
  - `sources_raw.csv`: 123
 
- **`/daten/`**: Daten...
- **`/govdata-catalog/`**: Daten...
- **`/govdata-sparql/`**: Daten...
- **`/metadaten/`**: Daten...


---


## 📄❔❓ Das Fragenset

(`de-questions.csv` auf deutsch,  `en-questions.csv` auf englisch)

Das Fragenset umfasst 202 Fragen unterschiedlicher Schwierigkeit und bezieht sich auf die verschiedenen Themenbereiche der auf govdata.de verfügbaren Verwaltungsdaten. 
Sie ähneln Fragen, die Bürger* innen oder Nutzer* innen in der Realität stellen würden.

Jede Zeile enthält eine Frage mit folgenden Informationen:

-	„**frage_id**“: Die Fragen sind nummeriert.
-	"**frage**“: Die Frage der Bürger*innen, die dem automatischen System gestellt wird.
-	„**antwort**“: die zu erwartende richtige Antwort
-	„**frage_typ**“: Es gibt acht Fragetypen, die die Komplexität echter Bürger*innen-Fragen abdecken:
    - *Simple*: einfache Fragen, die eine einfache allgemeingültige Antwort haben
    - *Simple with restriction/condition*: einfache Fragen mit der Einschränkung eines Datums oder eines Ortes

    - *Set*: Die Antwort ist eine Aufzählung mehrerer Dinge. 

      >Frage 14 - Welche Defibrillatoren in Oldenburg sind durchgängig erreichbar? – "Johanniter-Unfall-Hilfe e. V., Bäcker Bruno, Stadt Oldenburg Zentraler Außendienst und City Wache, Dorfgemeinschaft Bunker Club Bornhorst e. V."
    - *Comparison*: Es werden mehrere Daten oder Datensätze miteinander verglichen. 

      > Frage 73 - Wessen Kanalnetz war 2022 das längere? Rotenburg (Wümme) oder Heidekreis? – "Rotenburg (Wümme)"
    
    - *Aggregation*:  Für die Antwort werden mehrere Zahlen zusammen gerechnet. 

      >Frage 99: Wie viele Poststationen in der Metropolregion Rhein-Neckar sind uneingeschränkt mit dem Rollstuhl erreichbar? – "80"
     
    - *Multi-hop*: Komplexe Fragen, für die mehrere Informationen kombiniert werden müssen. 

      >Frage 157: Wieviel Verstöße hat die Stadt Aachen jeweils im November 2021 und Dezember 2021 bei Geschwindigkeitskontrollen gemessen? In welchem Monat waren es mehr? – „Im November 2021 wurden 3298 Verstöße verzeichnet, wohingegen es im Dezember nur 2800 waren.“

    - *Post-processing heavy*: Komplexe Fragen, die mehrere Informationen kombinieren und verarbeiten müssen. 

      >Frage 171: Welche drei Vornamen wurden im Jahr 2020 in Kerpen am häufigsten vergeben und wie viele Kinder erhielten jeweils diese Namen? – „Die drei am häufigsten vergebenen Vornamen in Kerpen im Jahr 2020 waren Sophie (14 Kinder), Marie (9 Kinder) und Maximilian (8 Kinder)“

    - *False Premise*:  Fragen, die eine nicht erfüllbare Bedingung haben. 

      >Frage 179: Wie viele Geburten von Einhörnern wurden 2012 im Standesamt Düsseldorf registriert? – „Es wurden keine Geburten von Einhörnern im Standesamt Düsseldorf registriert, da Einhörner Fabelwesen sind und nicht in offiziellen Geburtenregistern erfasst werden.“

-	„**datengrundlage**“: Die Fragen werden nach zwei Datengrundlagen (task types) unterschieden:

      - *Datengrundlage 1*: Data search: Fragen, ob und wo es Daten zu dem gesuchten Thema gibt. Die Antwort ist der Link zu den Daten. 

        >Frage 110: Gibt es Daten zu Saatkrähen in Soest? –"https://opendata.soest.de/dataset/ba6457dc-aceb-435f-a8e5-d12bb55ab27b"

    - *Datengrundlage 2*: question answering: Fragen, die inhaltlich beantwortet werden. 

      >Frage 3: Wie viele Plätze hat die Kinderkrippe "Biene Maja" in Rostock? – "66"

-	„**bemerkungen**“: weitere Informationen, die bei der Erstellung der Fragen aufgefallen sind und Hinweise auf die Beantwortung der Fragen geben könnten.


___

## Usage

text text
___

## Support

text text
___

## Roadmap

text text

___
## Contributing

text text
___

## Authors and acknowledgment

text text
___

## License
text text

[MIT](https://choosealicense.com/licenses/mit/)
___

## Project Status

text text
