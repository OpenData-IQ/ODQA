# ODQA Open Data Benchmark

Ein Benchmark zur Bewertung automatischer Frage-Antwort-Systeme auf Basis öffentlich verfügbarer Verwaltungs- und Statistikdaten.

Diese Datei beschreibt die Struktur des Repository und den Aufbau des Benchmark.

___

## 🧭 Übersicht

Das ODQA Open Data Benchmark stellt ein Fragenset aus **200 Fragen** bereit, um automatische Systeme zu testen: Wie gut beantworten sie Fragen über öffentlich verfügbare statistische Daten? Insbesondere über Verwaltungsdaten des deutschen Datenportals [GovData](http://govdata.de).

Das Benchmark orientiert sich am Design des „CRAG - Comprehensive RAG Benchmark“ [^1] und bildet die Diversität, Komplexität und Struktur authentischer Fragen von <span>Bürger*innen</span>, <span>Journalist</span>*innen, <span>Planer</span>*innen und anderen Interessierten ab.

<p align="left">
  <img src="../img/Stadtplanerin2.png" alt="Anfrage einer Stadtplanerin über Bevölkerungszuwachs" width="23%"/>
  <img src="../img/Journalist2.png" alt="Anfrage eines Journalisten über Biomüll" width="23%"/>
  <img src="../img/Eltern2.png" alt="Anfrage von Eltern über Krippenplätze" width="23%"/>
  <img src="../img/Antwort3.png" alt="anstehende Antwort des Systems" width="23%"/>
</p>  


[^1]: Yang, Xiao, et al. „CRAG - Comprehensive RAG Benchmark“. arXiv:2406.04744v1 (2024).

___

## 📂 Repository Struktur

- **`open-data-benchmark/`**  
  - `benchmarks.csv`: 121 selbst erstellte Fragen
  - `cleaned_questions_dataset.csv`: überarbeitete Version des Fragenset inklusive generierter Fragen (204?) und Quellenangaben
  - `de-questions.csv`: hat 200 Fragen
  - `en-questions.csv`: übersetzte Version
  - `generate_questions.ipynb`: Skript zur Generierung weiterer Fragen
  - `sources.csv`: Quellen der Antworten inklusive URL
  - `sources_raw.csv`: Quellen der Antworten ohne URL
 
- **`/daten/`**: Quelldateien in `.csv`, `.xml`, `.html` und `.json`
- **`/govdata-catalog/`**: Katalog der Daten
- **`/govdata-sparql/`**: sparql-Anfrage, um geeignete Dateien in *govdata.de* anzuzeigen
- **`/metadaten/`**: Metadaten von *govdata.de*


---


## ❓ Das Fragenset

Die Dateien `de-questions.csv` und `en-questions.csv` enthalten jeweils **200 Fragen** verschiedener Schwierigkeit. Die Fragen orientieren sich an den Themenbereichen der auf *govdata.de* verfügbaren offenen Verwaltungsdaten. 
Sie bilden typische reale Anfragen von Bürger*innen ab.

### 🔎 Enthaltene Felder

Jede Zeile der CSV-Datei enthält eine Frage mit folgenden Informationen:

-	**frage_id**: eindeutige ID
-	**frage**: die gestellte Frage
-	**antwort**: die zu erwartende richtige Antwort
-	**frage_typ**: Klassifikation nach Komplexität
-	**datengrundlage**: Die Fragen werden nach zwei Datengrundlagen (task types) unterschieden
-	**bemerkungen**: weitere Informationen, die bei der Erstellung der Fragen aufgefallen sind und Hinweise auf die Beantwortung der Fragen geben könnten

*Beispiel:*

| **frage_id**   |**frage**      |**antwort** |**frage_typ** |**datengrundlage** |**bemerkungen** |
|---|-------------|-------------|----|-|-------------|
| 1 | Wo finde ich Daten zu Spielgeräten auf Dortmunder Spielplätzen? | https://<span>open-data.dortmund.</span>de/api/v2/catalog/datasets/fb63-spielgeraete | simple | 1 | ja bzw Link zu Datei |
| 104 | In welchem Monat wurde 2025 bisher die meiste Margarine hergestellt? | im März | post processing heavy | 2 | 26.028t Margarine|
    

## 🧩 Fragetypen (frage_typ)
Die acht Fragetypen sind an das CRAG-Design angelehnt:

- **Simple**: einfache Fragen, die eine einfache allgemeingültige Antwort haben

- **Simple with restriction/condition**: einfache Fragen mit der Einschränkung eines Datums oder eines Ortes

- **Set**: Die Antwort ist eine Aufzählung mehrerer Elemente

  - *Beispiel:* Welche Defibrillatoren in Oldenburg sind durchgängig erreichbar?

- **Comparison**: Es werden mehrere Daten oder Datensätze miteinander verglichen

   - *Beispiel:* Wessen Kanalnetz war 2022 das längere? Rotenburg (Wümme) oder Heidekreis? 
    
- **Aggregation**:  Für die Antwort werden mehrere Zahlen zusammen geführt 
   - *Beispiel:* Wie viele Poststationen in der Metropolregion Rhein-Neckar sind uneingeschränkt mit dem Rollstuhl erreichbar? 
     
- **Multi-hop**: Komplexe Fragen, für die mehrere Informationen logisch kombiniert werden müssen
   - *Beispiel:* Wieviel Verstöße hat die Stadt Aachen jeweils im November 2021 und Dezember 2021 bei Geschwindigkeitskontrollen gemessen? In welchem Monat waren es mehr? 

- **Post-processing heavy**: Komplexe Fragen, für die in mehreren Schritten Informationen kombiniert und verarbeitet werden
   - *Beispiel:* Welche drei Vornamen wurden im Jahr 2020 in Kerpen am häufigsten vergeben und wie viele Kinder erhielten jeweils diese Namen? 

- **False Premise**:  Fragen, die eine nicht erfüllbare Bedingung haben

   - *Beispiel:* Wie viele Geburten von Einhörnern wurden 2012 im Standesamt Düsseldorf registriert?

## 🗂️ Datengrundlage (Task Type)
Es gibt zwei Aufgabentypen:
### 1. Data search
Fragen, ob und wo es Daten zu dem gesuchten Thema gibt. Die Antwort ist der Link zu den Daten bzw. eine URL. 
   - *Beispiel:* Gibt es Daten zu Saatkrähen in Soest?

 
### 2. Question Answering 
Fragen, die anhand der Daten inhaltlich beantwortet werden. 
   - *Beispiel:* Wie viele Plätze hat die Kinderkrippe "Biene Maja" in Rostock?
___

## 📝 Fragenerstellung

Die Fragenerstellung fand in zwei Etappen statt. In beiden Etappen wurden die Daten und Metadaten zeitgleich heruntergeladen, sowie die Felder der CSV-Datei `benchmarks.csv` bzw. `cleaned_questions_dataset.csv` ausgefüllt.
1. Zunächst wurden geeignete Daten auf *govdata.de* gesucht. Bedingungen dafür waren die Vollständigkeit der Dateien, das Format in CSV oder XML, und auch die Breite an Themen und Datenbereitstellern. 
Um das Finden solcher passenden Daten zu erleichtern, kann die `/govdata-sparql/`-Anfrage auf der Seite von *govdata.de* verwendet werden.
Je nach Art der Daten wurde dann ein Fragentyp ausgewählt und eine passende und realistische Frage formuliert, deren Antwort eindeutig in den zugeordneten Dateien und/oder Metadaten zu finden ist.

   Daraus sind 121 Fragen entstanden.
2. Im nächsten Schritt sind mit Hilfe des `generate_questions.ipynb`-Skriptes weitere Fragen generiert worden. Das Python-Skript orientiert sich an den bereits formulierten Fragen, kann je nach Fragetyp manuell verstellt werden und lässt auch eine Anpassung des Prompts zu. Eine manuelle Prüfung der Richtigkeit war trotzdem notwendig; so auch das Sicherstellen, ob die Metadaten wirklich im RDF-Format vorliegen.

   Daraus sind 79 Fragen entstanden.

___


## 🚀 Usage

text text
___

## 🆘 Support

text text
___

## 🛣️ Roadmap

text text

___
## 🤝 Contributing

text text
___

## 👩‍💻 Authors and acknowledgment

text text
___

## 📜 License
text text

[MIT](https://choosealicense.com/licenses/mit/)
___

## 📈 Project Status

text text
