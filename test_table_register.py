import chardet
from table_register import TableRegister
from pathlib import Path
import unittest
import datetime


class TestTableTools(unittest.TestCase):

    def setUp(self):
        self.frames = []
        self.hub = TableRegister()
        #self.files = ["2021_november.csv", "index_toi.csv"]
        files = ["2021_november.csv", "index_toi.csv"]
        path = "open-data-benchmark/daten"
        for idx, file in enumerate(files):
            with open(Path(path, files[idx]), "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            content = raw.decode(detected["encoding"])
            frame_id = self.hub.load_csv_string(content, "clean_download", True)
            self.frames.append(frame_id)

        self.loading_solutions = [
            [{'datum': datetime.date(2021, 11, 2), 'messplatz': 'Eupener Straße', 'fahrtrichtung': 'Jahnplatz',
              'fahrzeuge_gesamt': 310, 'verste': 17}],
            [{'id': 1, 'einrichtung': 'Öffentliche Einrichtung', 'benennung': 'Bezirksamt Lichtenberg - Dienstgebäude',
              'strasse': 'Alt-Friedrichsfelde 60', 'plz': '10315 Berlin', 'benutzung': 'kostenlos',
              'oeffnungszeiten': 'Mo-Fr 08:00-18:00 Uhr', 'barrierefreiheit': 'ja'}]

        ]
        self.agg_metrics = [
            [{"op": "sum", "col": 'fahrzeuge_gesamt', "alias": 'value'}],
            [{"op": "count", "col": None, "alias": 'value'}]
        ]

        self.agg_solutions = [
            "literal,value\n,66334\n",
            "literal,value\n,84\n"
        ]

        self.sorting = [
            [{"col": "fahrzeuge_gesamt", "desc": True}],
            [{"col": "benennung", "desc": False}]
        ]

        self.filter = [
            [{"op": "contains", "col": 'messplatz', "value": 'Prager Ring'},
             {"op": "contains", "col": 'messplatz', "value": 'Trierer Straße'}],
            [{"op": "contains", "col": 'benennung', "value": 'Licht'},
             {"op": "contains", "col": 'benennung', "value": 'Dienst'}]
        ]

        self.sort_and_filter_solutions = [
            'datum,messplatz,fahrtrichtung,fahrzeuge_gesamt,verste\n'
            '2021-11-09,Trierer Straße,Kornelimünster,1910,8\n'
            '2021-11-09,Trierer Straße,Innenstadt,1851,2\n'
            '2021-11-13,Prager Ring,Krefelder Straße,1809,91\n'
            '2021-11-17,Prager Ring,Krefelder Straße,1795,17\n',
            'id,einrichtung,benennung,strasse,plz,benutzung,oeffnungszeiten,barrierefreiheit\n'
            '1,Öffentliche Einrichtung,Bezirksamt Lichtenberg - Dienstgebäude,Alt-Friedrichsfelde 60,10315 Berlin,kostenlos,Mo-Fr 08:00-18:00 Uhr,ja\n'
            '133,Öffentliche Einrichtung,Bezirksamt Lichtenberg - Dienstgebäude,Große-Leege-Str. 103,13055 Berlin,kostenlos,Mo-Fr 08:00-18:00 Uhr,ja\n'
            '148,Öffentliche Einrichtung,Bezirksamt Lichtenberg - Dienstgebäude,Egon-Erwin-Kisch-Str. 106,13059 Berlin,kostenlos,Mo-Fr 08:00-18:00 Uhr,ja\n'
        ]

        self.bool = [
            "OR", "AND"
        ]


    def test_csv_loading(self):
        for idx, frame_id in enumerate(self.frames):
            self.assertEqual(self.loading_solutions[idx], self.hub.preview(frame_id, 1))

    def test_agg_metrics(self):
        for idx, frame_id in enumerate(self.frames):
            self.assertEqual(self.agg_solutions[idx],
                         self.hub.to_csv_string(self.hub.aggregate(frame_id, None, self.agg_metrics[idx])))

    def test_sort_and_filter(self):

        #sorted = self.hub.sort(self.frames[1], self.sorting[1], 10)
        #print(self.hub.to_csv_string(self.hub.filter(sorted, self.filter[1], self.bool[1])))

        for idx, frame_id in enumerate(self.frames):
            sorted = self.hub.sort(self.frames[idx], self.sorting[idx], 10)
            self.assertEqual(self.sort_and_filter_solutions[idx],
                         self.hub.to_csv_string(self.hub.filter(sorted, self.filter[idx], self.bool[idx])))

        #print(self.hub.to_csv_string(sorted))
        #print(self.hub.to_csv_string(self.hub.filter(sorted, filter_metrics, "OR")))


if __name__ == '__main__':
    unittest.main()
