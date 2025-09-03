import logging
from typing import Optional, Type, Any
from rdflib import Literal, URIRef
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rdflib import Graph
from urllib.parse import quote
import re
from sparql_tool import SPARQLToolInput
import requests
logging.basicConfig(level=logging.INFO)


def fix_ckan_urls(xml_string):
    # Specifically target the CKAN URLs with spaces
    def fix_ckan_url(match):
        url = match.group(0)
        # Extract the query part and encode it
        if 'q=' in url:
            # Find the query parameter with spaces
            url = re.sub(r'q=([^&"<>]*)', lambda m: f'q={quote(m.group(1))}', url)
        return url

    # Pattern for CKAN API URLs
    pattern = r'https://ckan\.govdata\.de/api/3/action/[^"<>]+'
    return re.sub(pattern, fix_ckan_url, xml_string)

class SearchToolInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="The query to search the catalog"
    )


class SearchTool(BaseTool):
    name: str = "dataset_query"
    description: str = "Retrieve the suitable dataset for the query from the govdata portal."
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query):
        # Base API endpoint
        url = "https://ckan.govdata.de/api/3/action/dcat_catalog_search"
        logging.info(f"[TOOL] Running {self.name} with disease={query}")
        #encoded = quote(query)
        #print(encoded)
        # Query parameters
        params = {
            "q": query,
            "format": "rdf"
        }

        try:
            response = requests.get(url, params=params)

            # Raise exception if request failed (status code != 200)

            # Convert to JSON
            data = response.json()
            #print(data.get("result"))

            # The API itself signals failure
            if not data.get("success", True):
                return {
                    "success": False,
                    "status": "error",
                    "message": data.get("error", "Unknown error"),
                    "data": []
                }

            # Otherwise, parse the actual rows from the API's result
            #count = len(data.get("result"))
            #print(count)
            #if count < 100:
            xml_data = data.get("result")
            cleaned_xml = fix_ckan_urls(xml_data)

            g = Graph()
            g.parse(data=cleaned_xml, format="xml")
            #print(g.print())

            with open("sparql/search.sparql", "r", encoding="utf-8") as f:
                sparql_query = f.read()

            #print(sparql_query)

            #result: dict[str, list[dict[str, Any]]] = {}

            results = g.query(sparql_query)
            length = len(results.bindings)
            # Convert results into a list of dicts
            table = []
            for row in results:
                table.append({
                    "dataset": str(row.dataset),
                    "dataset_title": str(row.dataset_title),
                    "distribution_title": str(row.distribution_title),
                    "distribution_description": str(row.distribution_description),
                    "downloadURL": str(row.downloadURL),
                    "accessURL": str(row.accessURL)
                })

            logging.info(table)
            return {
                    "success": True,
                    "status": f"ok, {length} Treffer",
                    "data": table
            }

            #else:
            #    return {
            #        "success": False,
            #        "status": "error",
            #        "message": "Too many search results",
            #        "data": []
            #    }

        except Exception as e:
            # Network error, JSON parse fail, etc.
            return {
                "success": False,
                "status": "error",
                "message": str(e),
                "data": []
            }


if __name__ == "__main__":
    tool = SearchTool()
    output = tool._run("geschwindgikeitsüberschreitungen aachen 2021")
    print(output)