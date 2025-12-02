import logging
from typing import Optional, Type, Any
from rdflib import Literal, URIRef
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rdflib import Graph
from urllib.parse import quote
import re
import requests
logging.basicConfig(level=logging.INFO)


def fix_ckan_urls(xml_string):
    # Find all CKAN URL patterns
    pattern = re.compile(r'https://ckan\.govdata\.de/api/3/action/[^"<>]+')
    result_parts = []
    last_end = 0
    for match in pattern.finditer(xml_string):
        # Reconstruct the XML file, take the part between
        # the last change/pattern and the new one and append it
        # unmodified to the output
        result_parts.append(xml_string[last_end:match.start()])
        # get the string
        url = match.group(0)
        # Match query parameter strings and replace them with a quoted string
        if 'q=' in url:
            url = re.sub(
                r'q=([^&"<>]*)',
                lambda m: f'q={quote(m.group(1))}',
                url
            )
        # Add the fixed URL
        result_parts.append(url)
        last_end = match.end()
    # Add the rest of the string after the last match
    result_parts.append(xml_string[last_end:])
    return "".join(result_parts)


class SearchToolInput(BaseModel):
    #query: Optional[str] = Field(
    query: str = Field(
        #None,
        default="",
        description="The query to search the catalog"
    )


class SearchTool(BaseTool):
    name: str = "dataset_query"
    description: str = "Retrieve the suitable dataset for the query from the govdata portal."
    args_schema: Type[BaseModel] = SearchToolInput
    # The search currently extracts only the first 100 hits from the hydra:PagedCollection;
    # The first 100 hits should be the most relevant.
    # Otherwise context length would get unneccesarily full
    # Besides, the agent is instructed to reformulate the query anyway, when the result list contains
    # more than 30 hits
    def _run(self, query):
        # Base API endpoint
        url = "https://ckan.govdata.de/api/3/action/dcat_catalog_search"
        logging.info(f"[TOOL] Running {self.name} with query={query}")
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
                    "status": f"ok, {length} Matches",
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