
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from jinja2 import Environment, BaseLoader
from pydantic.v1 import root_validator
env = Environment(loader=BaseLoader())


def generate_query(disease, filename="sparql/disease_template.sparql"):
#def generate_query(genre, country, filename="sparql/genre_template.sparql"):
    try:
        with open(filename, "r") as file:
            query_template = file.read()
        template = env.from_string(query_template)
        #query = template.render(genre=disease)
        query = template.render(disease=disease)
        return query
    except FileNotFoundError:
        print("The query template file does not exist. Please check the filename and path.")
        return None


class SPARQLToolInput(BaseModel):
    disease: Optional[str] = Field(
        None,
        description="The name of the distribution"
    )
    '''
    genre: Optional[str] = Field(
        None,
        description="The genre of a series"
    )
    country: Optional[str] = Field(
        None,
        description="The country of a series"
    )
    '''

class SPARQLTool(BaseTool):
    #name: str = "series_preference_query"
    #description: str = "Retrieve series from Wikidata based on specific preferences."
    name: str = "disease_query"
    description: str = "Retrieve a disease based on its name."
    args_schema: Type[BaseModel] = SPARQLToolInput

    def _run(self, disease) -> str:
    #def _run(self, genre=None, country=None) -> str:
        try:
            query = generate_query(disease, "sparql/disease_template.sparql")
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            print(query)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if not results:
                return f"No items found for preferences"
            return results
        except Exception as e:
            return f"Error querying items: {str(e)}"
