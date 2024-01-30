from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import json
import pandas as pd
# Set up the DBpedia SPARQL endpoint URL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")


def get_country():
    # Write your SPARQL query
    query = """
    SELECT DISTINCT 
?country 
?countryLabel
?continentLabel
(LANG(?countryLabel) AS ?labelLang) 
(STRAFTER(str(?country), "http://www.wikidata.org/entity/") AS ?labelID)

WHERE {
  ?country wdt:P31 wd:Q6256.
  ?country rdfs:label ?countryLabel.
  ?country wdt:P30 ?continent.
  ?continent rdfs:label ?continentLabel.
  FILTER(LANG(?countryLabel) = "en").
  FILTER(LANG(?continentLabel) = "en").
}
    """
    # Set the query and response format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    countries = {}
    for result in results["results"]["bindings"]:
        countryLabel = result["countryLabel"]["value"]
        continentLabel = result["continentLabel"]["value"]
        if countryLabel != 'Taiwan':
            countries[countryLabel] = continentLabel

    # del countries['Taiwan']
    return countries

countries_info = get_country()

json_str = json.dumps(countries_info)
with open('/home/nlp/ZL/FmLAMA-master/data/country_info.json', 'w') as json_file:
    json_file.write(json_str)