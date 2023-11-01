from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import json
import pandas as pd
# Set up the DBpedia SPARQL endpoint URL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def get_country():
    # Write your SPARQL query
    query = """
    SELECT ?country ?countryLabel (LANG(?countryLabel) AS ?labelLang) (STRAFTER(str(?country), "http://www.wikidata.org/entity/") AS ?labelID)
    WHERE {
      ?country wdt:P31 wd:Q6256.
      ?country rdfs:label ?countryLabel.
      FILTER(LANG(?countryLabel) = "en")
    }
    """
    # Set the query and response format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    countries = {}
    for result in results["results"]["bindings"]:
        countryLabel = result["countryLabel"]["value"]
        label_ID = result["labelID"]["value"]
        if countryLabel != 'Taiwan':
            countries[countryLabel] = label_ID

    # del countries['Taiwan']
    return countries




def get_dish(countries):
    DISHES = []
    Dish_Count = {}
    # for country, countryID in tqdm(countries.items()):
    for country, countryID in tqdm(countries.items()):
        print(country)
        query = f"""
        SELECT DISTINCT ?dish 
?dishLabel 
?description
(LANG(?dishLabel) AS ?labelLang)
(GROUP_CONCAT(DISTINCT ?ingredient; separator=", ") AS ?hasPartsID)
(GROUP_CONCAT(DISTINCT ?ingredientLabel; separator=", ") AS ?hasParts)
(GROUP_CONCAT(DISTINCT ?MadeFromMateriaLabel; separator=", ") AS ?MadeFromMateria)
(GROUP_CONCAT(DISTINCT ?image; separator=", ") AS ?imageLable)

WHERE {{
  {{
    ?dish (wdt:P31|wdt:P279)+ wd:Q746549. 
  }}
  {{ 
    {{?dish wdt:P495 wd:{countryID}.}}
    UNION
    {{?dish (wdt:P2012|wdt:P361) [(wdt:P17|wdt:P495) wd:{countryID}].}}
  }}
  FILTER NOT EXISTS {{
    ?dish wdt:P495 ?otherCountry.
    FILTER (?otherCountry != wd:{countryID})
  }}

  ?dish rdfs:label ?dishLabel .
  ?dish wdt:P527 ?ingredient .  
  ?ingredient rdfs:label ?ingredientLabel .
  FILTER(LANG ( ?dishLabel )= LANG(?ingredientLabel) )
  
  OPTIONAL {{
    ?dish schema:description ?description .
    FILTER(LANG(?description) = LANG(?dishLabel))
  }}
  OPTIONAL {{
    ?dish wdt:P18 ?image.
    ?dish wdt:P186 ?Materia.
    ?Materia rdfs:label ?MadeFromMateriaLabel .
    FILTER(LANG(?MadeFromMateriaLabel) = LANG(?dishLabel))
  }}
  
}}
GROUP BY ?dish ?dishLabel ?description ?labelLang

"""
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        Dish_Count[country] = len(results["results"]["bindings"])
        print(Dish_Count[country])

        for result in tqdm(results["results"]["bindings"]):
            dish = {}
            dish["url"] = result["dish"]["value"]
            dish["origin"] = country
            dish["name"] = result["dishLabel"]["value"]
            dish["lang"] = result["labelLang"]["value"]
            dish["hasParts"] = set([i.strip() for i in result["hasParts"]["value"].split(";")])
            if "MadeFromMateria" in result.keys():
                dish["materia"] = set([i.strip() for i in result["MadeFromMateria"]["value"].split(";")])
            else:
                dish["materia"] = None
            if "image" in result.keys():
                dish["image"] = result["image"]["value"]
            else:
                dish["image"] = None
            DISHES.append(dish)
        
    return Dish_Count, DISHES

