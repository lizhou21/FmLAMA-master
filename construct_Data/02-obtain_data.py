from SPARQLWrapper import SPARQLWrapper, JSON
from template_klg import *
import pandas as pd
import json

countries = get_country()

Dish_Count, dishes = get_dish(countries)

dishes = pd.DataFrame(dishes)
with open('FmLAMA-master/data/Dish_Count.json', 'w') as json_file:
    json.dump(Dish_Count, json_file)
dishes.to_csv('FmLAMA-master/data/Dishes.csv', index=False)
