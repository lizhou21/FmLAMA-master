from SPARQLWrapper import SPARQLWrapper, JSON
from template_klg import *
import pandas as pd
import json

# aa = test()

countries = get_country()

# candidate countries

# "Japan": 2335,
# "Spain": 1117,
# "United States of America": 2675,
# "Italy": 2975,
# "Turkey": 1206,
# "United Kingdom": 991,
# "People's Republic of China": 1271,
# "France": 2858,
# "India": 1510,




Dish_Count, dishes = get_dish(countries)

dishes = pd.DataFrame(dishes)
with open('/home/nlp/ZL/FmLAMA-master/construct_Data/output_data/Dish_Count.json', 'w') as json_file:
    json.dump(Dish_Count, json_file)
dishes.to_csv('/home/nlp/ZL/FmLAMA-master/construct_Data/output_data/Dishes.csv', index=False)
