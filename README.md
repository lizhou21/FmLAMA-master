# FmLAMA

```
FmLAMA-master
|- access_LMs
  |- modules
  |- run_prompting_experiment.py
  |- eval_utils.py
  |- model_config.py
  |- utils.py
  |- metrics_utils.py
  |- get_results_MPA.py
|- construct_Data
  |- 01-obtain_country_info.py
  |- 02-obtain_data.py
  |- 03-language_split.py
  |- 04-data_filter.py
  |- template_klg.py
|- data
  |- data_filter
  |- output_filter
  |- Dish_Count.json
  |- Dishes.csv
  |- country_info.json
```

### Step 1: Construct datasets
1. Create Dishes.csv (FmLAMA): `run construct_Data/02-obtain_data.py`    
2. Language split:  `run construct_Data/03-language_split.py`
3. Filter su-dataset that common shared in 6 language: `run construct_Data/04-data_filter.py`

   **Note:** We have provide FmLAMA dataset and the related filter datasets in this repo. You can also use our code for data generation, but due to the dynamic updates in Wikidata, the data created may differ from the version we provide.


> Run probing Llama2 and Vicuna2

'''
cd access_LMs
sh run_llama_vicuna.sh
'''

