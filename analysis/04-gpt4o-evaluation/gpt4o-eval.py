
import json
from tqdm import tqdm
import argparse
import numpy as np
import re
import os



# def Check_hallucination(key, value, gold_list, hallucination_num):
#     if (value[0] == 'Correct') and ('Direct Match' in value[1]) and (key not in gold_list):
#         hallucination_num = hallucination_num + 1
#     return hallucination_num

def Check_hallucination(key, value, gold_list, hallucination_num):
    if (value[0] == 'Correct') and ('Direct Match' not in value[1]) and (key in gold_list):
        hallucination_num = hallucination_num + 1
    return hallucination_num

def Check_missing(key, value, gold_list, missing_num):
    if (value[0] != 'Correct') and (key in gold_list):
        missing_num = missing_num + 1
    return missing_num

parser = argparse.ArgumentParser()

# parser.add_argument("--prompt_file", default="FmLAMA/analysis/04-gpt4o-evaluation/prompt.txt", type=str)
parser.add_argument("--save_dir", default="FmLAMA/analysis/04-gpt4o-evaluation/output_file", type=str)

parser.add_argument("--data_dir", default="FmLAMA/data", type=str)
parser.add_argument("--model", default="gpt-4o", type=str,)
parser.add_argument('--type', nargs='+', default=['Direct Match', 'Substitutability', 'Missing Traditional Ingredient'], help='A list of items')
args = parser.parse_args()

with open(os.path.join(args.data_dir, 'English_human_evaluation.json'), 'r', encoding='utf-8') as file:
    dataset = json.load(file)

all_results = []

country_results = {}

wrong_number = 0
hallucination_num = 0
missing_num = 0
direct_match_num = 0
missing_traditional_num = 0
substitutability_num = 0
others_num = 0
key_in_not_correct_num = 0
key_not_in_correct_num = 0 # new golden match

di_new_num = 0

for country, data_list in dataset.items():
    print(country)
    country_results[country] = []
    

    with open(os.path.join(args.save_dir, f"{country}_output.json"), 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    
    for data in tqdm(dataset):
        correct_number = 0
        dirct_match = list(set(data['Gold_ingredients']) & set(data['Predicted_ingredient']))
        di_new_num = di_new_num + len(dirct_match)
        for d_m in data['direct_match']:
            data['evaluated_result'][d_m] = ("Correct", "Direct Match")

        
        if len(data['evaluated_result']) != len(data['Predicted_ingredient']):
            evaluated_output = eval(data['evaluated_output'][9:-3])
        else:
            evaluated_output = {}
        
        # merged_dict = {**data['evaluated_result'], **evaluated_output}
        for i, value in data['evaluated_result'].items():
            evaluated_output[i] = value
        for key, value in evaluated_output.items():
            


            if key in data['Predicted_ingredient']:
                if value[0] == 'Correct':
                    if 'Direct Match' in value[1]:
                        direct_match_num = direct_match_num + 1
                        if key not in data['Gold_ingredients']:
                            key_not_in_correct_num = key_not_in_correct_num + 1

                    elif 'Substitutability' in value[1]:
                        substitutability_num = substitutability_num + 1
                    elif 'Missing Traditional Ingredient' in value[1]:
                        missing_traditional_num = missing_traditional_num + 1
                    else:
                        others_num = others_num + 1
                else:
                    if key in data['Gold_ingredients']:
                        key_in_not_correct_num = key_in_not_correct_num + 1

                        

                hallucination_num=Check_hallucination(key, value, data['Gold_ingredients'], hallucination_num)
                missing_num=Check_hallucination(key, value, data['Gold_ingredients'], missing_num)


                if value[0] == 'Correct':
                    for ty in args.type:
                        if ty in value[1]:
                            correct_number = correct_number + 1

                # if value[0] == 'Correct' and value[1] in ['Direct Match', 'Substitutability']:# - Dish-Specific
                # if value[0] == 'Correct' and value[1] in ['Direct Match', 'Dish-Specific Ingredient']:# - Substitutability
                # if value[0] == 'Correct' and value[1] in ['Direct Match']:# - both
                

                    # if value[1] == 'Direct Match':
                        # correct_number = correct_number + 1
                # if value[0] == 'Maybe':
                #     correct_number = correct_number + 0.5

            else:
                wrong_number = wrong_number + 1
        
        country_results[country].append(correct_number/len(data['Predicted_ingredient']))
        all_results.append(correct_number/len(data['Predicted_ingredient']))


country_results['all'] = all_results

for country, results in country_results.items():
    country_results[country]=round(np.mean(results)* 100, 2)
    
print('a')
