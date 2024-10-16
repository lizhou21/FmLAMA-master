from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
import argparse

import re
import os

parser = argparse.ArgumentParser()

# parser.add_argument("--prompt_file", default="/mntcephfs/lab_data/zhouli/personal/FmLAMA/analysis/04-gpt4o-evaluation/prompt.txt", type=str)
parser.add_argument("--save_dir", default="/mntcephfs/lab_data/zhouli/personal/FmLAMA/analysis/04-gpt4o-evaluation", type=str)

parser.add_argument("--data_dir", default="/mntcephfs/lab_data/zhouli/personal/FmLAMA/data", type=str)
parser.add_argument("--model", default="gpt-4o", type=str,)
args = parser.parse_args()


# if args.model == 'gpt-4o':
#     api_key = "sk-7ldxnC50jJ1tnT1r1aA4F171843a4880B4B5238bE29eC462"
#     api_base = "https://api.ai-gaochao.cn/v1"
#     client = OpenAI(api_key=api_key, base_url=api_base)






# save_file = os.path.join(args.save_dir, f"{args.model}_output.json")

with open(os.path.join(args.save_dir, 'prompt/prompt.txt'), 'r') as files:
    instruction_prompt = files.readlines()
    instruction_prompt = "".join(instruction_prompt)




with open(os.path.join(args.data_dir, 'English_human_evaluation.json'), 'r', encoding='utf-8') as file:
    dataset = json.load(file)



# data_output = {}

erros_count = 0
data_statistic = {}
for country, data_list in dataset.items():
    print(country)
    data_statistic[country] = {}
    data_output = []
    save_file = os.path.join(args.save_dir, f"ingredient_statistics.json")

    data_statistic[country]['len'] = len(data_list)
    diff_num = 0
    all_num = 0
    for data in tqdm(data_list):
        d = {}
        evaluation_need = []
        d['Dish'] = data['dish']
        d['Gold_ingredients'] = data['gold_ingredient']
        d['Predicted_ingredient'] = data['predicted_ingredient']
        all_num = all_num + len(data['predicted_ingredient'])
        d['evaluated_result'] = {}
        # for i in d['Predicted_ingredient']:
        #     if i in d['Gold_ingredients']:
        #         d['evaluated_result'][i] = ('Correct, Direct Match')
        #     else:
        #         evaluation_need.append(i)

        dish_info = 'Dish: ' + d['Dish']
        gold_info = 'Reference ingredient label: ' + ', '.join(d['Gold_ingredients'])
        predict_info = 'Predicted ingredients: ' + ', '.join(d['Predicted_ingredient'])
        final_promts = instruction_prompt + "\n\n" + "### Input:\n"+'\n'.join([dish_info, gold_info, predict_info])
        dirct_match = list(set(d['Gold_ingredients']) & set(d['Predicted_ingredient']))
        d['direct_match'] = dirct_match
        difference = list(set(d['Predicted_ingredient']) - set(dirct_match))
        diff_num = diff_num + len(difference)
        
        if len(dirct_match)>0:
            more_prompt = 'After a quick judgment,' + ', '.join(dirct_match) + 'can be directly evaluated as (\'Correct\', \'Direct Match\'). Please directly evaluate the remaining predicted ingredients: ' + ', '.join(difference)
        else:
            more_prompt = 'After a quick judgment, none of predicted ingredients can be directly evaluated as (\'Correct\', \'Direct Match\'). Please directly evaluate all the predicted ingredients: ' + ', '.join(difference)

        final_promts = final_promts + '\n\n' + more_prompt

        data_statistic[country]['all_num'] = all_num/len(data_list)
        data_statistic[country]['diff_num'] = diff_num/len(data_list)
        # try:
        #     if args.model == 'gpt-4o':
        #         completion = client.chat.completions.create(
        #             model="gpt-4o-2024-08-06",
        #             messages=[
        #                 {"role": "user", "content": final_promts}
        #                 ]
        #         )

        #     content = completion.choices[0].message.content
        #     d['evaluated_output'] = content
            
        #     data_output.append(d)
        #     if len(data_output)%10 == 0:
        #         with open(save_file, 'w', encoding='utf-8') as f:
        #             json.dump(data_output, f, ensure_ascii=False, indent=4)
        # except Exception as e:
        #     # 捕获所有其他异常并记录
        #     erros_count = erros_count + 1




with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(data_statistic, f, ensure_ascii=False, indent=4)

print(f'error:{erros_count}')




