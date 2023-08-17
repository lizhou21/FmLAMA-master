import pandas as pd
import json
import ast
import os
from collections import Counter

def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        ret = []
        for i, item in enumerate(f.readlines()):
            record = json.loads(item)
            ret.append(record)
    return ret

def wirte_data(list, file_path):
    
    with open(file_path, 'w', encoding='utf-8') as  f:
        for l in list:
            json_str = json.dumps(l, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
            # json.dumps(list, f)

languages = ['en', 'es', 'ja', 'fr', 'de', 'it', 'nl', 'ru', 'zh', 'ko', 'uk', 'ca', 'eo', 'id', 'pt', 'ar', 'pl', 'he']

FmLAMA = pd.read_csv('/home/nlp/ZL/FmLAMA-master/data/FmLAMA.csv')

for la in languages:
    FmLAMA_sub = FmLAMA[FmLAMA["lang"]==la].reset_index(drop=True)
    # FmLAMA_sub.rename(columns={'name':'sub_label'}, inplace = True)
    # FmLAMA_sub.rename(columns={'hasParts':'obj_label'}, inplace = True)
    # count_by_value = FmLAMA_sub['origin'].value_counts().to_frame()
    # count_by_url = FmLAMA_sub['url'].value_counts().to_frame()
    
    subj_id = []
    generate_results = []

    # 去query重
    for index, row in FmLAMA_sub.iterrows():
        sp = row["url"].split('/')
        if sp[-1] not in subj_id:
            subj_id.append(sp[-1])
            new_dish = {}
            new_dish['url'] = row['url']
            new_dish['origin'] = row['origin']
            new_dish['lang'] = row['lang']
            new_dish['sub_label'] = row['name']
            new_dish['obj_label'] = row['hasParts']
            generate_results.append(new_dish)
    count_country = Counter([ge['origin'] for ge in generate_results])
    count_country = sorted(count_country.items(),key=lambda x:x[1],reverse=True)
    count_country.insert(0, ("All", len(generate_results)))

    save_dir = '/home/nlp/ZL/FmLAMA-master/data/' + la
    if not os.path.exists(save_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_dir)

    # save_count
    wirte_data(count_country, save_dir + '/' + la + '_' + 'count.jsonl', )
    
    # save dishes
    wirte_data(generate_results, save_dir + '/' + la + '_' + 'dishes.jsonl', )



