import pandas as pd
import json
import ast
import os
from collections import Counter
from zhconv import convert
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

languages = ['en', 'ru', 'zh', 'ko', 'ar', 'he']

FmLAMA = pd.read_csv('/home/nlp/ZL/FmLAMA-master/construct_Data/output_data/Dishes.csv')

for la in languages:
    FmLAMA_sub = FmLAMA[FmLAMA["lang"]==la].reset_index(drop=True)
    subj_id = []
    generate_results = []

    # 去query重
    url_list = list(FmLAMA_sub['url'])
    url_list_count = Counter(url_list)
    url_list_single = [item for item, count in url_list_count.items() if count == 1]
    for index, row in FmLAMA_sub.iterrows():
        sp = row["url"].split('/')
        if row["url"] in url_list_single:
            subj_id.append(sp[-1])
            if la == 'zh':
                row['name'] = convert(row['name'], 'zh-cn')
                row['hasParts'] = convert(row['hasParts'], 'zh-cn')

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

    save_dir = '/home/nlp/ZL/FmLAMA-master/data/data_lang/' + la
    if not os.path.exists(save_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_dir)

    # save_count
    wirte_data(count_country, save_dir + '/' + la + '_' + 'count.jsonl', )
    
    # save dishes
    wirte_data(generate_results, save_dir + '/' + la + '_' + 'dishes.jsonl', )


