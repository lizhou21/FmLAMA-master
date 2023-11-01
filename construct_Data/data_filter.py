import json
import pickle
from collections import Counter
import os

def wirte_data(list, file_path):
    
    with open(file_path, 'w', encoding='utf-8') as  f:
        for l in list:
            json_str = json.dumps(l, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')

root_dir = "/home/nlp/ZL/FmLAMA-master/data/data_lang/"
languages = ['ar', 'he', 'ko', 'ru', 'en', 'zh']

id_list = {}

for la in languages:
    relation_triples_raw = []
    dishes = f"{root_dir}{la}/{la}_dishes.jsonl"
    with open(dishes, "r") as f:
        relation_triples_raw += [json.loads(line) for line in f]
    ids = [r['url'] for r in relation_triples_raw]
    id_list[la]=ids



candidate_lang = ['ar', 'he', 'ko', 'ru', 'en', 'zh']



intersection = set(id_list['ar']).intersection(id_list['he'], id_list['ko'], id_list['ru'], id_list['en'], id_list['zh'])



for la in candidate_lang:
    relation_triples_raw = []
    filter = []

    save_dir = '/home/nlp/ZL/FmLAMA-master/data/data_filter/' + la
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dishes = f"{root_dir}{la}/{la}_dishes.jsonl"
    with open(dishes, "r") as f:
        relation_triples_raw += [json.loads(line) for line in f]
    for item in relation_triples_raw:
        if item['url'] in intersection:
            filter.append(item)
    
    wirte_data(filter, save_dir + '/' + la + '_' + 'dishes.jsonl',)

    count_country = Counter([ge['origin'] for ge in filter])
    count_country = sorted(count_country.items(),key=lambda x:x[1],reverse=True)
    count_country.insert(0, ("All", len(filter)))
    wirte_data(count_country, save_dir + '/' + la + '_' + 'count.jsonl', )
