import fasttext
import numpy as np
import pandas as pd
import json
import pickle
from metrics_utils import *
root_dir = "/mntcephfs/lab_data/zhouli/personal/FmLAMA"
model = fasttext.load_model(f'{root_dir}/fasttext/cc.en.300.bin')



country_lang = {
    'Italy': 'Italy',
    'United States of America': 'U.S.',
    'Turkey': 'Turkey',
    'Japan': 'Japan',
    'France': 'France',
    'United Kingdom': 'U.K.',
    'Mexico': 'Mexico',
    'India': 'India',
    'Germany': 'Germany',
    'People\'s Republic of China': 'China',
    'Iran': 'Iran',
    'Greece': 'Greece',
    'Spain': 'Spain',
    'Russia': 'Russia',
    'aggregated': 'ALL',
}





f = open(f'{root_dir}/data/country_info.json', 'r')
content = f.read()
f.close()
#转化为字典
countries_info = json.loads(content) # country: continent


type_data = 'lang'
# langs = ['en', 'ar', 'ko', 'ru', 'zh', 'he']
langs = ['en']

Groups = ['withC']
# Groups = ['without']


for lang in langs:
    for Group_name in Groups:

        if type_data in ['lang', 'wo_lang']:
            country_name = country_lang
        else:
            country_name = {i:i for i in list(set(countries_info.values()))}
            country_name['aggregated'] = 'ALL'


        if Group_name == 'withC':
            Group = ['country_1', 'country_2', 'country_3', 'country_4', 'country_5']
        else:
            Group = ['hasParts_1', 'hasParts_2', 'hasParts_3', 'hasParts_4', 'hasParts_5']

        all_data = {}
        all_stat = {}
        for v in country_name.values():
            vm =  v + '_mWS'
            all_data[vm] = {}


        results_dir=f'{root_dir}/output/results/'
        if lang == 'en':
            LMs = EN_LMs
        elif lang == 'zh':
            LMs = ZH_LMs
        elif lang == 'ar':
            LMs = AR_LMs
        elif lang == 'he':
            LMs = HE_LMs
        elif lang == 'ko':
            LMs = KO_LMs
        elif lang == 'ru':
            LMs = RU_LMs

        model_name = f'{root_dir}/fasttext/cc.'+lang+'.300.bin'
        fasttext_model = fasttext.load_model(model_name)

        for model_name, model in LMs.items():
            for relation_id in Group:
                results = load_rank_results(results_dir, relation_id, model_name, lang, countries_info)
                sim_results = compute_sim(results, country_name, type_data, fasttext_model)
                for name, value in sim_results.items():
                    name = name.split('_')
                    if name[0] == 'mWS' and name[1] in country_name.keys():
                        c_name = country_name[name[1]] + '_' + name[0]
                        if LMs[model_name] not in all_data[c_name].keys():
                            all_data[c_name][LMs[model_name]] = []
                            all_data[c_name][LMs[model_name]].append(value)
                        else:
                            all_data[c_name][LMs[model_name]].append(value)


                
                all_num = sim_results['Support_aggregated']
                for name, value in sim_results.items(): # 统计
                    name = name.split('_')
                    if name[0] == 'Support':
                        c_name = country_name[name[1]]
                        all_stat[c_name] = str(value) + f' ({round(value/all_num,2)*100}%)'


        f_save = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mWS_{Group_name}.pkl', 'wb')
        pickle.dump(all_data, f_save)
        f_save.close()

        f_save = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mWS_stat.pkl', 'wb')
        pickle.dump(all_stat, f_save)
        f_save.close()



