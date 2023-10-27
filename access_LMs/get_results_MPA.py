import pandas as pd
from metrics_utils import *
import json
import pickle


 
# Using the function with the ranked list from our example:
# ranked_lists = [[1, 0, 0, 1, 1, 0, 0]]  # 1 for Relevant, 0 for Non-relevant
# print(calculate_map(ranked_lists))  # Outputs: 0.75


ZH_LMs = {"chinese_bert_base": 'Bb-zh', 
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'}


EN_LMs = {"bert-base_cased": 'Bb-c', 
          "bert-base_uncased": 'Bb-u', 
          "bert-large_cased": 'Bl-c', 
          "bert-large_uncased": 'Bl-u', 
          "roberta_base": 'Rb', 
          "roberta_large": 'Rl',
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5',
          "T5_base": 'T5'}


AR_LMs = {"arabic_bert_base": "Bb-ar",
          "arabic_bert_large": "Bl-ar",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'}


HE_LMs = {"mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'}

KO_LMs = {"kykim_bert_base": "Bb-ky",
          "klue_bert_base": "Bb-kl",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'}


RU_LMs = {"rubert-base-cased": "Bb-ru",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'}

langs = ['ar', 'en', 'he', 'ko', 'ru', 'zh']
Groups = ['withC', 'without']

for lang in langs:
    for Group_name in Groups:

        # Group_name = 'withC' # without
        # lang = 'zh'


        if Group_name == 'withC':
            Group = ['country_1', 'country_2', 'country_3', 'country_4', 'country_5']
        else:
            Group = ['hasParts_1', 'hasParts_2', 'hasParts_3', 'hasParts_4', 'hasParts_5']

        country_name = {
            'United States of America': 'U.S.',
            'France': 'France',
            'Italy': 'Italy',
            'India': 'India',
            'Japan': 'Japan',
            'United Kingdom': 'U.K.',
            'Spain': 'Spain',
            'People\'s Republic of China': 'China',
            'Turkey': 'Turkey',
            # 'Indonesia': 'Indonesia',
            # 'Korea': 'Korean',
            'Mexico': 'Mexico',
            'Others': 'Others',
            'aggregated': 'ALL',
        }


        all_data = {}
        all_stat = {}
        for v in country_name.values():
            # v_s = v + '_stat'
            v1 = v + '_P@1'
            v2 = v + '_P@5'
            # all_data[v_s] = []
            all_data[v1] = {}
            all_data[v2] = {}
            vm =  v + '_mAP'
            all_data[vm] = {}


        results_dir='/home/nlp/ZL/FmLAMA-master/output_filter'
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


        for model_name, model in LMs.items():
            for relation_id in Group:
                results = load_rank_results(results_dir, relation_id, model_name, lang)
                mAP_results = compute_mAP(results)
                for name, value in mAP_results.items():
                    name = name.split('_')
                    if name[0] == 'mAP' and name[1] in country_name.keys():
                        c_name = country_name[name[1]] + '_' + name[0]
                        if LMs[model_name] not in all_data[c_name].keys():
                            all_data[c_name][LMs[model_name]] = []
                            all_data[c_name][LMs[model_name]].append(value)
                        else:
                            all_data[c_name][LMs[model_name]].append(value)

                    if name[0] == 'P@1' and name[1] in country_name.keys():
                        c_name = country_name[name[1]] + '_' + name[0]
                        if LMs[model_name] not in all_data[c_name].keys():
                            all_data[c_name][LMs[model_name]] = []
                            all_data[c_name][LMs[model_name]].append(value)
                        else:
                            all_data[c_name][LMs[model_name]].append(value)
                            
                    if name[0] == 'P@5' and name[1] in country_name.keys():
                        c_name = country_name[name[1]] + '_' + name[0]
                        if LMs[model_name] not in all_data[c_name].keys():
                            all_data[c_name][LMs[model_name]] = []
                            all_data[c_name][LMs[model_name]].append(value)
                        else:
                            all_data[c_name][LMs[model_name]].append(value)

                
                all_num = mAP_results['Support_aggregated']
                for name, value in mAP_results.items(): # 统计
                    name = name.split('_')
                    if name[0] == 'Support':
                        c_name = country_name[name[1]]
                        all_stat[c_name] = str(value) + f' ({round(value/all_num,2)*100}%)'


        f_save = open(f'/home/nlp/ZL/FmLAMA-master/results_filter/{lang}_mAP_{Group_name}.pkl', 'wb')
        pickle.dump(all_data, f_save)
        f_save.close()

        f_save = open(f'/home/nlp/ZL/FmLAMA-master/results_filter/{lang}_mAP_stat.pkl', 'wb')
        pickle.dump(all_stat, f_save)
        f_save.close()


