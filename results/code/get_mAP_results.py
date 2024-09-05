import pandas as pd
from metrics_utils import *
import json
import pickle

# output_wo_lang 是同种语言所有的结果
# output_wo是公用的recipe
root_dir = "/mntcephfs/lab_data/zhouli/personal/FmLAMA"

# with open('/mntcephfs/lab_data/zhouli/personal/FmLAMA/output/results/Llama3/en/hasParts_2/result_temp.pkl', "rb") as f:
#     results = pickle.load(f)

# for r in results:
#     if r["subj_name"] == "Jujeh kabab":
#         print('a')

# with open('/mntcephfs/lab_data/zhouli/personal/FmLAMA/output/results/mT5_base/en/hasParts_2/result.pkl', "rb") as f:
#     results_1 = pickle.load(f)

# for i in range(len(results)):
#     l2 = results[i]
#     mt5 = results_1[i]
#     print('a')

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
# langs = ['en', 'ar', 'he', 'ko', 'ru', 'zh']
langs = ['en']
Groups = ['withC']
# Groups = ['without']


for lang in langs:
    for Group_name in Groups:

        if type_data in ['lang']:
            country_name = country_lang
        else:
            country_name = {i:i for i in list(set(countries_info.values()))}
            country_name['aggregated'] = 'ALL'


        if Group_name == 'withC':
            Group = ['country_1', 'country_2', 'country_3', 'country_4', 'country_5']
            # Group = ['country_2']
        else:
            Group = ['hasParts_1', 'hasParts_2', 'hasParts_3', 'hasParts_4', 'hasParts_5']
            # Group = ['hasParts_2',]

        all_data = {}
        all_stat = {}
        for v in country_name.values():
            vm =  v + '_mAP'
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


        for model_name, model in LMs.items():
            for relation_id in Group:
                results = load_rank_results(results_dir, relation_id, model_name, lang, countries_info)
                mAP_results = compute_mAP(results, country_name, type_data)
                for name, value in mAP_results.items():
                    name = name.split('_')
                    if name[0] == 'mAP' and name[1] in country_name.keys():
                        c_name = country_name[name[1]] + '_' + name[0]
                        if LMs[model_name] not in all_data[c_name].keys():
                            all_data[c_name][LMs[model_name]] = []
                            all_data[c_name][LMs[model_name]].append(value)
                        else:
                            all_data[c_name][LMs[model_name]].append(value)

                    # if name[0] == 'P@1' and name[1] in country_name.keys():
                    #     c_name = country_name[name[1]] + '_' + name[0]
                    #     if LMs[model_name] not in all_data[c_name].keys():
                    #         all_data[c_name][LMs[model_name]] = []
                    #         all_data[c_name][LMs[model_name]].append(value)
                    #     else:
                    #         all_data[c_name][LMs[model_name]].append(value)
                            
                    # if name[0] == 'P@5' and name[1] in country_name.keys():
                    #     c_name = country_name[name[1]] + '_' + name[0]
                    #     if LMs[model_name] not in all_data[c_name].keys():
                    #         all_data[c_name][LMs[model_name]] = []
                    #         all_data[c_name][LMs[model_name]].append(value)
                    #     else:
                    #         all_data[c_name][LMs[model_name]].append(value)

                
                all_num = mAP_results['Support_aggregated']
                for name, value in mAP_results.items(): # 统计
                    name = name.split('_')
                    if name[0] == 'Support':
                        c_name = country_name[name[1]]
                        all_stat[c_name] = str(value) + f' ({round(value/all_num,2)*100}%)'


        
        f_save = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_{Group_name}.pkl', 'wb')
        pickle.dump(all_data, f_save)
        f_save.close()

        f_save = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_stat.pkl', 'wb')
        pickle.dump(all_stat, f_save)
        f_save.close()

        all_map = all_data['ALL_mAP']
        


