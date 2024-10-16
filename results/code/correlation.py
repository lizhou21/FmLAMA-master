import pandas as pd
import json
import numpy as np

import pickle


from scipy.stats import pearsonr, spearmanr

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


root_dir = 'FmLAMA'
f = open(f"{root_dir}/data/country_info.json", 'r')
content = f.read()
f.close()

countries_info = json.loads(content)






type_data = 'lang'
langs = ['en']
Groups = ['without']




Best_prompt = {
    'ar': [0,1,2,3,4],
    'en': [0,1,2,3,4], 'he':[0,1,2,3,4], 'ko':[0,1,2,3,4], 'ru':[0,1,2,3,4], 'zh':[0,1,2,3,4] 
}

model_mAP = {
    'Bb': {}, 
    'Bl': {}, 
    'mB': {}, 
    'mT5': {}, 
    'T5': {}, 
    'Qwen2': {}, 
    'Llama2': {}, 
    'Llama3': {}
}

mAP_result = []
for lang in langs:
    for Group_name in Groups:
        if type_data in ['lang', 'wo_lang']:
            country_name = country_lang
        else:
            country_name = {i:i for i in list(set(countries_info.values()))}
            country_name['aggregated'] = 'ALL'

        f_read = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_{Group_name}.pkl', 'rb')
        f_read_stat = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_stat.pkl', 'rb')

        results = pickle.load(f_read)
        stat = pickle.load(f_read_stat)

        f_read.close()
        f_read_stat.close()

        all_data = {}
        for v in country_name.values():
            if v not in ['Eurasia', 'Insular Oceania']:
                v_m = v + '_mAP'
                all_data[v_m] = {}

        choose_data = {}
        choose_data['mAP'] = {}

        # index_list = []
        import statistics
        for country, res in results.items():
            c = country.split('_')[0]
            m = country.split('_')[1]
            for model, values in res.items():
                input_value = [values[i] for i in Best_prompt[lang]]
                mean = statistics.mean(input_value)
                std = statistics.stdev(input_value)
                all_data[country][model] = f'{mean:.2f}'

    
                if m == 'mAP':
                    if model in choose_data['mAP']:
                        choose_data['mAP'][model].append((c,mean))
                    else:
                        choose_data['mAP'][model]=[(c,mean)]
                all_data[country][model] = f'{mean:.2f}±{std:.2f}'
                mAP_result.append(mean)
                model_mAP[model][country]=mean
                # mAP_result

mWS_result = []

model_mWS = {
    'Bb': {}, 
    'Bl': {}, 
    'mB': {}, 
    'mT5': {}, 
    'T5': {}, 
    'Qwen2': {}, 
    'Llama2': {}, 
    'Llama3': {}
}
for lang in langs:
    for Group_name in Groups:
        if type_data in ['lang', 'wo_lang']:
            country_name = country_lang
        else:
            country_name = {i:i for i in list(set(countries_info.values()))}
            country_name['aggregated'] = 'ALL'



        f_read = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mWS_{Group_name}.pkl', 'rb')
        f_read_stat = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mWS_stat.pkl', 'rb')


        results = pickle.load(f_read)
        stat = pickle.load(f_read_stat)


        f_read.close()
        f_read_stat.close()

      
      
        all_data = {}
        for v in country_name.values():
            if v not in ['Eurasia', 'Insular Oceania']:
                # v1 = v + '_P@1'
                # v2 = v + '_P@5'
                v_m = v + '_mWS'
                # all_data[v1] = {}
                # all_data[v2] = {}
                all_data[v_m] = {}

        choose_data = {}
        # choose_data['P@1'] = {}
        # choose_data['P@5'] = {}
        choose_data['mWS'] = {}


        # index_list = []
        import statistics
        for country, res in results.items():
            c = country.split('_')[0]
            m = country.split('_')[1]
            for model, values in res.items():
                input_value = [values[i] for i in Best_prompt[lang]]
                mean = statistics.mean(input_value)
                std = statistics.stdev(input_value)
                all_data[country][model] = f'{mean:.4f}'
                if m == 'mWS':
                    if model in choose_data['mWS']:
                        choose_data['mWS'][model].append((c,mean))
                    else:
                        choose_data['mWS'][model]=[(c,mean)]
                all_data[country][model] = f'{mean:.4f}±{std:.2f}'
                mWS_result.append(mean*100)
                model_mWS[model][country]=mean*100


from scipy.stats import pearsonr, spearmanr
print(spearmanr(mAP_result, mWS_result))

model_spearmanr = {}
model_pearsonr = {}
for model in model_mWS.keys():
    model_pearsonr[model]=pearsonr(list(model_mWS[model].values()), list(model_mAP[model].values()))[0]
    model_spearmanr[model]=spearmanr(list(model_mWS[model].values()), list(model_mAP[model].values()))[0]

print(pearsonr(mAP_result, mWS_result))
print('a')
