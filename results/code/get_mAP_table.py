import pandas as pd
import json
import numpy as np

import pickle




root_dir = "FmLAMA"

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

countries_info = json.loads(content)




type_data = 'lang'
langs = ['ar', 'en', 'he', 'ko', 'ru', 'zh']
Groups = ['without']
# Groups = ['without', 'withC']
langs = ['ko']
# langs = ['zh']




Best_prompt = {
    'ar': [0,1,2,3,4],
    'en': [0,1,2,3,4], 'he':[0,1,2,3,4], 'ko':[0,1,2,3,4], 'ru':[0,1,2,3,4], 'zh':[0,1,2,3,4] 
}




Best_prompt = {}
for l in langs:
    Best_prompt[l] = [0,1,2,3,4]



for lang in langs:
    for Group_name in Groups:
        if type_data in ['lang', 'wo_lang']:
            country_name = country_lang
        else:

            country_name = {
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
                    # 'Iran': 'Iran',
                    'Greece': 'Greece',
                    'Spain': 'Spain',
                    'Russia': 'Russia',
                    'aggregated': 'ALL',
                }



        f_read = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_{Group_name}.pkl', 'rb')
        f_read_stat = open(f'{root_dir}/results/results_{type_data}/results_pickle/{lang}_mAP_stat.pkl', 'rb')


        results = pickle.load(f_read)
        stat = pickle.load(f_read_stat)


        f_read.close()
        f_read_stat.close()

      
      
        all_data = {}
        for v in country_name.values():
            if v not in ['Eurasia', 'Insular Oceania']:
                # v1 = v + '_P@1'
                # v2 = v + '_P@5'
                v_m = v + '_mAP'
                # all_data[v1] = {}
                # all_data[v2] = {}
                all_data[v_m] = {}

        choose_data = {}
        # choose_data['P@1'] = {}
        # choose_data['P@5'] = {}
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
                # all_data[country][model] = f'{mean:.2f}'

        for country, model_value in all_data.items():
            country_value = []
            for model, v in model_value.items():
                v = v.split('±')
                country_value.append(float(v[0]))
            all_data[country]['Average'] = f'{statistics.mean(country_value):.2f}'


        for metric, values in choose_data.items():
            for model, country_mean in values.items():
                mean_value_list = [i[1] for i in country_mean]
                max_v = max(mean_value_list)
                max_in = mean_value_list.index(max_v)
                value = all_data[f'{country_mean[max_in][0]}_{metric}'][model]
                all_data[f'{country_mean[max_in][0]}_{metric}'][model] = rf'\textbf{{{value}}}'

                sorted_data = sorted(enumerate(mean_value_list), key=lambda x: x[1], reverse=True)


                second_v = sorted_data[1][1]
                second_in = sorted_data[1][0]
                value_s = all_data[f'{country_mean[second_in][0]}_{metric}'][model]
                all_data[f'{country_mean[second_in][0]}_{metric}'][model] = rf'\underline{{{value_s}}}'




        all_data = pd.DataFrame(all_data).T

        all_data.to_excel(f'{root_dir}/results/results_{type_data}/results_table/{lang}_mAP_{Group_name}.xlsx', index=True)


        stat = pd.DataFrame.from_dict(stat, orient='index', columns=['Value'])
        stat.to_excel(f'{root_dir}/results/results_{type_data}/results_table/{lang}.xlsx', index=True)



