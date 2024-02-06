import pandas as pd
import json
import numpy as np

import pickle

# en最好的prompt2
# zh最好的prompt0



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
    'Others': 'Others',
    'aggregated': 'ALL',
}

f = open('/home/nlp/ZL/FmLAMA-master/data/country_info.json', 'r')
content = f.read()
f.close()
#转化为字典
countries_info = json.loads(content)

country_filter = {
    'Italy': 'Italy',
    'United States of America': 'U.S.',
    'Japan': 'Japan',
    'France': 'France',
    'United Kingdom': 'U.K.',
    'Mexico': 'Mexico',
    'India': 'India',
    'Germany': 'Germany',
    'People\'s Republic of China': 'China',
    'Greece': 'Greece',
    'Spain': 'Spain',
    'Others': 'Others',
    'aggregated': 'ALL',
}





root_dir = '/home/nlp/ZL/FmLAMA-master'
type_data = 'wo_lang'
# langs = ['ar', 'en', 'he', 'ko', 'ru', 'zh']
langs = ['en']
# Groups = ['withC']
Groups = ['without']




Best_prompt = {
    'ar': [0,1,2,3,4],
    'en': [0,1,2,3,4], 'he':[0,1,2,3,4], 'ko':[0,1,2,3,4], 'ru':[0,1,2,3,4], 'zh':[0,1,2,3,4] 
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
                # if m == 'P@1':
                #     if model in choose_data['P@1']:
                #         choose_data['P@1'][model].append((c,mean))
                #     else:
                #         choose_data['P@1'][model]=[(c,mean)]
                # elif m == 'P@5':
                #     if model in choose_data['P@5']:
                #         choose_data['P@5'][model].append((c,mean))
                #     else:
                #         choose_data['P@5'][model]=[(c,mean)]
    
                if m == 'mWS':
                    if model in choose_data['mWS']:
                        choose_data['mWS'][model].append((c,mean))
                    else:
                        choose_data['mWS'][model]=[(c,mean)]
                all_data[country][model] = f'{mean:.4f}±{std:.2f}'

        for country, model_value in all_data.items():
            country_value = []
            for model, v in model_value.items():
                v = v.split('±')
                country_value.append(float(v[0]))
            all_data[country]['Average'] = f'{statistics.mean(country_value):.4f}'

        # 设置最大值，最小值格式
        for metric, values in choose_data.items():
            for model, country_mean in values.items():
                mean_value_list = [i[1] for i in country_mean]
                max_v = max(mean_value_list)
                max_in = mean_value_list.index(max_v)
                value = all_data[f'{country_mean[max_in][0]}_{metric}'][model]
                all_data[f'{country_mean[max_in][0]}_{metric}'][model] = rf'\textbf{{{value}}}'

                sorted_data = sorted(enumerate(mean_value_list), key=lambda x: x[1], reverse=True)

                # 第二大的数据及其索引
                second_v = sorted_data[1][1]
                second_in = sorted_data[1][0]
                value_s = all_data[f'{country_mean[second_in][0]}_{metric}'][model]
                all_data[f'{country_mean[second_in][0]}_{metric}'][model] = rf'\underline{{{value_s}}}'

                # print('a')


        all_data = pd.DataFrame(all_data).T

        all_data.to_excel(f'{root_dir}/results/results_{type_data}/results_table/{lang}_mWS_{Group_name}.xlsx', index=True)


        # stat = pd.DataFrame.from_dict(stat, orient='index', columns=['Value'])
        # stat.to_excel(f'{root_dir}/results/results_{type_data}/results_table/{lang}.xlsx', index=True)



