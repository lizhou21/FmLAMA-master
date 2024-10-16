import json
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import ast
ZH_LMs = {"chinese_bert_base": 'Bb-zh', 
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }

EN_LMs = {
        #   "bert-base_uncased": 'Bb',
        #   "bert-large_uncased": 'Bl', 
        #   "mbert_base_uncased": 'mB', 
        #   "mT5_base": 'mT5',
        #   "T5_base": 'T5',
        #   "Qwen2": 'Qwen2',
        #   "Llama2": 'Llama2',
          "Llama3": 'Llama3',
          
          }


AR_LMs = {"arabic_bert_base": "Bb-ar",
          "arabic_bert_large": "Bl-ar",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }


HE_LMs = {"mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }

KO_LMs = {"kykim_bert_base": "Bb-ky",
          "klue_bert_base": "Bb-kl",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }


RU_LMs = {"rubert-base-cased": "Bb-ru",
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }




def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        ret = []
        for i, item in enumerate(f.readlines()):
            record = json.loads(item)
            ret.append(record)
    return ret

root_dir = "FmLAMA"
results_dir = f"{root_dir}/output/results/"
f = open(f'{root_dir}/data/ingredient_info_English.json', 'r')
ingredient_info_English = f.read()
f.close()
ingredient_info_English = json.loads(ingredient_info_English)


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
}


country_language = {
    'Italy': 'it',
    'United States of America': 'en',
    'Turkey': 'tr',
    'Japan': 'ja',
    'France': 'fr',
    'United Kingdom': 'en',
    'Mexico': 'es',
    'India': 'hi',
    'Germany': 'de',
    'People\'s Republic of China': 'zh',
    'Iran': 'fa',
    'Greece': 'el',
    'Spain': 'es',
    'Russia': 'ru',
    # 'aggregated': 'ALL',
}

f = open(f'{root_dir}/data/country_info.json', 'r')
content = f.read()
f.close()
#转化为字典
countries_info = json.loads(content) # country: continent

type_data = 'lang'
# langs = ['en', 'ar', 'he', 'ko', 'ru', 'zh']
langs = ['en']
# Groups = ['withC']
Groups = ['without']

FmLAMA = pd.read_csv('/mntcephfs/lab_data/zhouli/personal/FmLAMA/data/Dishes.csv')

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
            Group = ['hasParts_1']
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

        c = 0
        for model_name, model in LMs.items():
            for relation_id in Group:
                file_path = str(Path(results_dir, model_name, lang, relation_id, "result.pkl"))
                with open(file_path, "rb") as f:
                    results = pickle.load(f)
                    
                results_for_analysis = {}
                countries_count = {}
                
                for count in country_lang.keys():
                    results_for_analysis[count] = []
                    countries_count[count] = 0
                for dish in tqdm(results):
                    d = {}
                    d['url'] = dish['ID']
                    d['origin'] =dish['origin']
                    d['dish'] = dish['subj_name']
                    
                    d['gold_ingredient'] = dish['obj_name']
                    count = len(d['gold_ingredient'])
                    d['predicted_ingredient'] = dish['predicted'][:count]

                    FmLAMA_url = FmLAMA[FmLAMA["url"]==dish['ID']].reset_index(drop=True)
                    official_name = country_language[dish['origin']]

                    image_info = list(set(list(FmLAMA_url['image'])))

                    image = ast.literal_eval(image_info[0])
                    d['image'] = image
                    dish_origin = FmLAMA_url[FmLAMA_url['lang']==official_name].reset_index(drop=True)
                    origin_name = list(dish_origin['name'])
                    if len(origin_name) == 1:
                        d["dish_name"] = origin_name[0]
                        
                    elif len(origin_name) == 0:
                        d["dish_name"] = ''
                        countries_count[d['origin']] = countries_count[d['origin']] + 1
                        if "China" in d['origin']:
                            print(d)
                    else:
                        print('a')
                        

                    results_for_analysis[dish['origin']].append(d)

with open(f'{root_dir}/data/English_human_evaluation.json', 'w') as f:
    json.dump(results_for_analysis, f, indent=4, ensure_ascii=False)
