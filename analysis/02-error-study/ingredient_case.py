import json
from pathlib import Path
import pickle

ZH_LMs = {"chinese_bert_base": 'Bb-zh', 
          "mbert_base_cased": 'mB-c', 
          "mbert_base_uncased": 'mB-u', 
          "xlm-roberta_base": 'XRb', 
          "xlm-roberta_large": 'XRl',
          "mT5_base": 'mT5'
          }

EN_LMs = {
          "bert-base_uncased": 'Bb',
          "bert-large_uncased": 'Bl', 
          "mbert_base_uncased": 'mB', 
          "mT5_base": 'mT5',
          "T5_base": 'T5',
          "Qwen2": 'Qwen2',
          "Llama2": 'Llama2',
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

root_dir = "/mntcephfs/lab_data/zhouli/personal/FmLAMA"
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
# Groups = ['withC']
Groups = ['without']


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
                file_path = str(Path(results_dir, model_name, lang, relation_id, "result.pkl"))
                with open(file_path, "rb") as f:
                    results = pickle.load(f)
                    
                results_for_analysis = []
                print('a')

print()