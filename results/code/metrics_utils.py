import glob
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
# from natsort import natsorted
import ast
from tqdm import tqdm
#  TODO: Load this list from the constants.py file within dlama
DOMAINS = [
    "sports",
    "politics",
    "music",
    "cinema_and_theatre",
    "history",
    "science",
    "geography",
]


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
        #   "PhiMini": 'PhiMini',
          
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




# COUNTRY = {
#     'United States of America': 'U.S.',
#     'France': 'France',
#     'Italy': 'Italy',
#     'India': 'India',
#     'Japan': 'Japan',
#     'United Kingdom': 'U.K.',
#     'Spain': 'Spain',
#     'People\'s Republic of China': 'China',
#     'Turkey': 'Turkey',
#     # 'Indonesia': 'Indonesia',
#     # 'Korea': 'Korean',
#     'Mexico': 'Mexico',
#     'Others': 'Others',
#     'aggregated': 'ALL',
# }




def calculate_map(ranked_lists):
    ap_sum = 0
    for ranked_list in ranked_lists:
        precision_sum = 0
        relevant_docs = 0
        for i, doc in enumerate(ranked_list):
            if doc == 1:  # doc is relevant
                relevant_docs += 1 # 获取预测到的
                precision_sum += relevant_docs / (i + 1)#当前预测到的
        ap_sum += precision_sum / relevant_docs
    map = ap_sum / len(ranked_lists)
    return map

def load_rank_results(results_dir, relation_id, model_name, lang, countries_info):
    """
    Return a dataframe of a model's predictions for a specific relation in a specific language.
    """

    def compute_probs(logits):
        exps = np.exp(logits)
        return exps / exps.sum()

    # Load the pickle file
    # with open('/mntcephfs/lab_data/zhouli/personal/FmLAMA/output/results/Llama2/en/hasParts_2/result_temp.pkl', "rb") as f:
    #     results = pickle.load(f)

    file_path = str(Path(results_dir, model_name, lang, relation_id, "result.pkl"))
    with open(file_path, "rb") as f:
        results = pickle.load(f)


    

    predictions_list = []

    for sample in tqdm(results):

        origin = sample["origin"] #获取国家
        continent = countries_info[origin]
        

        
        if len(sample["ranks"]) == 0:
            continue
        # try:
        rank = sample["ranks"]#获取所有的预测排名
        input_rank = [0 for i in list(range(max(rank)+1))]
        for r in rank:
            input_rank[r]=1

        # probs = compute_probs(sample["masked_topk"]["probs"])

        predictions_list.append(
            {
                "origin": origin,
                'continent': continent,
                "input_rank": input_rank,
                "id": sample['ID'],
                "rank": rank,
                "subject": sample["subj_name"],
                "valid_objects": sample['obj_name'],
                "predictions": sample["predicted"],
                # "probabilities": probs,
            }
        )

    df = pd.DataFrame(predictions_list)

    return df






def mAP(df):
    """Compute P@1 score for a dataframe of predictions."""
    rank_lists = [row.input_rank for index, row in df.iterrows()]# 获取每个gold label在预测序列中的位置，预测序列的顺序由大概率降序排列
    # row_values = row.values
    map = calculate_map(rank_lists)
    return round(100*map,2)

def sim(df, fasttext_model):
    """Compute P@1 score for a dataframe of predictions."""
    mWS = []
    id = 0
    for index, row in df.iterrows():
        id = id + 1

        objects = row.valid_objects
        predictions = row.predictions[:len(objects)]
        
        WS = []
        for o in objects:
            Pl = []
            for p in predictions:# vec存在全零的现象
                cs = cosine_similarity(fasttext_model.get_word_vector(o),fasttext_model.get_word_vector(p))
                Pl.append(cs)
            WS.append(max(Pl))
        # if len(WS)>0:
            # mWS.append(sum(WS) / len(WS))
        # else:
            # print('a')
        mWS.append(sum(WS) / len(WS))
    return sum(mWS) / len(mWS)
 

def P_K(df,k):
    rank_lists = [row.input_rank[:k] for index, row in df.iterrows()]
    total_sum = 0
    for row in rank_lists:
        total_sum += sum(row)
    P_K = total_sum/(k*len(rank_lists))
    return round(100*P_K,2)




    
def compute_mAP(df, country_name,type_data):
    if type_data in ['lang', 'wo_lang']:
        regions = list(country_name.keys())
        all_regions = regions
        exlude_others = [i for i in regions if i not in ['Others', 'aggregated']]
        


        scores = {}
        for region in all_regions:
            if region in exlude_others:
                region_df = df[df["origin"] == region]
                scores[f"mAP_{region}"] = mAP(region_df) # metric
                scores[f"P@1_{region}"] = P_K(region_df, 1)
                scores[f"P@5_{region}"] = P_K(region_df, 5)
                scores[f"Support_{region}"] = region_df.shape[0] #数量
        scores[f"mAP_aggregated"] = mAP(df)
        scores[f"P@1_aggregated"] = P_K(df, 1)
        scores[f"P@5_aggregated"] = P_K(df, 5)
        scores[f"Support_aggregated"] = df.shape[0]
    else:
        regions = list(country_name.keys())
        all_regions = regions
        scores = {}
        for region in all_regions:
            region_df = df[df["continent"] == region]
            if region_df.shape[0]==0:
                continue
            scores[f"mAP_{region}"] = mAP(region_df) # metric
            scores[f"P@1_{region}"] = P_K(region_df, 1)
            scores[f"P@5_{region}"] = P_K(region_df, 5)
            scores[f"Support_{region}"] = region_df.shape[0] #数量
        scores[f"mAP_aggregated"] = mAP(df)
        scores[f"P@1_aggregated"] = P_K(df, 1)
        scores[f"P@5_aggregated"] = P_K(df, 5)
        scores[f"Support_aggregated"] = df.shape[0]

    return scores

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    ### START CODE HERE ###
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u,v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.linalg.norm(u)

    # Compute the L2 norm of v (≈1 line)
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot/(norm_u * norm_v)
    ### END CODE HERE ###

    return cosine_similarity



def compute_sim(df, country_name,type_data, fasttext_model):
    if type_data in ['lang', 'wo_lang']:
        regions = list(country_name.keys())
        all_regions = regions
        exlude_others = [i for i in regions if i not in ['Others', 'aggregated']]
        
        scores = {}
        for region in all_regions:
            if region in exlude_others:
                region_df = df[df["origin"] == region]
                scores[f"mWS_{region}"] = sim(region_df, fasttext_model) # metric
                scores[f"Support_{region}"] = region_df.shape[0] #数量
           
        scores[f"mWS_aggregated"] = sim(df, fasttext_model)
        scores[f"Support_aggregated"] = df.shape[0]
    else:
        regions = list(country_name.keys())
        all_regions = regions
        scores = {}
        for region in all_regions:
            region_df = df[df["continent"] == region]
            if region_df.shape[0]==0:
                continue
            scores[f"mWS_{region}"] = sim(region_df, fasttext_model) # metric
            scores[f"Support_{region}"] = region_df.shape[0] #数量
        scores[f"mWS_aggregated"] = sim(df, fasttext_model)
        scores[f"Support_aggregated"] = df.shape[0]

    return scores