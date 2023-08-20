from metrics_utils import *
import json


def wirte_data(list, file_path):
    
    with open(file_path, 'w', encoding='utf-8') as  f:
        for l in list:
            json_str = json.dumps(l, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
            # json.dumps(list, f)

results_dir='/home/nlp/ZL/FmLAMA-master/output/results'
relation_id='hasParts_1'
model_name='xlm-roberta_large'
lang='en' 

results = load_predicate_results(results_dir, relation_id, model_name, lang)
# aa=load_model_results(results_dir, model_name, lang, relation_predicates=None)

aa = compute_P_at_1(results)
t5 = compute_P_at_5(results)
bb_1 = compute_P_scores_at_1(results)
bb_1 = [(k,v) for k,v in bb_1.items()]
bb_5 = compute_P_scores_at_5(results)
bb_5 = [(k,v) for k,v in bb_5.items()]
wirte_data(bb_1, str(Path(results_dir, model_name, lang, relation_id, "evaluation_1.jsonl")))
wirte_data(bb_5, str(Path(results_dir, model_name, lang, relation_id, "evaluation_5.jsonl")))
print('a')