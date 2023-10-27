from metrics_utils import *
import json




results_dir='/home/nlp/ZL/FmLAMA-master/output/results'
relation_id=['hasParts_1', 'hasParts_2', 'hasParts_3', 'hasParts_4', 'hasParts_5']

models_name= ['bert-base_cased', 'bert-base_uncased', 'bert-large_cased', 'bert-large_uncased',
              'roberta_base', 'roberta_large', 'mbert_base_cased', 'mbert_base_uncased',
              'xlm-roberta_base', 'xlm-roberta_large', 'mT5_base', 'T5_base']

countries = ['United States of America', 'France', 'Italy', 'India',
             'Japan', 'United Kingdom', 'Spain', 'People\'s Republic of Chin',
             'Turkey', 'Indonesia', 'Korea', 'Mexico',
             'aggregated']

lang='en' 

results = load_predicate_results(results_dir, relation_id, model_name, lang)
# aa=load_model_results(results_dir, model_name, lang, relation_predicates=None)

aa = compute_P_at_1(results)
t5 = compute_P_at_5(results)
bb_1 = compute_P_scores_at_1(results)
bb_1 = [(k,v) for k,v in bb_1.items()]
bb_5 = compute_P_scores_at_5(results)
bb_5 = [(k,v) for k,v in bb_5.items()]
# wirte_data(bb_1, str(Path(results_dir, model_name, lang, relation_id, "evaluation_1.jsonl")))
# wirte_data(bb_5, str(Path(results_dir, model_name, lang, relation_id, "evaluation_5.jsonl")))
print('a')