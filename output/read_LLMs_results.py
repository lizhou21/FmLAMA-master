import pandas as pd
# from metrics_utils import *
import json
import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='output_LLMs/', help='experiment results type')
parser.add_argument('--model', type=str, default='Vicuna', help='[Bloom, chatGPT, LLaMa, Vicuna]')
parser.add_argument('--model_name', type=str, default='vicuna-7b-v1.5', 
                    help='only for LLaMa, Vicuna: [vicuna-7b-v1.5, vicuna-13b-v1.5, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf]')
parser.add_argument('--lang', type=str, default='en_en', help='prompt language: [en_en, en_zh, zh_zh, zh_en]')
parser.add_argument('--instruct_type', type=str, default='one', 
                    help='instruct_type, one: please fill this sentence with one word. / full: please fill this sentence.')

parser.add_argument('--prompt', type=str, default='hasParts_1', help='prompt type')
args = parser.parse_args()

if args.model == 'Bloom':
    result_file = args.root_dir + args.model + '/' + args.lang + '/' + args.instruct_type + '/' +  args.prompt + '/' + args.lang + '_dishes_results.jsonl'

elif args.model == 'chatGPT':
    result_file = args.root_dir + args.model + '/WC/' + args.lang + '/' + args.instruct_type + '/' +  args.prompt + '/' + args.lang + '_dishes_results.jsonl'

else:
    result_file = args.root_dir + args.model + '/WC/' + args.lang + '/' + args.instruct_type + '/' +  args.prompt + '/' + args.model_name + '_' + args.lang + '_dishes_results.jsonl'

# result_file = str(Path(args.root_dir, args.model, args.lang, args.prompt, "result.pkl"))
dishes = []
with open(result_file, "r", encoding='utf-8') as f:
    dishes += [json.loads(line) for line in f]



