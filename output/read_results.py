import pandas as pd
# from metrics_utils import *
import json
import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='output_code/', help='experiment results type')
parser.add_argument('--model', type=str, default='mbert_base_cased', help='probing model')
parser.add_argument('--lang', type=str, default='en', help='prompt language')
parser.add_argument('--prompt', type=str, default='hasParts_1', help='prompt type')
args = parser.parse_args()

result_file = args.root_dir + args.model + '/' + args.lang + '/' + args.prompt + '/' + 'result.pkl'



# result_file = str(Path(args.root_dir, args.model, args.lang, args.prompt, "result.pkl"))
with open(result_file, "rb") as f:
    results = pickle.load(f)

print('a')