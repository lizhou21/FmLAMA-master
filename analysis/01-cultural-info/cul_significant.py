import pandas as pd
import json
import numpy as np
import re
import pickle
from scipy import stats


def get_value(data,co):
    value_output = []
    value_list = data[co].tolist()
    delimiters = r'[{}±]+'
    for value in value_list:
        value = re.split(delimiters, value)
        if len(value) == 2:
            value_output.append(float(value[0]))
        else:
            value_output.append(float(value[1]))
    return value_output


root_dir = "/mntcephfs/lab_data/zhouli/personal/FmLAMA"

for metric in ['mAP', 'mWS']:

    withC_file = f"{root_dir}/results/results_lang/results_table/en_{metric}_withC.xlsx"

    with_file = f"{root_dir}/results/results_lang/results_table/en_{metric}_without.xlsx"

    withC_data = pd.read_excel(withC_file)
    without_data = pd.read_excel(with_file)

    column_names = withC_data.columns.tolist()

    sig_score = {}
    for co in column_names[1:-1]:
        withC = get_value(withC_data, co)
        without = get_value(without_data, co)
        reslut = stats.wilcoxon(withC, without,  alternative='greater')
        sig_score[co]=reslut[1]
    print(metric)
    print(sig_score)
    print('\n')

        