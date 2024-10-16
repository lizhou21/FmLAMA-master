import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


root_dir = 'FmLAMA'
all_results = {}
langs = ['ar', 'en', 'he', 'ru', 'zh', 'ko']
# metric = 'mAP'
# group_name = 'without'
for model in ['mB', 'mT5', 'Llama3', 'Llama2', 'Qwen2']:
    all_data_mAP = []
    for la in langs:
        file_name = f'{root_dir}/results/results_filter/results_table/{la}_mAP_without.xlsx'
        data_table = pd.read_excel(file_name)
        data_table = data_table[['Unnamed: 0', model]]
        for i, (index, row) in enumerate(data_table.iterrows()):
            if row[1][0] == "\\":
                score = float(row[1].split("{")[1].split("±")[0])
            else:
                score = float(row[1].split("±")[0])

            
            d = {
                'Language': la,
                'Country': row[0].split('_')[0],
                'mAP(%)': score
            }
            all_data_mAP.append(d)

    all_data_mAP = pd.DataFrame(all_data_mAP)

    all_data_mWS = []

    langs_WS = ['ar', 'en', 'he', 'ru']
    for la in langs_WS:
        file_name = f'{root_dir}/results/results_filter/results_table/{la}_mWS_without.xlsx'
        data_table = pd.read_excel(file_name)
        data_table = data_table[['Unnamed: 0', model]]
        for i, (index, row) in enumerate(data_table.iterrows()):
            if row[1][0] == "\\":
                score = float(row[1].split("{")[1].split("±")[0])
            else:
                score = float(row[1].split("±")[0])
        # for i, (index, row) in enumerate(data_table.iterrows()):
            d = {
                'Language': la,
                'Country': row[0].split('_')[0],
                'mWS': score
            }
            all_data_mWS.append(d)
            
    all_data_mWS = pd.DataFrame(all_data_mWS)




    cm = plt.get_cmap('Set3')
    sns.set(style="darkgrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1)

    # fig = plt.figure(figsize=(5,6), tight_layout=True)
    sns.barplot(x='Country', y='mAP(%)', hue='Language', 
                    data=all_data_mAP,
                    palette=cm.colors,
                    hatch='//', 
                    edgecolor='black',
                    width=0.65,
                    ax=ax1)

    # ax1.set_ylim(bottom=4)
    # ax1.set_title('(a) mAP comparison')

    sns.barplot(x='Country', y='mWS', hue='Language', 
                    data=all_data_mWS,
                    palette=cm.colors,
                    hatch='//', 
                    edgecolor='black',
                    width=0.65,
                    ax=ax2)

    # ax2.set_ylim(bottom=0.20)
    # ax2.set_title('(b) mWS comparison')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(all_data_mAP['Language'].unique()))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(all_data_mWS['Language'].unique()))


    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(f'/mntcephfs/lab_data/zhouli/personal/FmLAMA/analysis/05-language-comparison/language_compare_{model}.pdf', dpi=300)
