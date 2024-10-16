import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the languages and a random matrix for the heatmap
languages = ['en', 'ar', 'he', 'ko', 'ru', 'zh']
root_dir = 'E:/01research/06.probingLLMs/FmLAMA/results/results_code/results_table'

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
ax1, ax2, ax3, ax4 = axes.flatten()

for model in ['mB', 'mT5', 'Llama3', 'Qwen2']:
    data_df = pd.DataFrame(1.00, index=languages, columns=languages)
    for s in languages:
        for m in languages: 
            lang = f'{s}_{m}'
            table_file = f'{root_dir}/{lang}_mAP_without.xlsx'
            data_table = pd.read_excel(table_file)
            data_table = data_table[[model]]
            value = data_table.iloc[-1].values[0].split('±')[0]
            data_df.at[m, s] = float(value)
            # print('a')

    # plt.figure(figsize=(5, 5))

    if model == 'mB':
        sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax1)
        ax1.set_title(model)
        ax1.set_xlabel('Subject language')
        ax1.set_ylabel('Main language')
    elif model == 'mT5':
        sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax2)
        ax2.set_title(model)
        ax2.set_xlabel('Subject language')
        ax2.set_ylabel('Main language')
    
    elif model == 'Qwen2':
        sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax3)
        ax3.set_title(model)
        ax3.set_xlabel('Subject language')
        ax3.set_ylabel('Main language')

    elif model == 'Llama3':
        sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax4)
        ax4.set_title(model)
        ax4.set_xlabel('Subject language')
        ax4.set_ylabel('Main language')

plt.tight_layout()
plt.savefig(f'E:/01research/06.probingLLMs/FmLAMA/analysis/06-code-switch/cs_2.pdf', dpi=300)


