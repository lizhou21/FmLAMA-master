import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

country_lang = {
    # 'Italy': 'Italy',
    # 'United States of America': 'U.S.',
    # 'Turkey': 'Turkey',
    # 'Japan': 'Japan',
    # 'France': 'France',
    # 'United Kingdom': 'U.K.',
    # 'Mexico': 'Mexico',
    # 'India': 'India',
    # 'Germany': 'Germany',
    # 'People\'s Republic of China': 'China',
    # 'Iran': 'Iran',
    'Greece': 'Greece',
    'Spain': 'Spain',
    'Russia': 'Russia',
    'aggregated': 'ALL',
}

# Define the languages and a random matrix for the heatmap
languages = ['en', 'ar', 'he', 'ko', 'ru', 'zh']
root_dir = 'FmLAMA/results/results_code/results_table'


for country in country_lang.values():
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.5))
    ax1, ax2, ax3, ax4 = axes.flatten()

    for model in ['mB', 'mT5', 'Llama3', 'Qwen2']:
        data_df = pd.DataFrame(1.00, index=languages, columns=languages)
        for s in languages:
            for m in languages: 
                lang = f'{s}_{m}'
                table_file = f'{root_dir}/{lang}_mAP_without.xlsx'
                data_table = pd.read_excel(table_file)
                
                row_name = country + '_mAP'
                value = data_table.loc[data_table['Unnamed: 0'] == row_name, model].values[0]
                if value[0] == "\\":
                    value = value.split('{')[1].split('±')[0]
                else:
                    value = value.split('±')[0]
                data_df.at[m, s] = float(value)

                # print('a')

        # plt.figure(figsize=(5, 5))

        if model == 'mB':
            heatmap = sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax1)
            ax1.set_title(model)
            ax1.set_xlabel('Subject language')
            ax1.set_ylabel('Main language')
            # for text in heatmap.texts:
            #     text.set_rotation(45)

        elif model == 'mT5':
            heatmap = sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax2)
            ax2.set_title(model)
            ax2.set_xlabel('Subject language')
            ax2.set_ylabel('Main language')
            # for text in heatmap.texts:
            #     text.set_rotation(45)
        
        elif model == 'Qwen2':
            heatmap = sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax3)
            ax3.set_title(model)
            ax3.set_xlabel('Subject language')
            ax3.set_ylabel('Main language')
            # for text in heatmap.texts:
            #     text.set_rotation(45)

        elif model == 'Llama3':
            heatmap = sns.heatmap(data_df, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=True, ax=ax4)
            ax4.set_title(model)
            ax4.set_xlabel('Subject language')
            ax4.set_ylabel('Main language')
            # for text in heatmap.texts:
            #     text.set_rotation(45)

    plt.tight_layout()
    plt.savefig(f'FmLAMA/analysis/06-code-switch/cs_{country}.pdf', dpi=300)


