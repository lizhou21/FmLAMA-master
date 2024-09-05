import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

root_dir = '/mntcephfs/lab_data/zhouli/personal/FmLAMA'

# 1. read data
mAP_wo_data = []
dfs_mAP_wo = pd.read_excel(f'{root_dir}/results/results_lang/results_table/en_mAP_without.xlsx')
dfs_mAP_wo = dfs_mAP_wo.iloc[-1]
for i, (index, row) in enumerate(dfs_mAP_wo.items()):
    if i != 0 and index!='Average':
        d = {
            'Model': index,
            'mAP(%)': float(row.split('±')[0]),
            'Cultural_info': 'without'
        }
        mAP_wo_data.append(d)
        
dfs_mAP_w = pd.read_excel(f'{root_dir}/results/results_lang/results_table/en_mAP_withC.xlsx')
dfs_mAP_w = dfs_mAP_w.iloc[-1]
for i, (index, row) in enumerate(dfs_mAP_w.items()):
    if i != 0 and index!='Average':
        d = {
            'Model': index,
            'mAP(%)': float(row.split('±')[0]),
            'Cultural_info': 'with'
        }
        mAP_wo_data.append(d)
mAP_wo_data = pd.DataFrame(mAP_wo_data)



# 2. draw figure
cm = plt.get_cmap('Set3')
sns.set(style="darkgrid")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))


# fig = plt.figure(figsize=(5,6), tight_layout=True)
sns.barplot(x='Model', y='mAP(%)', hue='Cultural_info', 
                 data=mAP_wo_data,
                 palette=cm.colors,
                 hatch='//', 
                 edgecolor='black',
                 width=0.5,
                 ax=ax1)

ax1.set_ylim(bottom=4)
ax1.set_title('(a) mAP comparison')
# plt.savefig(f'/home/nlp/ZL/FmLAMA-master/figure/culture_context/Cultural_info_mAP.pdf', dpi=300)


########################
mWS_wo_data = []
dfs_mWS_wo = pd.read_excel(f'{root_dir}/results/results_lang/results_table/en_mWS_without.xlsx')
dfs_mWS_wo = dfs_mWS_wo.iloc[-1]
for i, (index, row) in enumerate(dfs_mWS_wo.items()):
    if i != 0 and index!='Average':
        d = {
            'Model': index,
            'mWS': float(row.split('±')[0]),
            'Cultural_info': 'without'
        }
        mWS_wo_data.append(d)
        
dfs_mWS_w = pd.read_excel(f'{root_dir}/results/results_lang/results_table/en_mWS_withC.xlsx')
dfs_mWS_w = dfs_mWS_w.iloc[-1]
for i, (index, row) in enumerate(dfs_mWS_w.items()):
    if i != 0 and index!='Average':
        d = {
            'Model': index,
            'mWS': float(row.split('±')[0]),
            'Cultural_info': 'with'
        }
        mWS_wo_data.append(d)
mWS_wo_data = pd.DataFrame(mWS_wo_data)



# 2. draw figure
# cm = plt.get_cmap('Set3')
# sns.set(style="darkgrid")

# fig = plt.figure(figsize=(5,6), tight_layout=True)

sns.barplot(x='Model', y='mWS', hue='Cultural_info', 
                 data=mWS_wo_data,
                 palette=cm.colors,
                 hatch='//', 
                 edgecolor='black',
                 width=0.5,
                 ax=ax2)

ax2.set_ylim(bottom=0.24)
ax2.set_title('(b) mWS comparison')

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'{root_dir}/analysis/01-cultural-info/Cultural_comparison.pdf', dpi=300)



print('a')