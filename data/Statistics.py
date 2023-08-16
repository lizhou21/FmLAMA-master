import pandas as pd
import json


FmLAMA = pd.read_csv('FmLAMA.csv')

count_by_value = FmLAMA['origin'].value_counts().to_frame()
count_by_lang = FmLAMA['lang'].value_counts()
count_by_two = FmLAMA.groupby(['origin', 'lang']).size().reset_index(name='count')

with pd.ExcelWriter('Statistics.xlsx') as writer:
    # 将df1保存到第一个工作表
    count_by_value.to_excel(writer, sheet_name='Sheet1', index=True)
    # 将df2保存到第二个工作表
    count_by_lang.to_excel(writer, sheet_name='Sheet2', index=True)
    count_by_two.to_excel(writer, sheet_name='Sheet3', index=False)

print('a')