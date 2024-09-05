import json


def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        ret = []
        for i, item in enumerate(f.readlines()):
            record = json.loads(item)
            ret.append(record)
    return ret

root_dir = "/mntcephfs/lab_data/zhouli/personal/FmLAMA"

f = open(f'{root_dir}/data/country_info.json', 'r')
continent_info = f.read()
f.close()
continent_info = json.loads(continent_info)



# 1. 统计各ingredient的country info
ingredient_info = {}

English_dish = read_data(f"{root_dir}/data/data_lang/en/en_dishes.jsonl")
for dish in English_dish:
    for ingre in dish['obj_label']:
        if ingre not in ingredient_info.keys():
            ingredient_info[ingre]={}
            ingredient_info[ingre]['count'] = 1
            ingredient_info[ingre]['origin'] = [dish['origin']]
            ingredient_info[ingre]['continent'] = [continent_info[dish['origin']]]
        else:
            ingredient_info[ingre]['count'] = ingredient_info[ingre]['count'] + 1
            if dish['origin'] not in ingredient_info[ingre]['origin']:
                ingredient_info[ingre]['origin'].append(dish['origin'])
            if continent_info[dish['origin']] not in ingredient_info[ingre]['continent']:
                ingredient_info[ingre]['continent'].append(continent_info[dish['origin']])


json_str = json.dumps(ingredient_info)
with open(f'{root_dir}/data/ingredient_info_English.json', 'w') as json_file:
    json_file.write(json_str)

print('a')
