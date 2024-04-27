
import json
import glob
import os
import argparse
import time
from fastchat.model import load_model, get_conversation_template
import torch
from QA_prompt import run_eval
from knockknock import slack_sender

webhook_url = "your_webhook" # you can also delete knockknock usage


def load_jsonl(filename):
    data = []
    for file in glob.glob(str(filename)):
        with open(file, "r") as f:
            data += [json.loads(line) for line in f]
    return data

def fill_template_with_values(language, template, subject_label, origin, relation_name):
    """Fill template with a subject/object from a triple"""

    if relation_name.startswith("country"):
        if language=='he':
            template = template.replace("[3]", origin)
            template = template.replace("[1]", subject_label)
            # template = template.replace("[2]", object_label)
        else:
            template = template.replace("[C]", origin)
            template = template.replace("[X]", subject_label)
            # template = template.replace("[Y]", object_label)
    else:
        if language=='he':
            # template = template.replace("[3]", origin)
            template = template.replace("[1]", subject_label)
            # template = template.replace("[2]", object_label)
        else:
            # template = template.replace("[C]", origin)
            template = template.replace("[X]", subject_label)
            # template = template.replace("[Y]", object_label)

    return template

def wirte_data(list, file_path):
    
    with open(file_path, 'w', encoding='utf-8') as  f:
        for l in list:
            json_str = json.dumps(l, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')

@slack_sender(webhook_url=webhook_url, channel="model_knockknock")
def main():
    prompt_instruct = {
        'en':{
            'one': ' Please fill the sentence with only one entity.',
            'full': ' Please fill the sentence.'
        },
        'zh':{
            'one': '请用一个词填充这个句子。',
            'full': '请填充这个句子。'
        }
    }

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True
    )
    args = parser.parse_args()
    num_gpus_total = 1
    num_gpus_per_model = 1
    model_map = {"vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
            "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
            "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
            "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
            "SeaLLM-7B-Chat": "SeaLLMs/SeaLLM-7B-Chat",
            "FlagAlpha/Llama2-Chinese-13b-Chat":"FlagAlpha/Llama2-Chinese-13b-Chat",
            "THUDM/chatglm2-6b":"THUDM/chatglm2-6b",
            "baichuan-inc/baichuan-7B":"baichuan-inc/baichuan-7B"}
    model_name = args.model

    params = {'temperature': 0.7, 'max_new_token': 1024, 'num_choices': 1, 'batch_size': 5}
    print(f"Loading model {model_name}...")
    model, tokenizer = load_model(
        model_map[args.model],
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory="",
        load_8bit=False,
        cpu_offloading=False,
        debug=False
    )

    root_dir = '/ceph/hpc/home/eujongc/recipe_probing/'
    datasets = ['en_en', 'en_zh', 'zh_en', 'zh_zh']

    for data in datasets:
        # 1. read dishes
        data_file = root_dir + data + '/' + data + '_dishes.jsonl'
        dishes = []
        with open(data_file, "r") as f:
            dishes += [json.loads(line) for line in f]
            
        # 2. obtain templates
        lang_temp = data.split('_')[0]
        relations_templates = load_jsonl('/ceph/hpc/home/eujongc/recipe_probing/templates/WC/'+lang_temp +'_temp.json')
        for template in relations_templates:
            template_name = template['relation']
            template_content = template['template']
            

            for instruct_name, instruct_content in prompt_instruct[lang_temp].items():
                
                save_dir = root_dir + 'probing_results_WC/' + data + '/' + instruct_name + '/' + template_name + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            
                
                # 3. start to generate prompt for each dish
                for d in dishes:
                    prompt = fill_template_with_values(lang_temp,
                                                    template_content.strip(), 
                                                    d["sub_label"].strip(), 
                                                    d['origin_name'], 
                                                    template_name) + instruct_content
                    print(prompt)
                    # 4. probing LLMs with the generated prompt, provide the answer of LLMs, 
                    # and parse the predicted object list
                    '''
                    please write your probing code here
                    
                    '''
                    d['answer'] = run_eval(model, tokenizer, model_name, prompt, params)
                    # d['answer'] = ""
                    # d['predicted_objects'] = [obj1, obj2, ……]
                    
                
                # 4. Save the probing results
                # print('a')
                wirte_data(dishes, save_dir + model_name + '_' + data + '_' + 'dishes_results.jsonl', )
        # print('a')
    print('total time: ', time.time() - start_time)
    return "\n  model: {}, task: {}".format(model_name, "wc")


if __name__ == "__main__":
    main()