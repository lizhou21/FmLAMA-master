# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from copy import deepcopy
from modules import build_model_by_name
import pprint
import json
import sys
from model_config import LANG_TO_LMs
from utils import *
from eval_utils import run_evaluation
from pathlib import Path
import glob
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer


# T5 dependencies
import os
import re
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from eval_utils import get_T5_ranking, get_decoder_ranking
import ast
import torch

def run_T5_experiments(
    relations_templates,
    data_path_pre,
    language,
    device,
    output_path,
    input_param={
        "lm": "T5",
        "label": "mt5_base",
        "model_name": "T5",
        "T5_model_name": "google/mt5-small",
    },
    use_dlama=False,
):
    # Load the model
    model_name = input_param["T5_model_name"]
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    LOGDIR = output_path

    for relation in relations_templates:
        relation_name = relation["relation"]

        # Build the list of candidate objects
        if language == 'ar':
            relation_files_path = str(Path(data_path_pre, "ar_dishes.jsonl"))
        elif language == 'en':
            relation_files_path = str(Path(data_path_pre, "en_dishes.jsonl"))
        elif language == 'he':
            relation_files_path = str(Path(data_path_pre, "he_dishes.jsonl"))
        elif language == 'ko':
            relation_files_path = str(Path(data_path_pre, "ko_dishes.jsonl"))
        elif language == 'ru':
            relation_files_path = str(Path(data_path_pre, "ru_dishes.jsonl"))
        elif language == 'zh':
            relation_files_path = str(Path(data_path_pre, "zh_dishes.jsonl"))

        relation_files = [f for f in glob.glob(relation_files_path)]

        if not relation_files:
            print("Relation {} excluded.".format(relation["relation"]))
            continue

        relation_triples = []
        for file in set(relation_files):
            with open(file, "r") as f:
                relation_triples += [json.loads(line) for line in f]

        # TODO: Augment valid objects with normalized values
        candidate_objects = [
            triple["obj_label"]
            for triple in relation_triples
        ]

        unique_candidate_objects = sorted(
            set([c for c_l in candidate_objects for c in c_l])
        )

        relation_template = relation["template"]
        triples_results = []
        predictions_list=[]
        for triple in tqdm(relation_triples):
            experiment_result = {}
            triple_results = {"sample": triple, "uuid": triple["origin"]}
            sub_label = triple["sub_label"]
            obj_labels = triple['obj_label']
            origin = triple['origin_name']
            
            if relation['relation'].startswith('country'):
                if language == 'he':
                    prompt_for_t4 = relation_template.replace("[1]", sub_label)
                    prompt_for_t4 = prompt_for_t4.replace("[3]", origin)
                else:
                    prompt_for_t4 = relation_template.replace("[X]", sub_label)
                    prompt_for_t4 = prompt_for_t4.replace("[C]", origin)
            else:
                if language == 'he':
                    prompt_for_t4 = relation_template.replace("[1]", sub_label)
                else:
                    prompt_for_t4 = relation_template.replace("[X]", sub_label)
                


            # Find the candidate answers probabilities for this triple
            answers_probabilities = get_T5_ranking(
                language,
                model,
                tokenizer,
                unique_candidate_objects,
                prompt_for_t4,
                device,
            )

            # Sort the answers
            sorted_answers_probabilities = sorted(
                [
                    (answers_probabilities[answer], answer)
                    for answer in answers_probabilities
                ]
            )
            sorted_probablities = [t[0] for t in sorted_answers_probabilities]
            sorted_answers = [t[1] for t in sorted_answers_probabilities]

            # Form the output dictionary for this relation
            ranks = [sorted_answers.index(obj_label) for obj_label in obj_labels]
            probs = [answers_probabilities[obj_label] for obj_label in obj_labels]

            experiment_result['ID'] = triple['url']
            experiment_result['origin'] = triple['origin']
            experiment_result['origin_name'] = triple['origin_name']
            experiment_result['subj_name'] = triple['sub_label']
            experiment_result['obj_name'] = triple['obj_label']
            experiment_result["ranks"] = ranks
            experiment_result["prob_true"] = [(sorted_answers_probabilities[r][1],sorted_answers_probabilities[r][0],r) for r in ranks]
            experiment_result["predicted"] = [t[1] for t in sorted_answers_probabilities]
            experiment_result["probs"] = [t[0] for t in sorted_answers_probabilities]

            predictions_list.append(experiment_result)

   

        log_directory = str(
            Path(
                LOGDIR, "results", input_param["label"], language, relation["relation"],
            )
        )
        os.makedirs(log_directory, exist_ok=True)

        # Dump the results to a .pkl file
        with open("{}/result.pkl".format(log_directory), "wb") as f:
            output_dict = predictions_list
            pickle.dump(output_dict, f)


def run_decoder_experiments(
    relations_templates,
    data_path_pre,
    language,
    device,
    output_path,
    input_param={
        "lm": "T5",
        "label": "mt5_base",
        "model_name": "T5",
        "T5_model_name": "google/mt5-small",
    },
    use_dlama=False,
):
    # Load the model
    model_name = input_param["bert_model_name"]

    
    # if 't5' in model_name:
    #     tokenizer = T5Tokenizer.from_pretrained(model_name)
    #     model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        # model.resize_token_embeddings(len(tokenizer))r
    model.to(device)
    # print(model.device)
    LOGDIR = output_path

    for relation in relations_templates:
        relation_name = relation["relation"]

        # Build the list of candidate objects
        if language == 'ar':
            relation_files_path = str(Path(data_path_pre, "ar_dishes.jsonl"))
        elif language == 'en':
            relation_files_path = str(Path(data_path_pre, "en_dishes.jsonl"))
        elif language == 'he':
            relation_files_path = str(Path(data_path_pre, "he_dishes.jsonl"))
        elif language == 'ko':
            relation_files_path = str(Path(data_path_pre, "ko_dishes.jsonl"))
        elif language == 'ru':
            relation_files_path = str(Path(data_path_pre, "ru_dishes.jsonl"))
        elif language == 'zh':
            relation_files_path = str(Path(data_path_pre, "zh_dishes.jsonl"))

        relation_files = [f for f in glob.glob(relation_files_path)]

        if not relation_files:
            print("Relation {} excluded.".format(relation["relation"]))
            continue

        relation_triples = []
        for file in set(relation_files):
            with open(file, "r") as f:
                relation_triples += [json.loads(line) for line in f]

        # TODO: Augment valid objects with normalized values
        candidate_objects = [
            triple["obj_label"]
            for triple in relation_triples
        ]

        unique_candidate_objects = sorted(
            set([c for c_l in candidate_objects for c in c_l])
        )

        relation_template = relation["template"]
        triples_results = []
        predictions_list=[]
        for triple in tqdm(relation_triples):
            experiment_result = {}
            triple_results = {"sample": triple, "uuid": triple["origin"]}
            sub_label = triple["sub_label"]
            obj_labels = triple['obj_label']
            origin = triple['origin_name']
            
            if relation['relation'].startswith('country'):
                if language == 'he':
                    prompt_for_t4 = relation_template.replace("[1]", sub_label)
                    prompt_for_t4 = prompt_for_t4.replace("[3]", origin)
                    
                else:
                    prompt_for_t4 = relation_template.replace("[X]", sub_label)
                    prompt_for_t4 = prompt_for_t4.replace("[C]", origin)
                    
            else:
                if language == 'he':
                    prompt_for_t4 = relation_template.replace("[1]", sub_label)
                    # fix_prompt = "The prediction of token \"[2]\" would be "
                else:
                    prompt_for_t4 = relation_template.replace("[X]", sub_label)
                    # fix_prompt = "The prediction of token \"[Y]\" would be "
                


            # Find the candidate answers probabilities for this triple
            answers_probabilities = get_decoder_ranking(
                language,
                model,
                tokenizer,
                unique_candidate_objects,
                prompt_for_t4,
                device,
            )

            # Sort the answers
            sorted_answers_probabilities = sorted(
                [
                    (answers_probabilities[answer], answer)
                    for answer in answers_probabilities
                ]
            )
            sorted_probablities = [t[0] for t in sorted_answers_probabilities]
            sorted_answers = [t[1] for t in sorted_answers_probabilities]

            # Form the output dictionary for this relation
            ranks = [sorted_answers.index(obj_label) for obj_label in obj_labels]
            probs = [answers_probabilities[obj_label] for obj_label in obj_labels]

            experiment_result['ID'] = triple['url']
            experiment_result['origin'] = triple['origin']
            experiment_result['origin_name'] = triple['origin_name']
            experiment_result['subj_name'] = triple['sub_label']
            experiment_result['obj_name'] = triple['obj_label']
            experiment_result["ranks"] = ranks
            experiment_result["prob_true"] = [(sorted_answers_probabilities[r][1],sorted_answers_probabilities[r][0],r) for r in ranks]
            experiment_result["predicted"] = [t[1] for t in sorted_answers_probabilities]
            experiment_result["probs"] = [t[0] for t in sorted_answers_probabilities]

            predictions_list.append(experiment_result)
            
            log_directory = str(
                Path(
                    LOGDIR, "results", input_param["label"], language, relation["relation"],
                )
            )
            os.makedirs(log_directory, exist_ok=True)
            
            if len(predictions_list)%10 == 0:
                with open("{}/result_temp.pkl".format(log_directory), "wb") as f:
                    output_dict = predictions_list
                    pickle.dump(output_dict, f)

   

        log_directory = str(
            Path(
                LOGDIR, "results", input_param["label"], language, relation["relation"],
            )
        )
        os.makedirs(log_directory, exist_ok=True)

        # Dump the results to a .pkl file
        with open("{}/result.pkl".format(log_directory), "wb") as f:
            output_dict = predictions_list
            pickle.dump(output_dict, f)






def run_experiments(
    relations_templates,
    data_path_pre,
    language,
    device,
    output_path,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "model_name": "bert",
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_dlama=False,
):
    """
    TODO

    Args:
    - relations_templates: List of strings representing prompts
    - input_param: A model configuration dictionary
    """
    pp = pprint.PrettyPrinter(width=41, compact=True)

    # Load the model
    model = build_model_by_name(
        lm=input_param["model_name"],
        hf_model_name=input_param["bert_model_name"],
        device=device,
    )

    # LOGDIR = "output" if not use_dlama else "output_dlama"
    LOGDIR = output_path
    # Add the configuration parameters into a dictionary
    BASIC_CONFIGURATION_PARAMETERS = {
        "template": "",
        "batch_size": 4,
        "logdir": LOGDIR,
        "lowercase": False,
        "threads": -1,
        "interactive": False,
    }
    # Add model information to the configuration parameters
    BASIC_CONFIGURATION_PARAMETERS.update(input_param)

    for relation in relations_templates:
        relation_name = relation["relation"]

        # Build the list of candidate objects
        # if use_dlama:
        #     # The relation can have multiple subsets
        #     relation_files_path = str(Path(data_path_pre, f"{relation_name}_*.jsonl"))
        # else:
        if language == 'ar':
            relation_files_path = str(Path(data_path_pre, "ar_dishes.jsonl"))
        elif language == 'en':
            relation_files_path = str(Path(data_path_pre, "en_dishes.jsonl"))
        elif language == 'he':
            relation_files_path = str(Path(data_path_pre, "he_dishes.jsonl"))
        elif language == 'ko':
            relation_files_path = str(Path(data_path_pre, "ko_dishes.jsonl"))
        elif language == 'ru':
            relation_files_path = str(Path(data_path_pre, "ru_dishes.jsonl"))
        elif language == 'zh':
            relation_files_path = str(Path(data_path_pre, "zh_dishes.jsonl"))

        relation_files = [f for f in glob.glob(relation_files_path)]

        if not relation_files:
            print("Relation {} excluded.".format(relation["relation"]))
            continue

        relation_triples_raw = []
        for file in set(relation_files):
            with open(file, "r") as f:
                relation_triples_raw += [json.loads(line) for line in f]

        relation_triples = relation_triples_raw
        # for line in relation_triples_raw:
        #     line['obj_label'] = list(ast.literal_eval(line['obj_label']))
        #     relation_triples.append(line)

        
        # TODO: Augment valid objects with normalized values
        # for triple in relation_triples:
        #     triple['obj_label'] = list(ast.literal_eval(triple['obj_label']))


        candidate_objects = [
            triple["obj_label"]
            if type(triple["obj_label"]) == list
            else [triple["obj_label"]]
            for triple in relation_triples
        ]
        candidate_objects = sorted(set(sum(candidate_objects, [])))#将所有的objects进行set并排序

        configuration_parameters = deepcopy(BASIC_CONFIGURATION_PARAMETERS)
        configuration_parameters["template"] = relation["template"]
        configuration_parameters["dataset_filename"] = relation_files_path

        configuration_parameters["full_logdir"] = str(
            Path(
                LOGDIR,
                "results",
                configuration_parameters["label"],
                language,
                relation["relation"],
            )
        )

        pp.pprint(relation)
        print(configuration_parameters)

        # TODO: Why is this parsing done?!
        args = argparse.Namespace(**configuration_parameters)

        max_length = max(
            [len(model.tokenizer.tokenize(obj)) for obj in candidate_objects]
        )# 对每个obj进行分词
        if max_length>10:
            max_length=10

        # Split objects according to their length?!
        dict_num_mask = {}
        for l in range(1, max_length + 1):
            dict_num_mask[l] = {}

        #  Form list of candidates split by length
        for obj in candidate_objects:
            # TODO: What is get_id?
            if len(model.tokenizer.tokenize(obj)) <= 10:
                dict_num_mask[len(model.tokenizer.tokenize(obj))][obj] = model.get_id(obj)

        # Run the experiments
        # TODO: Don't send the whole args dictionary
        run_evaluation(args, language, max_length, dict_num_mask, model=model, relation_name=relation_name)


def run_experiment_on_list_of_lms(
    relations_templates, data_path_pre, language, language_models, use_dlama, device, output_path
):
    for lm in language_models:
        print(lm["label"])
        # try:
        if "T5" in lm["label"]:
            # run_decoder_experiments(
            #     relations_templates,
            #     data_path_pre,
            #     language,
            #     input_param=lm,
            #     use_dlama=use_dlama,
            #     device=device,
            #     output_path=output_path
            # )
            run_T5_experiments(
                relations_templates,
                data_path_pre,
                language,
                input_param=lm,
                use_dlama=use_dlama,
                device=device,
                output_path=output_path
            )
        if lm['label'] in ['Llama3','Llama2', 'Qwen2', 'Bloom', 'PhiMini', 'PhiMoE']:
            run_decoder_experiments(
                relations_templates,
                data_path_pre,
                language,
                input_param=lm,
                use_dlama=use_dlama,
                device=device,
                output_path=output_path
            )
        else:
            run_experiments(
                relations_templates,
                data_path_pre,
                language,
                input_param=lm,
                use_dlama=use_dlama,
                device=device,
                output_path=output_path
            )
        # except Exception as e:
        #     print(e)
        #     print(f'Failed for: {lm["label"]}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", type=str, default="fr", help="language")
    parser.add_argument(
        "--dlama", "-d", action="store_true", help="Evaluate on dlama data",
    )
    parser.add_argument(
        "--rel",
        nargs="*",
        default=None,
        help="Specify a set of Wikidata relations to evaluate on",
    )
    parser.add_argument(
        "--models", nargs="*", default=None, help="A list of model names to probe.",
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory containing jsonl files of tuples",
    )
    parser.add_argument(
        "--templates_file_path",
        required=False,
        default=None,
        help="The path of the templates file",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        default=None,
        help="The path of output",
    )
    parser.add_argument(
        "--device", required=False, default="cpu", help="GPU's device ID to use",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')  # 使用第一个可用的GPU
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        args.device = torch.device('cpu')  # 如果没有GPU可用，则使用CPU
        print("Using CPU")

    language = args.lang
    language_models = LANG_TO_LMs[language]

    if args.models:
        language_models = [lm for lm in language_models if lm["label"] in args.models]

    data_path_pre = str(Path(args.dataset_dir, language))

    # Load the templates file
    # if args.templates_file_path:
    #     relations_templates = load_jsonl(args.templates_file_path)
    # else:
    relations_templates = load_jsonl(args.templates_file_path)

    if args.rel:
        relations_templates = [
            relation_template
            for relation_template in relations_templates
            if relation_template["relation"] in args.rel
        ]

    run_experiment_on_list_of_lms(
        relations_templates,
        str(Path(args.dataset_dir, language)),
        language,
        language_models,
        use_dlama=args.dlama,
        device=args.device,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
