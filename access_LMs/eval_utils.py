# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import torch
import pickle
from tqdm import tqdm
import modules.base_connector as base
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
import utils
import ast

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    T5TokenizerFast,
    PreTrainedTokenizerFast,
)

NUM_BEAMS=1

get_sequence = {
    # Ignore the prompt.
    LlamaTokenizer: lambda seq, input_ids: seq[input_ids.shape[1] :].cpu().tolist(),
    PreTrainedTokenizerFast: lambda seq, input_ids: seq[input_ids.shape[1] :]
    .cpu()
    .tolist(),
    # Ignore the BOS token.
    T5TokenizerFast: lambda seq, _: seq.cpu().tolist()[1:],
}

ids_to_ignore = {
    # Ignore BOS, EOS.
    LlamaTokenizer: [1, 2],
    # Ignore EOS.
    T5TokenizerFast: [1],
    # Ignore EOS.
    PreTrainedTokenizerFast: [11],
}

full_stop = {LlamaTokenizer: 29889, T5TokenizerFast: 5, PreTrainedTokenizerFast: 25}


def get_generation_config(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return GenerationConfig(
        max_new_tokens=50,
        num_beams=NUM_BEAMS,
        do_sample=False,
        output_hidden_states=False,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )

def get_ranking(
    log_probs, sample, masked_tokens_indecies, candidate_objects_dict,
):
    experiment_result = {}
    objects_probabilities = {}
    objects_true = sample['obj_label'] #正确的obj
    
    # Make sure the type of objects_true is a list
    # if type(objects_true) == type(""):
    #     objects_true = [objects_true]
    # sample['obj_label'] = objects_true

    for i, num_masks in enumerate(candidate_objects_dict):#
        # Find the range of the masked indecies
        masked_indecies = masked_tokens_indecies[i] #

        # Extract the probabilities of subwords in the range of the masked tokens
        predictions = log_probs[i][masked_indecies]#

        for object in candidate_objects_dict[num_masks]:#
            object_subword_probabiltiies = [
                prediction[subword_id]
                for subword_id, prediction in zip(
                    candidate_objects_dict[num_masks][object], predictions
                )
            ]
            objects_probabilities[object] = np.mean(object_subword_probabiltiies)#multi-token取平均

    # Sort the dictionary by the probabilities of the candidate objects
    sorted_objects_probabilities = sorted(
        objects_probabilities.items(), key=lambda t: t[1], reverse=True
    )

    # Ranks of all true objects
    ranks = [
        i for i, t in enumerate(sorted_objects_probabilities) if t[0] in objects_true
    ]

    experiment_result['ID'] = sample['url']
    experiment_result['origin'] = sample['origin']
    experiment_result['origin_name'] = sample['origin_name']
    experiment_result['subj_name'] = sample['sub_label']
    experiment_result['obj_name'] = sample['obj_label']
    experiment_result["ranks"] = ranks
    experiment_result["prob_true"] = [(sorted_objects_probabilities[r][0],sorted_objects_probabilities[r][1],r) for r in ranks]
    experiment_result["predicted"] = [t[0] for t in sorted_objects_probabilities]
    experiment_result["probs"] = [t[1] for t in sorted_objects_probabilities]

    return experiment_result


def run_evaluation(args, language, NUM_MASK, candidate_objects_dict, model=None, relation_name=None):
    model_name = args.model_name.title()

    # initialize logging
    log_directory = args.full_logdir
    logger = utils.init_logging(log_directory)
    logger.info("\n" + "model name: {}\n".format(model_name) + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # Load the data of a specific relation
    data = utils.load_jsonl(args.dataset_filename)

    if args.lowercase:
        # lowercase all samples
        logger.info("lowercasing all samples...")
        all_samples = utils.lowercase_samples(data)
    else:
        # keep samples as they are
        all_samples = data

    # Only keep samples having a uuid!
    # TODO: Why do some samples have no id?!
    # TODO: This filtering should be done on saving the file, not on loading!
    # all_samples = [
    #     sample
    #     for sample in all_samples
    #     if "uuid" in sample and sample["sub_label"] and sample["obj_label"]
    # ]

    # Form the prompts for the model
    for sample in all_samples:
        # Add the sample's subject to the template
        if args.bert_model_name in ['xlm-roberta-base', 'xlm-roberta-large', 'roberta-base', 'roberta-large']:
            sample["masked_sentence"] = utils.fill_template_with_values(language,
                args.template.strip(), sample["sub_label"].strip(), base.ROBERTA_MASK, sample['origin_name'], relation_name
            )
        else:
            sample["masked_sentence"] = utils.fill_template_with_values(language,
                args.template.strip(), sample["sub_label"].strip(), base.MASK, sample['origin_name'], relation_name
            )

    samples_batches = utils.batchify(all_samples, args.batch_size)

    # ThreadPool
    num_threads = args.threads
    if num_threads <= 0:
        # use all available threads
        num_threads = multiprocessing.cpu_count()
    pool = ThreadPool(num_threads)
    list_of_results = []

    for i in tqdm(range(len(samples_batches))):
        samples_b = samples_batches[i]
        sentences_b = []
        current_batch_size = len(samples_b)

        # Form multiple versions of the template
        # with different number of masked tokens
        for i, sample in enumerate(samples_b):
            masked_sentences = []
            for num_mask in range(1, NUM_MASK + 1):
                sentence = sample["masked_sentence"]
                if args.bert_model_name in ['xlm-roberta-base', 'xlm-roberta-large', 'roberta-base', 'roberta-large']:
                    sentence = sentence.replace(base.ROBERTA_MASK, base.ROBERTA_MASK * num_mask)
                    # sentence = sentence.replace("><", "> <")
                else:
                    sentence = sentence.replace(base.MASK, base.MASK * num_mask)
                    sentence = sentence.replace("][", "] [")
                # sentence = sentence.replace(base.MASK, base.MASK * num_mask)
                # sentence = sentence.replace("][", "] [")
                masked_sentences.append(sentence)
                sentences_b.append([sentence])
            samples_b[i]["masked_sentences"] = masked_sentences

        #  Fill the masks for all the templates of the current batch， # 
        (
            original_log_probs_tensor,
            tokens_ids_list,
            indecies_of_masked_tokens_list,
        ) = model.get_batch_generation(sentences_b, logger=logger)

        # Group the templates of each sample
        dim_reshape = (
            current_batch_size,
            NUM_MASK,
            original_log_probs_tensor.shape[1],
            original_log_probs_tensor.shape[2],
        )
        original_log_probs_tensor = torch.reshape(
            original_log_probs_tensor, dim_reshape
        )
        #  
        indecies_of_masked_tokens_list = [
            indecies_of_masked_tokens_list[
                sample_index * NUM_MASK : (sample_index + 1) * NUM_MASK
            ]
            for sample_index in range(len(indecies_of_masked_tokens_list))
        ]

        ranking_function_arguments = [
            (
                original_log_probs,
                sample,
                masked_tokens_indecies,
                candidate_objects_dict,
            )
            for sample, original_log_probs, masked_tokens_indecies in zip(
                samples_b, original_log_probs_tensor, indecies_of_masked_tokens_list,
            )
        ]

        # 
        # 
        # experiment_result = get_ranking(ranking_function_arguments[0][0], ranking_function_arguments[0][1], ranking_function_arguments[0][2], ranking_function_arguments[0][3])
        batch_ranking_results = pool.starmap(get_ranking, ranking_function_arguments)

        assert len(batch_ranking_results) == len(samples_b)

        for sample, batch_ranking_result in zip(samples_b, batch_ranking_results):
            element = batch_ranking_result
            # Add the sample results to a list
            list_of_results.append(element)

    pool.close()
    pool.join()

    # dump pickle with the result of the experiment
    # all_results = dict(list_of_results=list_of_results)
    with open("{}/result.pkl".format(log_directory), "wb") as f:
        pickle.dump(list_of_results, f)


def get_T5_ranking(language, model, tokenizer, candidate_answers, prompt, device):
    """Rank the answers according to their probability of filling the masked object.

    Args:
        model: A T5 model
        tokenizer: The model's tokenizer
        candidate_answers: A list of strings for all the candidate answers
        prompt: The manual prompt used to probe the model
        device: The GPU to use or "cpu"

    Returns:
        The answers with their corresponding probabilities.
    """
    #  Replace the span for the object within the template
    if len(language) == 2:
        main_lang = language
    else:
        main_lang = language.split('_')[1]
    if main_lang in ['he', 'ar']:
        input_ids = tokenizer(
        prompt.replace("[2]", "<extra_id_0>"), return_tensors="pt").input_ids
    
    else:
        input_ids = tokenizer(
            prompt.replace("[Y]", "<extra_id_0>"), return_tensors="pt"
        ).input_ids

    # Tokenize the different answers for the span
    answers_probabilities = {}

    # TODO: Make this an argument to the function
    # BATCH_SIZE = 128
    BATCH_SIZE = 64
    for i in tqdm(range(0, len(candidate_answers), BATCH_SIZE)):
        answers = candidate_answers[i : i + BATCH_SIZE]
        labels = tokenizer(
            ["<extra_id_0> " + answer + " <extra_id_1>" for answer in answers],
            return_tensors="pt",
            padding=True,
        ).input_ids

        # Output in the form (Queries, Token Index, Value in Vocab)
        # T5 generates an output in the form "<extra_id_0> 'answer' <extra_id_1>"# output就是
        outputs = model(
            input_ids=torch.concat([input_ids for _ in range(len(answers))]).to(device),
            labels=labels.to(device),
        ).logits

        # Find the ids of the extra tokens
        EXTRA_ID_0_index = tokenizer("<extra_id_0>").input_ids[0]
        EXTRA_ID_1_index = tokenizer("<extra_id_1>").input_ids[0]

        for answer_id in range(len(answers)):
            target_ids = labels[answer_id]
            answer_subword_probabilities = []

            for idx, t_idx in enumerate(target_ids):#获取每个target id的token
                # Skip the first t_idx which is always <extra_id_0>
                if idx == 0:
                    assert t_idx == EXTRA_ID_0_index
                    continue

                #  Stop computing the probabilities just before the <extra_id_1>
                if t_idx == EXTRA_ID_1_index:
                    break

                logits = outputs[answer_id, idx, :]
                probs = logits.cpu().softmax(dim=-1).detach().numpy()
                answer_subword_probabilities.append(-np.log(probs[t_idx]))

            answer_probability = np.mean(answer_subword_probabilities)
            answers_probabilities[answers[answer_id]] = answer_probability

    return answers_probabilities


def calculate_candidate_probability(tokenizer, logits, candidate, candidate_start):
    # Replace [MASK] with the candidate phrase

    # Calculate probability of the candidate phrase
    softmax = torch.nn.Softmax(dim=-1)
    token_probabilities = softmax(logits)

    # Get probabilities of the tokens in the candidate phrase
    candidate_tokens = tokenizer.encode(candidate, add_special_tokens=False)
    candidate_probability = []
    for i, token_id in enumerate(candidate_tokens, start=candidate_start+1):
        token_probability = token_probabilities[i, token_id].item()
        candidate_probability.append(-np.log(token_probability))
        # candidate_probability.append(token_probability)
    
    return np.mean(candidate_probability)



def get_decoder_ranking(sub_label, language, model, tokenizer, candidate_answers, prompt, device):
    """Rank the answers according to their probability of filling the masked object.

    Args:
        model: A T5 model
        tokenizer: The model's tokenizer
        candidate_answers: A list of strings for all the candidate answers
        prompt: The manual prompt used to probe the model
        device: The GPU to use or "cpu"

    Returns:
        The answers with their corresponding probabilities.
    """
    config = get_generation_config(tokenizer)
    answers_probabilities = {}
    if len(language) == 2:
        main_lang = language
    else:
        main_lang = language.split('_')[1]

    if main_lang in ['he', 'ar']:
        prompt_prefix = prompt.split('[2]')[0].strip()
        # prompt_prefix = prompt + '[1]='+sub_label + ', [2]='
    elif main_lang in ['ko']:
        prompt_prefix = prompt.replace(sub_label, "[X]") + '[Y]는 다음과 같을 수 있어요:'
        # prompt_prefix = "\"" +prompt+"\"" + ' 이 문장에서, 만약 [X]=\"'+sub_label + '\", [Y]=\"'

        
    else:
        prompt_prefix = prompt.split('[Y]')[0].strip()
        # prompt_prefix = prompt + '[X]='+sub_label + ', [Y]='
    input_ids = tokenizer.encode(prompt_prefix, return_tensors="pt").to(device)
    # if language in ['he', 'ar']:
    #     input_ids[:, 1:] = torch.flip(input_ids[:, 1:], dims=[1])
    with torch.no_grad():
        model_output = model.generate(
                    input_ids, generation_config=config, output_scores=True
                )

    for i, answer in tqdm(enumerate(candidate_answers)):
        answer_subword_probabilities = []
        if len(language) == 2:
            main_lang = language
        else:
            main_lang = language.split('_')[1]

        if main_lang == 'en':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)[1:]
        elif main_lang == 'zh':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer) # confirm
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[2:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
        elif main_lang == 'ru':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer) # confirm
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)[1:]
        elif main_lang == 'ko':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer) # confirm
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)[1:]

        elif main_lang == 'ar':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer) # confirm
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)[1:]
        elif main_lang == 'he':
            if "Qwen2" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer) # confirm
            elif "Llama-2" in model.name_or_path:
                answer_token_id = tokenizer.encode(answer)[1:]
            elif "Llama-3" in model.name_or_path:
                answer_token_id = tokenizer.encode(" "+answer)[1:]


        for j, t_idx in enumerate(answer_token_id):
            try:
                score = model_output["scores"][j]
            except:
                continue
            probs = torch.softmax(score, 1)
            answer_subword_probabilities.append(-np.log(probs[0][t_idx].cpu().item()))
        answer_probability = np.mean(answer_subword_probabilities)
        answers_probabilities[answer] = answer_probability

      
    return answers_probabilities


