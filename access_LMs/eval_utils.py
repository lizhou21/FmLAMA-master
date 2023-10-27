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


def get_ranking(
    log_probs, sample, masked_tokens_indecies, candidate_objects_dict,
):
    experiment_result = {}
    objects_probabilities = {}
    objects_true = list(ast.literal_eval(sample['obj_label'])) #正确的obj

    # Make sure the type of objects_true is a list
    if type(objects_true) == type(""):
        objects_true = [objects_true]

    for i, num_masks in enumerate(candidate_objects_dict):#获取每个长度设置下的obj list
        # Find the range of the masked indecies
        masked_indecies = masked_tokens_indecies[i] #一个mask位置时的mask位置

        # Extract the probabilities of subwords in the range of the masked tokens
        predictions = log_probs[i][masked_indecies]#获取当前位置下的prediction logits

        for object in candidate_objects_dict[num_masks]:#获取当前每个候选词的预测概率
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

    experiment_result["ranks"] = ranks
    experiment_result["prob_true"] = [sorted_objects_probabilities[r] for r in ranks]
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
                args.template.strip(), sample["sub_label"].strip(), base.ROBERTA_MASK, sample['origin'], relation_name
            )
        else:
            sample["masked_sentence"] = utils.fill_template_with_values(language,
                args.template.strip(), sample["sub_label"].strip(), base.MASK, sample['origin'], relation_name
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
        # with different number of masked tokens使用不同数量的掩码令牌形成模板的多个版本
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

        #  Fill the masks for all the templates of the current batch， # 返回logits[sentence_num, sen_len, vocab_size],token_list:每个句子的token id list, mask_indeces
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
        #  Group the indecies of masked tokens of each sample掩码的位置
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

        # Run the ranking computation in parallel for the samples in the batch!为批处理中的样本并行运行排名计算!
        # ranking_function_arguments中每个元素有4个部分，第一个部分logits[10,seq_len,vocab_size],第二个sample,三个mask index, 第四个每个长度对应的candidate set
        # experiment_result = get_ranking(ranking_function_arguments[0][0], ranking_function_arguments[0][1], ranking_function_arguments[0][2], ranking_function_arguments[0][3])
        batch_ranking_results = pool.starmap(get_ranking, ranking_function_arguments)

        assert len(batch_ranking_results) == len(samples_b)

        for sample, batch_ranking_result in zip(samples_b, batch_ranking_results):
            element = {
                "sample": sample,
                "uuid": sample["origin"],
                "masked_topk": batch_ranking_result,
            }
            # Add the sample results to a list
            list_of_results.append(element)

    pool.close()
    pool.join()

    # dump pickle with the result of the experiment
    all_results = dict(list_of_results=list_of_results)
    with open("{}/result.pkl".format(log_directory), "wb") as f:
        pickle.dump(all_results, f)


def get_T5_ranking(model, tokenizer, candidate_answers, prompt, device):
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
        # T5 generates an output in the form "<extra_id_0> 'answer' <extra_id_1>"
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

            for idx, t_idx in enumerate(target_ids):
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
