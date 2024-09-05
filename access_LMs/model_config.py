root_model = "/mntcephfs/data/haizhouli/LLM-models"



def prepare_bert_lms_configuration(models_tuples):
    """
    Convert list of models' names to configuration dictionaries

    Args:
    model_tuples -- A tuple of tuples of the models' labels and names on the HF site
    """
    return [
        {
            "lm": "bert",
            "label": model_label,
            "model_name": "bert",
            "bert_model_name": hf_model_name,
            "bert_model_dir": None,
        }
        for model_label, hf_model_name in models_tuples
    ]


def prepare_T5_lms_configuration(models_tuples):
    """
    Convert list of models' names to configuration dictionaries

    Args:
    model_tuples -- A tuple of tuples of the models' labels and names on the HF site
    """
    return [
        {
            "lm": "T5",
            "label": model_label,
            "model_name": "T5",
            "T5_model_name": hf_model_name,
        }
        for model_label, hf_model_name in models_tuples
    ]


AR_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("xlm-roberta_base", "xlm-roberta-base"),
        ("xlm-roberta_large", "xlm-roberta-large"),
        # ("arabert_base_v0.1", "aubmindlab/bert-base-arabertv01"),
        # ("arabert_base_v0.2", "aubmindlab/bert-base-arabertv02"),
        # ("arabert_large_v0.2", "aubmindlab/bert-large-arabertv02"),
        ("arabic_bert_base", "asafaya/bert-base-arabic"),
        ("arabic_bert_large", "asafaya/bert-large-arabic"),
        # ("camel_da", "CAMeL-Lab/bert-base-arabic-camelbert-da"),
        # ("camel_mix", "CAMeL-Lab/bert-base-arabic-camelbert-mix"),
        # ("camel_msa", "CAMeL-Lab/bert-base-arabic-camelbert-msa"),
        # ("qarib", "qarib/bert-base-qarib"),
        # ("darijabert", "Kamel/DarijaBERT"),
        # ("dziribert", "alger-ia/dziribert"),
        # ("gigabert_v3", "lanwuwei/GigaBERT-v3-Arabic-and-English"),
        # ("gigabert_v4", "lanwuwei/GigaBERT-v4-Arabic-and-English"),
        # ("arbert", "UBC-NLP/ARBERT"),
        # ("marbert", "UBC-NLP/MARBERT"),
        # ("marbert_v2", "UBC-NLP/MARBERTv2"),
        # ("morrbert", "otmangi/MorrBERT"),
    )
)+ prepare_T5_lms_configuration((("mT5_base", "google/mt5-base"),))

EN_LMs = prepare_bert_lms_configuration(
    (
        # ("mbert_base_cased", "bert-base-multilingual-cased"),
        # ("mbert_base_uncased", f"{root_model}bert-base-multilingual-uncased"),
        # ("bert-base_cased", "bert-base-cased"),
        # ("bert-base_uncased", f"{root_model}bert-base-uncased"),
        # ("bert-large_cased", "bert-large-cased"),
        # ("bert-large_uncased", f"{root_model}bert-large-uncased"),
        # ("xlm-roberta_base", "xlm-roberta-base"),
        # ("xlm-roberta_large", "xlm-roberta-large"),
        # ("roberta_base", "roberta-base"),
        # ("roberta_large", "roberta-large"),
        # ("Bloom", "/mntcephfs/lab_data/zhouli/model/bloom-560m"),
        ("PhiMini", f"{root_model}/Phi-3.5-mini-instruct"),
        ("PhiMoE", f"{root_model}/Phi-3.5-MoE-instruct"),
        ("Llama3", f"{root_model}/Meta-Llama-3-8B-Instruct"),
        ("Llama2",f"{root_model}/Llama-2-7b-hf"),
        ("Qwen2", f"{root_model}/Qwen2-7B-Instruct"),
        # ("mT5_base", f"{root_model}/mt5-base"), 
        # ("T5_base", f"{root_model}/t5-base"),
        
    
    )
) + prepare_T5_lms_configuration(
    (
        ("mT5_base", f"{root_model}/mt5-base"), 
        ("T5_base", f"{root_model}/t5-base"),
        
        
    )
)

JA_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("tohoku_bert_base", "cl-tohoku/bert-base-japanese"),
        ("tohoku_bert_base_v2", "cl-tohoku/bert-base-japanese-v2"),
        ("tohoku_bert_base_char", "cl-tohoku/bert-base-japanese-char"),
        ("tohoku_bert_base_char_v2", "cl-tohoku/bert-base-japanese-char-v2"),
        ("tohoku_bert_large", "cl-tohoku/bert-large-japanese"),
        ("tohoku_bert_large_char", "cl-tohoku/bert-large-japanese-char"),
        ("japanese_bert_base", "colorfulscoop/bert-base-ja"),
    )
)

KO_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("xlm-roberta_base", "xlm-roberta-base"),
        ("xlm-roberta_large", "model/xlm-roberta-large"),
        ("kykim_bert_base", "bert-kor-base"),
        ("klue_bert_base", "klue-bert-base"),
    )
)+ prepare_T5_lms_configuration((("mT5_base", f"{root_model}mt5-base"),))

ES_LMs = prepare_bert_lms_configuration(
    (
        ("beto_cased", "dccuchile/bert-base-spanish-wwm-cased"),
        ("beto_uncased", "dccuchile/bert-base-spanish-wwm-uncased"),
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
    )
)

ZH_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("xlm-roberta_base", "xlm-roberta-base"),
        ("xlm-roberta_large", "xlm-roberta-large"),
        ("chinese_bert_base", "bert-base-chinese"),
    )
)+ prepare_T5_lms_configuration((("mT5_base", f"{root_model}mt5-base"),))



HE_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("xlm-roberta_base", "xlm-roberta-base"),
        ("xlm-roberta_large", "xlm-roberta-large"),
    )
)+ prepare_T5_lms_configuration((("mT5_base", f"{root_model}mt5-base"),))


RU_LMs = prepare_bert_lms_configuration(
    (
        ("mbert_base_cased", "bert-base-multilingual-cased"),
        ("mbert_base_uncased", "bert-base-multilingual-uncased"),
        ("xlm-roberta_base", "xlm-roberta-base"),
        ("xlm-roberta_large", "xlm-roberta-large"),
        ("rubert-base-cased", "DeepPavlov/rubert-base-cased")
    )
)+ prepare_T5_lms_configuration((("mT5_base", f"{root_model}mt5-base"),))


LANG_TO_LMs = {
    "ar": AR_LMs,
    "en": EN_LMs,
    "es": ES_LMs,
    "ja": JA_LMs,
    "ko": KO_LMs,
    "zh": ZH_LMs,
    'he': HE_LMs,
    'ru': RU_LMs,
}
