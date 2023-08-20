import os
import sys
import json
import time
import logging
import modules.base_connector as base
import glob


def load_jsonl(filename):
    data = []
    for file in glob.glob(str(filename)):
        with open(file, "r") as f:
            data += [json.loads(line) for line in f]
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def lowercase_samples(samples):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentence"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentence"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


def batchify(data, batch_size):
    # sort to group together sentences with similar length
    data = sorted(data, key=lambda k: len(" ".join(k["masked_sentence"]).split()))

    # Split data into batches
    list_samples_batches = [
        data[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]

    return list_samples_batches


def fill_template_with_values(template, subject_label, object_label):
    """Fill template with a subject/object from a triple"""
    template = template.replace("[X]", subject_label)
    template = template.replace("[Y]", object_label)

    return template

def fill_template_with_values_ro(template, subject_label, object_label):
    """Fill template with a subject/object from a triple"""
    template = template.replace("[X]", subject_label)
    template = template.replace(" [Y]", object_label)

    return template
