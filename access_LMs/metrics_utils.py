import glob
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
import ast
from tqdm import tqdm
#  TODO: Load this list from the constants.py file within dlama
DOMAINS = [
    "sports",
    "politics",
    "music",
    "cinema_and_theatre",
    "history",
    "science",
    "geography",
]


def load_predicate_results(results_dir, relation_id, model_name, lang):
    """
    Return a dataframe of a model's predictions for a specific relation in a specific language.
    """

    def compute_probs(logits):
        exps = np.exp(logits)
        return exps / exps.sum()

    # Load the pickle file
    file_path = str(Path(results_dir, model_name, lang, relation_id, "result.pkl"))
    with open(file_path, "rb") as f:
        results = pickle.load(f)




    predictions_list = []

    for sample in tqdm(results["list_of_results"]):
        # if sample['sample']['sub_uri'] == 'Q217059':
        #     print('a')
        # Remove the "_REGION" from the sample name
        origin = sample["sample"]["origin"]
        # sample_id = re.sub(r"_REGION", "", sample["uuid"])
        # fields = sample_id.split("_")

        # # Infer the domain from the sample ID
        # domains = [d for d in DOMAINS if d in sample_id]
        # domain = domains[0] if domains else "general"

        # region = (
        #     normalize_region_name(fields[-2])
        #     if not "SOUTH_AMERICA" in sample_id
        #     else "SOUTH_AMERICA"
        # )
        # sample_id = int(fields[-1])
        
        if len(sample["masked_topk"]["ranks"]) == 0:
            continue
        # try:
        rank = sample["masked_topk"]["ranks"][0]#取最高的
        # except Exception as e:
        #     from pprint import pprint

        #     pprint(sample)
        #     exit(0)
        probs = compute_probs(sample["masked_topk"]["probs"])

        predictions_list.append(
            {
                "origin": origin,
                "predicate": relation_id,
                "id": sample['sample']['url'],
                "rank": rank,
                "subject": sample["sample"]["sub_label"],
                "valid_objects": list(ast.literal_eval(sample["sample"]['obj_label'])),
                "predictions": sample["masked_topk"]["predicted"],
                "probabilities": probs,
            }
        )

    df = pd.DataFrame(predictions_list)

    return df


def load_model_results(results_dir, model_name, lang, relation_predicates=None):

    #  Infer the predicates if not provided
    if not relation_predicates:
        relation_predicates = [
            Path(p).name
            for p in glob.glob(str(Path(results_dir, model_name, lang, "*")))
        ]

    predicates_results_df = [
        load_predicate_results(results_dir, predicate, model_name, lang)
        for predicate in relation_predicates
    ]

    return pd.concat(predicates_results_df)


def compute_P_at_1(df):
    """Compute P@1 score for a dataframe of predictions."""
    number_correct = int((df["rank"] == 0).sum())
    n_samples = int(df.shape[0])
    return round(100 * (number_correct / n_samples), 3)

def compute_P_at_5(df):
    """Compute P@1 score for a dataframe of predictions."""
    number_correct = int((df["rank"] <= 5).sum())
    n_samples = int(df.shape[0])
    return round(100 * (number_correct / n_samples), 3)

def compute_P_scores_at_5(
    df, aggregation_method="split_by_region", regions=None, skip_predicates=["P527"]
):
    """Aggregate scores from a results dataframe for different relation predicates."""

    assert aggregation_method in [
        "all",
        "split_by_predicate",
        "split_by_region",
        "split_by_domain",
    ]

    # Skip predicates
    if skip_predicates:
        df = df[~df["predicate"].isin(skip_predicates)]

    if aggregation_method == "all":
        return {"P@5_aggregated": compute_P_at_5(df), "Support_aggregated": df.shape[0]}

    #  Infer regions in case they are not provided
    if not regions:
        regions = sorted(df["origin"].unique())
    else:
        assert all([region in sorted(df["origin"].unique()) for region in regions])

    if aggregation_method == "split_by_region":
        scores = {}
        for region in regions:
            region_df = df[df["origin"] == region]

            scores[f"P@5_{region}"] = compute_P_at_5(region_df)
            scores[f"Support_{region}"] = region_df.shape[0]
        scores["P@5_aggregated"] = compute_P_at_5(df)
        scores["Support_aggregated"] = df.shape[0]
        return scores

    relation_predicates = [
        p for p in natsorted(df["predicate"].unique()) if p not in skip_predicates
    ]

    if aggregation_method == "split_by_predicate":
        results = []
        for relation_predicate in relation_predicates:
            scores = {}
            for region in regions:
                region_df = df[
                    (df["predicate"] == relation_predicate) & (df["region"] == region)
                ]
                scores[f"P@5_{region}"] = compute_P_at_5(region_df)
                scores[f"Support_{region}"] = region_df.shape[0]

            relation_df = df[df["predicate"] == relation_predicate]
            scores["P@5_aggregated"] = compute_P_at_5(relation_df)
            scores["Support_aggregated"] = relation_df.shape[0]

            scores["predicate"] = relation_predicate
            scores["domain"] = relation_predicate
            results.append(scores)

        return results

    domains = [d for d in sorted(df["domain"].unique())]

    if aggregation_method == "split_by_domain":
        results = []
        for domain in domains:
            domain_df = df[df["domain"] == domain]
            for relation_predicate in relation_predicates:
                scores = {}
                for region in regions:
                    predicate_region_df = domain_df[
                        (domain_df["predicate"] == relation_predicate)
                        & (domain_df["region"] == region)
                    ]
                    if predicate_region_df.shape[0]:
                        scores[f"P@5_{region}"] = compute_P_at_5(predicate_region_df)
                        scores[f"Support_{region}"] = predicate_region_df.shape[0]

                relation_df = domain_df[domain_df["predicate"] == relation_predicate]
                if relation_df.shape[0]:
                    scores["P@5_aggregated"] = compute_P_at_5(relation_df)
                    scores["Support_aggregated"] = relation_df.shape[0]

                    scores["predicate"] = relation_predicate
                    scores["domain"] = domain
                    results.append(scores)

            # TODO: Avoid copy-pasting here!
            scores = {}
            for region in regions:
                domain_region_df = domain_df[(domain_df["region"] == region)]
                if domain_region_df.shape[0]:
                    scores[f"P@5_{region}"] = compute_P_at_5(domain_region_df)
                    scores[f"Support_{region}"] = domain_region_df.shape[0]

            if domain_df.shape[0]:
                scores["P@5_aggregated"] = compute_P_at_5(domain_df)
                scores["Support_aggregated"] = domain_df.shape[0]

                scores["predicate"] = "Aggregated"
                scores["domain"] = domain
                results.append(scores)

        return results



def compute_P_scores_at_1(
    df, aggregation_method="split_by_region", regions=None, skip_predicates=["P527"]
):
    """Aggregate scores from a results dataframe for different relation predicates."""

    assert aggregation_method in [
        "all",
        "split_by_predicate",
        "split_by_region",
        "split_by_domain",
    ]

    # Skip predicates
    if skip_predicates:
        df = df[~df["predicate"].isin(skip_predicates)]

    if aggregation_method == "all":
        return {"P@1_aggregated": compute_P_at_1(df), "Support_aggregated": df.shape[0]}

    #  Infer regions in case they are not provided
    if not regions:
        regions = sorted(df["origin"].unique())
    else:
        assert all([region in sorted(df["origin"].unique()) for region in regions])

    if aggregation_method == "split_by_region":
        scores = {}
        for region in regions:
            region_df = df[df["origin"] == region]

            scores[f"P@1_{region}"] = compute_P_at_1(region_df)
            scores[f"Support_{region}"] = region_df.shape[0]
        scores["P@1_aggregated"] = compute_P_at_1(df)
        scores["Support_aggregated"] = df.shape[0]
        return scores

    relation_predicates = [
        p for p in natsorted(df["predicate"].unique()) if p not in skip_predicates
    ]

    if aggregation_method == "split_by_predicate":
        results = []
        for relation_predicate in relation_predicates:
            scores = {}
            for region in regions:
                region_df = df[
                    (df["predicate"] == relation_predicate) & (df["region"] == region)
                ]
                scores[f"P@1_{region}"] = compute_P_at_1(region_df)
                scores[f"Support_{region}"] = region_df.shape[0]

            relation_df = df[df["predicate"] == relation_predicate]
            scores["P@1_aggregated"] = compute_P_at_1(relation_df)
            scores["Support_aggregated"] = relation_df.shape[0]

            scores["predicate"] = relation_predicate
            scores["domain"] = relation_predicate
            results.append(scores)

        return results

    domains = [d for d in sorted(df["domain"].unique())]

    if aggregation_method == "split_by_domain":
        results = []
        for domain in domains:
            domain_df = df[df["domain"] == domain]
            for relation_predicate in relation_predicates:
                scores = {}
                for region in regions:
                    predicate_region_df = domain_df[
                        (domain_df["predicate"] == relation_predicate)
                        & (domain_df["region"] == region)
                    ]
                    if predicate_region_df.shape[0]:
                        scores[f"P@1_{region}"] = compute_P_at_1(predicate_region_df)
                        scores[f"Support_{region}"] = predicate_region_df.shape[0]

                relation_df = domain_df[domain_df["predicate"] == relation_predicate]
                if relation_df.shape[0]:
                    scores["P@1_aggregated"] = compute_P_at_1(relation_df)
                    scores["Support_aggregated"] = relation_df.shape[0]

                    scores["predicate"] = relation_predicate
                    scores["domain"] = domain
                    results.append(scores)

            # TODO: Avoid copy-pasting here!
            scores = {}
            for region in regions:
                domain_region_df = domain_df[(domain_df["region"] == region)]
                if domain_region_df.shape[0]:
                    scores[f"P@1_{region}"] = compute_P_at_1(domain_region_df)
                    scores[f"Support_{region}"] = domain_region_df.shape[0]

            if domain_df.shape[0]:
                scores["P@1_aggregated"] = compute_P_at_1(domain_df)
                scores["Support_aggregated"] = domain_df.shape[0]

                scores["predicate"] = "Aggregated"
                scores["domain"] = domain
                results.append(scores)

        return results

