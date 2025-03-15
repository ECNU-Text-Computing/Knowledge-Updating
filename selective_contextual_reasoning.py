from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import argparse
from retrieve import RetrieveTool
from summary import SummaryTool
from icl import EditTool
from prepare_requests import load_data, _prepare_requests


def prepare_memory(datapath, num=None):
    memory = []
    with open(datapath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        for entry in data:
            sentence = entry["sentence"] if "sentence" in entry else entry["prompt"]
            memory.append(sentence)
    if num is not None:
        memory = memory[:num]
    return memory

def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)

    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys


def summary_metrics(all_metrics):
    mean_metrics = dict()
    for key in ["rewrite_acc", "rephrase_acc"]:
        if key in all_metrics[0].keys():
            mean_metrics[key] = np.mean([metric[key] for metric in all_metrics])
    for key in ["rewrite_summary_acc", "rephrase_summary_acc"]:
        if key in all_metrics[0].keys():
            mean_metrics[key] = np.mean([metric[key] for metric in all_metrics])

    for key in ["locality", "portability"]:
        if key in all_metrics[0].keys() and all_metrics[0][key] != {}:
            mean_metrics[key] = dict()
            for lkey in get_all_acc_keys(all_metrics):

                # metrics = [metric[key][lkey] for metric in all_metrics if lkey in metric[key].keys()]
                metrics = [metric[key][lkey] for metric in all_metrics if lkey in metric.get(key, {})]

                if len(metrics) > 0:
                    sublist_means = [np.mean(sublist) for sublist in metrics]
                    mean_metrics[key][lkey] = np.mean(sublist_means)

    print("Metrics Summary: ", mean_metrics)


def edit(retrieve_tool, k, summary_tool, edit_tool, memory, requests, summary_prompt, answer_prompt, date_type,
         model_type, eval_metric, summary=True):
    all_metrics = []

    # 得到所有memory的embedding表示
    retrieve_tool.get_sent_embeddings(memory, BSZ=32)
    score_list = []
    for i, request in enumerate(tqdm(requests, total=len(requests))):

        ret = {}

        if "prompt" in request:
            question = request["prompt"]

            target_new = request["target_new"]

            # retrieve
            retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)

            # summary
            if summary:

                related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)
            else:

                related_knowledge = retrieved_facts[0]

            if related_knowledge is not None:
                rewrite_summary_acc = 1.0
            else:
                rewrite_summary_acc = 0

            if k == 1 and related_knowledge is not None:
                related_knowledge = retrieved_facts[0]

            # edit
            rewrite_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge, locality=False,
                                                vanilla_genration=True, answer_prompt=answer_prompt,
                                                eval_metric=eval_metric)

            ret["rewrite_acc"] = rewrite_acc
            ret["rewrite_summary_acc"] = rewrite_summary_acc

        if "rephrase_prompt" in request:
            question = request["rephrase_prompt"]
            target_new = request["target_new"]

            # retrieve
            retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)

            # summary

            if summary:
                related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)
            else:
                related_knowledge = retrieved_facts[0]

            if related_knowledge is not None:
                rephrase_summary_acc = 1.0
            else:
                rephrase_summary_acc = 0

            if k == 1 and related_knowledge is not None:
                related_knowledge = retrieved_facts[0]

            # edit
            rephrase_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge, locality=False,
                                                 vanilla_genration=True, answer_prompt=answer_prompt,
                                                 eval_metric=eval_metric)

            ret["rephrase_acc"] = rephrase_acc
            ret["rephrase_summary_acc"] = rephrase_summary_acc

        if "portability" in request and any(request["portability"]):

            ret['portability'] = {}

            for portability_key in request['portability'].keys():
                ret['portability'][f"{portability_key}_acc"] = []
                ret['portability'][f"{portability_key}_summary_acc"] = []
                for j in range(len(request['portability'][portability_key]["prompt"])):
                    question = request['portability'][portability_key]["prompt"][j]
                    target_new = request['portability'][portability_key]["ground_truth"][j]

                    # retrieve
                    retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)

                    # summary
                    if summary:
                        related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt,
                                                                 max_new_tokens=50)
                    else:
                        related_knowledge = retrieved_facts[0]

                    if related_knowledge is not None:
                        portability_summary_acc = 1.0
                    else:
                        portability_summary_acc = 0

                    if k == 1 and related_knowledge is not None:
                        related_knowledge = retrieved_facts[0]

                    # edit
                    portability_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                            locality=False, vanilla_genration=True,
                                                            answer_prompt=answer_prompt, eval_metric=eval_metric)

                    ret['portability'][f"{portability_key}_acc"].append(portability_acc)
                    ret['portability'][f"{portability_key}_summary_acc"].append(portability_summary_acc)

        if "locality" in request and any(request["locality"]):

            ret['locality'] = {}

            for locality_key in request['locality'].keys():
                ret['locality'][f"{locality_key}_acc"] = []
                ret['locality'][f"{locality_key}_summary_acc"] = []
                for j in range(len(request['locality'][locality_key]["prompt"])):
                    question = request['locality'][locality_key]["prompt"][j]
                    target_new = request['locality'][locality_key]["ground_truth"][j]

                    # retrieve
                    retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)

                    # summary
                    if summary:
                        related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt,
                                                                 max_new_tokens=50)
                    else:
                        related_knowledge = retrieved_facts[0]

                    if related_knowledge is None:
                        locality_summary_acc = 1.0
                    else:
                        locality_summary_acc = 0
                    if k == 1 and related_knowledge is not None:
                        related_knowledge = retrieved_facts[0]

                    # edit
                    locality_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                         locality=True, vanilla_genration=True,
                                                         answer_prompt=answer_prompt, eval_metric=eval_metric)

                    ret['locality'][f"{locality_key}_acc"].append(locality_acc)
                    ret['locality'][f"{locality_key}_summary_acc"].append(locality_summary_acc)

        all_metrics.append(ret)

    summary_metrics(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--retriever_type', type=str, default="contriever-ms")
    parser.add_argument('--data_type', type=str, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ds_size', type=int, default=None)
    parser.add_argument('--eval_metric', type=str, default=None, choices=['token_em', 'contain'])
    parser.add_argument('--summary', action='store_true', default=False)

    args = parser.parse_args()

    model_type = args.model_type
    k = args.k
    device = args.device
    data_type = args.data_type
    num = args.ds_size
    eval_metric = args.eval_metric
    summary = args.summary
    retriever_type = args.retriever_type

    model_name = None
    if model_type == "llama2":
        model_name = "./model/llama-2-7b-chat"  # model path
    elif model_type == "llama3":
        model_name = "./model/llama3.1-8b-instruct" # model path
    elif model_type == "mistral":
        model_name = "./model/Mistral-7B-Instruct-v0.1" # model path
    assert model_name is not None

    retriever = None
    retriever_tok = None
    if retriever_type == "contriever-ms":
        retriever_path = "./retriever/contriever-msmarco"
        retriever = AutoModel.from_pretrained(retriever_path).to(device)
        retriever_tok = AutoTokenizer.from_pretrained(retriever_path)
    elif retriever_type == "contriever":
        retriever_path = "./retriever/contriever"
        retriever = AutoModel.from_pretrained(retriever_path).to(device)
        retriever_tok = AutoTokenizer.from_pretrained(retriever_path)
    elif retriever_type == "ance":
        retriever_path = "./retriever/ance"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)

    summary_model = model
    summary_tok = tok

    data_dir = None
    memory = None
    if data_type == "zsre":
        data_dir = "./data/ZsRE/ZsRE-test-all.json"
        memory = prepare_memory("./data/ZsRE/ZsRE-test-all-sentence.json", num=num)
    elif data_type == "counterfact":
        data_dir = "./data/wiki_counterfact/test_cf.json"
        memory = prepare_memory("./data/wiki_counterfact/wiki_counterfact-test-all-sentence.json", num=num)
    assert data_dir is not None

    data = load_data(data_dir, data_type, eval_metric)
    if num is None:
        requests = _prepare_requests(
            prompts=data["prompts"],
            target_new=data["target_new"],

            ground_truth=data["ground_truth"],
            rephrase_prompts=data["rephrase_prompts"],
            locality_inputs=data["locality_inputs"],
            portability_inputs=data["portability_inputs"]
        )
    else:
        requests = _prepare_requests(
            prompts=data["prompts"],
            target_new=data["target_new"],

            ground_truth=data["ground_truth"],
            rephrase_prompts=data["rephrase_prompts"],
            locality_inputs=data["locality_inputs"],
            portability_inputs=data["portability_inputs"]
        )[:num]

    summary_prompt = None
    answer_prompt = None

    if k == 1:
        summary_prompt = """Given a set of facts and a question, return the fact that best matches the core knowledge asked in the question. If the question cannot be answered with the facts, return "no relevant fact."\n\nFacts: {retrieved_facts}\nQuestion: {question}\n\nOutput:"""

    else:
        summary_prompt = """Given a set of facts and a question, return the fact (the entire sentence) that best matches the core knowledge asked in the question. If the question cannot be answered with the facts, return "no relevant fact."\n\nFacts: {retrieved_facts}\nQuestion: {question}\n\nOutput:"""

    # answer1
    answer_prompt = """Answer the question based on the Fact.\n\nFact: {knowledge}\nQuestion: {question}\nAnswer:"""


    retrieve_tool = RetrieveTool(retriever_type, retriever_path, retriever, retriever_tok, device)
    summary_tool = SummaryTool(summary_model, summary_tok, device)
    edit_tool = EditTool(model, tok, device)
    edit(retrieve_tool, k, summary_tool, edit_tool, memory, requests, summary_prompt, answer_prompt, data_type,
         model_type, eval_metric, summary=summary)

    # nohup python edit_rag.py --model_type --retriever_type --data_type --k --device --eval_metric --ds_size --summary








