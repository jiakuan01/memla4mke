import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip

sys.path.append(os.getcwd())

from AMIG.knowledge_neurons import garns, initialize_model_and_tokenizer, model_type


def read_prompts_for_identification(data_path: str, language: str, eval_id_path=None):
    """
    Read prompts, answers and relations from dataset.
    Returns:
        - prompts: List[List[str]], each item is an example
        - ground_truths: List[List[str]]
        - relations: relation of each example
        - case_ids: example id of each example
    """
    with open(data_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    if eval_id_path is not None:
        with open(eval_id_path, "r") as f:
            eval_ids = [int(x.strip()) for x in f.readlines()]
        eval_ids = sorted(eval_ids)
        datas = [datas[x] for x in eval_ids]
    prompts = []
    ground_truths = []
    relations = []
    case_ids = []

    for item in datas:
        gt = []
        paraphrase = []
        paraphrase.append(
            item["edit"][language]["prompt"].format(item["edit"][language]["subject"])
        )
        paraphrase.extend(item["paraphrase_prompts"][language])
        for _ in range(len(paraphrase)):
            gt.append(item["edit"][language]["target_true"])

        prompts.append(paraphrase)
        ground_truths.append(gt)
        relations.append(item["relation_description"])
        case_ids.append(item["example_id"])
    return prompts, ground_truths, relations, case_ids


def load_prev_if(neuron_dir: str, lang: str):
    path = neuron_dir + f"{lang}_neurons.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            neurons_all_examples = json.load(f)
    else:
        neurons_all_examples = []
    return neurons_all_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--lm", type=str, default="ai-forever/mGPT")
    parser.add_argument("--use_cluster", action="store_true", default=False)
    parser.add_argument("--data_file", type=str, default="data/MLiKE_v2.json")
    parser.add_argument("--eval_id_file", type=str, default="data/test_ids_2000.txt")
    parser.add_argument("--baseline_vector_dir", type=str, default="baseline_vector_qwen2.5_1204/")
    parser.add_argument("--neuron_save_dir", type=str, default="neurons_qwen/")
    parser.add_argument("--adp_threshold", type=float, default=0.3)
    args = parser.parse_args()

    prompts, gts, relations, case_ids = read_prompts_for_identification(args.data_file, args.language, eval_id_path=args.eval_id_file)
    model_name = "/mnt/usercache/huggingface/Qwen2.5-7B-Instruct/"
    print(model_name)
    model, tokenizer = initialize_model_and_tokenizer(model_name)
    if 'mgpt' in model_name.lower():
        model = model.cuda()
    kn = garns(model, tokenizer, model_type=model_type(model_name))
    # baseline_vector_dir = args.baseline_vector_dir + f"average_{args.language}/"  # baseline_vector_dir/average_en/
    baseline_vector_dir = args.baseline_vector_dir + f"average_{args.language}"  # baseline_vector_dir/average_en/
    os.makedirs(args.neuron_save_dir, exist_ok=True)
    
    # neurons_for_all_examples = []
    neurons_for_all_examples = load_prev_if(args.neuron_save_dir, args.language)
    curr_length = len(neurons_for_all_examples)
    if curr_length > 0:
        prompts = prompts[curr_length:]
        gts = gts[curr_length:]
        relations = relations[curr_length:]
        case_ids = case_ids[curr_length:]

    for prompt, gt, rel, case_id in tzip(prompts, gts, relations, case_ids):
        assert len(set(gt)) == 1
        print(f"Case id: {case_id}")
        # fr 153, 1379
        if case_id in [153] and args.language == 'fr':
            neurons = []
        else:
            neurons = kn.get_refined_neurons(
                prompt,
                gt[0],
                batch_size=4,  # 10  5 for zh
                steps=20,
                coarse_adaptive_threshold=args.adp_threshold,
                baseline_vector_path=baseline_vector_dir,
            )
        neurons_for_all_examples.append(
            {"example_id": case_id, "relation_name": rel, "neurons": neurons}
        )
        with open(args.neuron_save_dir + f"{args.language}_neurons.json", "w") as f:
            json.dump(neurons_for_all_examples, f, indent=4)
