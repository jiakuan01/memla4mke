from functools import reduce
import json
import random
import itertools
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Union, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.evaluator import DataForEvaluation


def show_inner_parameters(model: AutoModelForCausalLM):
    for n, p in model.named_parameters():
        print(f"{n}\tshape: {p.shape}")
        

def prepare_model_tokenizer(
    model_name: str,
    use_cuda: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    if "llama" in model_name.lower():
        return model, tok
    elif "mgpt" in model_name.lower():
        tok.truncation_side = "right"
        return model, tok
    else:
        raise NotImplementedError(f"We currently do not support {model_name} yet.")
        
        
def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_eval_ids(eval_id_path: str):
    """Read test ids from the specified file.

    Args:
        eval_id_path (str): File name with respect to the test ids.

    Returns:
        Set: The Set of test ids.
    """
    eval_ids = []
    with open(eval_id_path, "r") as f:
        for line in f.readlines():
            eval_ids.append(int(line.strip()))
    return sorted(eval_ids)


def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def get_attributes(x, attrib):
    for attr in attrib.split("."):
        x = getattr(x, attr)
    return x


def get_parameters(model: AutoModelForCausalLM, p_name: str):
    for n, p in model.named_parameters():
        if n == p_name:
            return p
    raise LookupError(p_name)


def get_cross_neurons(neuron_dir: str, langs: List[str], test_id_path=None) -> List[List[List]]:
    """
    `neuron_dir`: The parent directory of the neurons file (in `json` format)
    """
    all_lgs_neurons = []
    for lg in langs:
        filename = neuron_dir + f"{lg}_neurons.json"
        lg_neurons = get_language_specific_neurons(filename, test_id_path=test_id_path)
        # Transform each neuron position into a tuple
        lg_neurons = [
            map(tuple, each_sample) for each_sample in lg_neurons
        ]  # List[List[tuple]]
        all_lgs_neurons.append(lg_neurons)

    # intersection
    def _intersect_pair_language(left, right: List[List[Tuple]]) -> List[Set]:
        return [set(x) & set(y) for x, y in zip(left, right)]

    cross_neurons = reduce(_intersect_pair_language, all_lgs_neurons)
    return cross_neurons


def get_language_specific_neurons(
    filename: str, test_id_path: Optional[str] = None
) -> List[List[List[int]]]:
    """
    Read Language-Specific Neurons from a pre-saved json file.\\
    Please make sure that the pre-saved file only contains test examples if you do not specify `test_id_path`
    
    Returns: List[List[List[int]]]
    """
    with open(filename, "r") as f:
        datas = json.load(f)
    if test_id_path is not None:
        with open(test_id_path, "r") as fr:
            test_ids = [int(x.strip()) for x in fr.readlines()]
            test_ids = sorted(test_ids)
        idxs = [item['example_id'] for item in datas if item['example_id'] in test_ids]
        neurons = [item["neurons"] for item in datas if item['example_id'] in test_ids]
    else:
        neurons = [x["neurons"] for x in datas]
    return neurons


def format_neurons(
    kn_pos: Union[Union[Set[Tuple], List[Set[Tuple[int]]]], List[List[List[int]]]]
):
    """
    Input: a set for an example, or a List of Set for a batch.\\
    A set contains some tuples. Each tuple represents a neuron position.
    
    Format neuron data into: Dict[int, List[int]] \\
    Returns a dict with the expected format, where key represents layer index,\\
    and value represents neuron indices for this layer.
    """

    def format_for_one_case(k_pos: Union[Set[Tuple[int]], List[List[int]]]):
        # For one case.
        each_layer = defaultdict(list)
        for l_idx, act_id in k_pos:
            each_layer[l_idx].append(act_id)
        for k in each_layer:
            each_layer[k] = list(set(each_layer[k]))
        return each_layer

    if isinstance(kn_pos, Set):
        # logging.info("Format neurons for a single case.")
        return format_for_one_case(kn_pos)
    # For batch case.
    # logging.info("Format neurons for batch cases.")
    res = {}
    for n in kn_pos:
        n = format_for_one_case(n)
        for k, v in n.items():
            if k not in res.keys():
                res[k] = v
            else:
                res[k].extend(v)
                res[k] = list(set(res[k]))
    return res


def forward_for_one_fact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prefix: str,
    target: str,
    inference: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Given a `prefix` and an expected `target`, calculate the probability and negative log-likelihood (NLL) of `target`

    Note that `target` should starts with a whitespace.
    """
    sentence = prefix + target
    target_ids = tok(target, return_tensors="pt")["input_ids"][0].to(model.device)
    input_toks = tok(sentence, return_tensors="pt").to(model.device)
    prefix_len = len(tok.encode(prefix))

    if inference:
        with torch.no_grad():
            logits = model(**input_toks).logits
    else:
        logits = model(**input_toks).logits
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    target_mask = torch.tensor(-100, device=model.device).repeat(
        input_toks["input_ids"].shape
    )  # [B, SL]
    target_mask[0, prefix_len - 1 : prefix_len - 1 + len(target_ids)] = target_ids
    indices = torch.where(target_mask != -100, target_mask, 0).unsqueeze(2)
    mask = (target_mask != -100).float()
    values = torch.gather(log_probs, dim=-1, index=indices).squeeze(2)  # [B=1, SL]
    ll = mask * values
    log_ps = ll.sum(1)  # log probabilities
    prob = log_ps.exp()
    nll = -log_ps
    return {
        "nll": nll,
        "target_prob": prob,
    }


def forward_batch_facts(model: AutoModelForCausalLM, tok: AutoTokenizer, prefixes: List[str], targets: List[str]):
    assert len(prefixes) == len(targets)
    sentences = [p + a for p, a in zip(prefixes, targets)]
    encodings = tok(sentences, padding=True, return_tensors='pt').to(model.device)   # left padding
    batch_size, seq_len = encodings['input_ids'].shape
    target_ids = torch.tensor([-100]).repeat(batch_size, seq_len)
    for i in range(batch_size):
        _t_ids = tok(targets[i], return_tensors='pt')['input_ids'][0]
        target_ids[i, -1 - len(_t_ids) : -1] = _t_ids
    
    logits = model(**encodings).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    
    target_ids = target_ids.to(log_probs.device)
    l_mask = (target_ids != -100).float()
    ll = torch.gather(log_probs, 2, torch.where(target_ids != -100, target_ids, 0).unsqueeze(2)).squeeze(2)
    temp = -(ll * l_mask).sum(1)
    avg_prob = torch.exp(-temp).mean()
    nll = -(ll * l_mask).sum(1).mean()
    return {
        'nll': nll,
        'target_prob': avg_prob,
    }


def prepare_inputs_for_evaluation(
    batch_samples: List[Dict], eval_lang: str
) -> DataForEvaluation:
    """Given a batch of edit requests, extract structured data for evaluation.

    Args:
        `batch_samples` `(List[Dict])`: batch edits
        `eval_lang` `(str)`: evaluation language

    Returns:
        `DataForEvaluation`: Structured data for evaluation.
    """
    edit_prompts = []
    para_prompts = []
    ngh_prompts = []
    ngh_answers = []
    multi_hop_questions = []
    multi_hop_answers = []
    target_news, target_trues = [], []
    for x in batch_samples:
        subj = x["edit"][eval_lang]["subject"]
        prompt = x["edit"][eval_lang]["prompt"].format(subj)
        edit_prompts.append(prompt)
        para_prompts.append(x["paraphrase_prompts"][eval_lang])
        target_news.append(x["edit"][eval_lang]["target_new"])
        target_trues.append(x["edit"][eval_lang]["target_true"])
        ngh_prompts.append(x["neighborhood_prompts"][eval_lang])
        ngh_answers.append(x["neighborhood_answers"][eval_lang])
        if "multi_hop_questions" in x:
            mh_questions = x["multi_hop_questions"][eval_lang]  # List[str]
            mh_ans = x["multi_hop_new_answer"][eval_lang]  # str
            multi_hop_questions.append(mh_questions)
            multi_hop_answers.append([mh_ans for _ in mh_questions])
    para_tgt_news = []
    para_tgt_trues = []
    for para, tgt_new, tgt_true in zip(para_prompts, target_news, target_trues):
        para_tgt_news.extend([tgt_new for _ in para])
        para_tgt_trues.extend([tgt_true for _ in para])
    para = list(itertools.chain(*para_prompts))

    return DataForEvaluation(
        edit_prompts=edit_prompts,
        edit_target_news=target_news,
        edit_target_trues=target_trues,
        para_prompts=para,
        para_target_news=para_tgt_news,
        para_target_trues=para_tgt_trues,
        ngh_prompts=ngh_prompts,
        ngh_answers=ngh_answers,
        multi_hop_questions=multi_hop_questions,
        multi_hop_answers=multi_hop_answers,
    )
