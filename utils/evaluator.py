from typing import List, Dict, NewType
from dataclasses import dataclass

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class DataForEvaluation:
    edit_prompts: List[str]
    edit_target_news: List[str]
    edit_target_trues: List[str]
    # Paraphrase
    para_prompts: List[str]
    para_target_news: List[str]
    para_target_trues: List[str]
    # Neighborhood
    ngh_prompts: List[List[str]]
    ngh_answers: List[List[str]]
    # Multi-hop
    multi_hop_questions: List[List[str]]
    multi_hop_answers: List[List[str]]


def compute_for_one_sentence(model, tok, prefix, target):
    target = " " + target
    sentence = prefix + target
    target_ids = tok(target, return_tensors="pt")["input_ids"][0]

    input_toks = tok(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**input_toks).logits
    probs = torch.softmax(logits, dim=-1)
    prefix_len = len(tok.encode(prefix))
    ps = []
    for i, cur_tok in enumerate(target_ids):
        ps.append(probs[0, prefix_len - 1 + i, cur_tok].item())
    return np.prod(ps)


def get_prediction_for_one_prefix(model, tok, prefix: str, steps: int):
    """
    `steps = len(tok.encode(' ' + answer))`

    Returns:
        Predicted token ids.
    """
    predict_tokens = []
    input_toks = tok(prefix, return_tensors="pt").to(model.device)

    for i in range(steps):
        if i > 0:
            # Re-tokenization
            input_toks = tok(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**input_toks).logits
        probs = torch.softmax(logits, dim=-1)
        argmax_id = probs[0, -1, :].argmax().item()
        predict_tokens.append(argmax_id)
        completion_str = tok.decode(argmax_id)
        # print(f"completion_str = ${completion_str}$")
        prefix += completion_str
        # print(f"prefix = ${prefix}$")
    return predict_tokens


def get_target_probabilities(model, tokenizer, prompts, targets):
    assert len(prompts) == len(targets)
    target_probs = []
    for prompt, target in zip(prompts, targets):
        prob = compute_for_one_sentence(model, tokenizer, prompt, target)
        target_probs.append(prob)
    return target_probs


def assess_reliability(model, tokenizer, prompts, target_news, target_trues):
    target_new_probs = get_target_probabilities(model, tokenizer, prompts, target_news)
    target_true_probs = get_target_probabilities(
        model, tokenizer, prompts, target_trues
    )
    matches = [pn > pt for pn, pt in zip(target_new_probs, target_true_probs)]
    return matches


def assess_soft_match(
    model, tokenizer, ngh_prompts: List[List[str]], ngh_answers: List[List[str]]
):
    f1s = []
    for prompts_one_sample, targets_one_sample in zip(ngh_prompts, ngh_answers):
        f1s.extend(
            partial_match_one_sample(
                model, tokenizer, prompts_one_sample, targets_one_sample
            )
        )
    return f1s


def partial_match_one_sample(
    model, tok, prefixes: List[str], answers: List[str]
) -> List[float]:
    """Partial soft match for one sample containing several prompts.

    Args:
        model (LanguageModel): LLM
        tok (AutoTokenizer): Tokenizer
        prefixes (List[str]): Neighborhood prompts
        answers (List[str]): Neighborhood answers.

    Returns:
        `List[float]`: F1 scores for each neighborhood prompt.
    """
    assert len(prefixes) == len(answers)
    f1s = []
    for p, a in zip(prefixes, answers):
        pred_ids = get_prediction_for_one_prefix(
            model, tok, p, steps=len(tok.encode(" " + a))
        )
        ans_ids = tok.encode(" " + a)
        print(f"Answer ids: {ans_ids}")
        print(f"Predicted ids: {pred_ids}")
        f1s.append(calc_f1(ans_ids, pred_ids))
    return f1s


def calc_f1(ref, pred):
    ref, pred = set(ref), set(pred)
    t = ref & pred
    if len(ref) == 0 and len(pred) == 0:
        raise ValueError("Empty prediction")
    if len(t) == 0:
        return 0.0
    p = len(t) / len(pred)
    r = len(t) / len(ref)
    f1 = 2 * p * r / (p + r)
    return f1


def compute_ppl(
    predictions,
    model,
    tokenizer,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    """device has been set
    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    """

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:  # Dont go
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:  # Dont go
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}