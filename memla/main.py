import sys
import os
import itertools
import json
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import torch.fx
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

sys.path.append(".")

import utils.util_funcs as utf
import utils.evaluator as evaluator
from memla.editor import NeuronBasedEditor


log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="memla_llama", version_base=None)
def main(config: DictConfig):
    torch.manual_seed(config.seed)
    log.info(f"Edit language = {config.edit_lang}")
    log.info(f"Evaluation languages = {config.eval_langs}")
    log.info(
        f"The following languages are used to find the multilingual neurons: {config.cross_langs}"
    )
    # Load LM
    model_name = os.getenv("MODEL_NAME_OR_PATH")
    model_type = os.getenv("MODEL_TYPE")
    model, tok = utf.prepare_model_tokenizer(model_name)

    # Read data
    log.info(f"Load data from {config.data_fname}")
    with open(config.data_fname, "r", encoding="utf-8") as f:
        datas = json.load(f)
    test_ids = utf.get_eval_ids(config.test_ids_fname)

    test_set = [datas[x] for x in test_ids]
    print(f"test_ids:\n{test_ids}")

    cross_neurons = utf.get_cross_neurons(config.neuron_dir, config.cross_langs, test_id_path=config.test_ids_fname)
    specific_neurons = utf.get_language_specific_neurons(
        config.neuron_dir + f"{config.edit_lang}_neurons.json", test_id_path=config.test_ids_fname
    )
    log.info(f"Reading neurons from {config.neuron_dir}")

    print(f"testset: {len(test_set)}, c_neuron {len(cross_neurons)}, s_neuron {len(specific_neurons)}")
    edit_to_edit = {}
    edit_to_others = []
    for eval_lang in config.eval_langs:
        es_list, ps_list, ns_list = [], [], []
        qs_list = []
        processed = 0
        for batch_edits, batch_neurons, batch_lang_neurons in zip(
            utf.chunks(test_set, n=1),
            utf.chunks(cross_neurons, n=1),
            utf.chunks(specific_neurons, n=1),
        ):
            # We first prepare inputs for evaluation
            edit_prompts = []
            para_prompts = []
            ngh_prompts = []
            ngh_answers = []
            multi_hop_answers = []
            multi_hop_questions = []
            target_news, target_trues = [], []
            for x in batch_edits:
                subj = x["edit"][eval_lang]["subject"]
                prompt = x["edit"][eval_lang]["prompt"].format(subj)
                edit_prompts.append(prompt)
                para_prompts.append(x["paraphrase_prompts"][eval_lang])
                ngh_prompts.append(x["neighborhood_prompts"][eval_lang])
                target_news.append(x["edit"][eval_lang]["target_new"])
                target_trues.append(x["edit"][eval_lang]["target_true"])
                ngh_answers.append(x["neighborhood_answers"][eval_lang])
                if "multi_hop_questions" in x:
                    mh_questions = x["multi_hop_questions"][eval_lang]  # List[str]
                    mh_ans = x["multi_hop_new_answer"][eval_lang]  # str
                    multi_hop_questions.append(mh_questions)
                    multi_hop_answers.append([mh_ans for _ in mh_questions])

            # Adjust the paraphrase inputs for evaluation
            expanded_tgt_news = []
            expanded_tgt_trues = []
            for para, tgt_new, tgt_true in zip(para_prompts, target_news, target_trues):
                expanded_tgt_news.extend([tgt_new for _ in para])
                expanded_tgt_trues.extend([tgt_true for _ in para])
            para = list(itertools.chain(*para_prompts))

            # Adjust cross-neurons into expected format, i.e., Dict[int, List]
            fmt_neurons = utf.format_neurons(batch_neurons)
            fmt_lang_neurons = utf.format_neurons(batch_lang_neurons)
            # Make sure that they have no intersections
            bad_keys = []
            for k in fmt_lang_neurons:
                if k not in fmt_neurons:
                    continue
                fmt_lang_neurons[k] = list(
                    set(fmt_lang_neurons[k]) - set(fmt_neurons[k])
                )
                if len(fmt_lang_neurons[k]) == 0:
                    bad_keys.append(k)
            for k in bad_keys:
                fmt_lang_neurons.pop(k)
            print(f"specific neurons: {fmt_lang_neurons}")
            print(f"cross neurons: {fmt_neurons}")
            # Do the edit
            
            editor = NeuronBasedEditor(model, config, fmt_lang_neurons, fmt_neurons, model_type=model_type)
            edited_model, weights_copy = editor.fit_on(
                model,
                tok,
                [
                    {
                        "example_id": record["example_id"],
                        **record["edit"][config.edit_lang],
                    }
                    for record in batch_edits
                ],
            )

            # Edit Success evaluation
            edit_matches = evaluator.assess_reliability(
                edited_model, tok, edit_prompts, target_news, target_trues
            )
            es_list.extend(edit_matches)

            # Paraphrase evaluation
            para_matches = evaluator.assess_reliability(
                edited_model, tok, para, expanded_tgt_news, expanded_tgt_trues
            )
            ps_list.extend(para_matches)

            # Neighborhood evaluation
            ns_list.extend(
                evaluator.assess_soft_match(edited_model, tok, ngh_prompts, ngh_answers)
            )

            # Multi-hop questions evaluation
            if len(multi_hop_questions) != 0:
                qs_match = evaluator.assess_soft_match(
                    edited_model, tok, multi_hop_questions, multi_hop_answers
                )
                qs_list.extend(qs_match)

            # We print the result during edit process
            processed += len(batch_edits)
            print(f"Processed {processed}.")
            print(f"Edit in {config.edit_lang} and Evaluate in {eval_lang}:")
            print(f"Edit Success = {np.mean(es_list)}")
            print(f"Paraphrase Score = {np.mean(ps_list)}")
            print(f"Neighborhood Score = {np.mean(ns_list)}")
            print(f"MQ Score = {np.mean(qs_list) if len(qs_list) else -1}")

            # Restore weights
            print(f"len(weights_copy) = {len(weights_copy)}")
            with torch.no_grad():
                for k, v in weights_copy.items():
                    utf.get_attributes(model, k)[...] = v.to(model.device)

        # Write results into log file
        log.info(f"Results of {config.edit_lang}-->{eval_lang}:")
        log.info(f"ES = {np.mean(es_list)}")
        log.info(f"PS = {np.mean(ps_list)}")
        log.info(f"NS = {np.mean(ns_list)}")
        log.info(f"QS = {np.mean(qs_list)}\n")

        # Collect results
        res = {
            "ES": np.mean(es_list),
            "PS": np.mean(ps_list),
            "NS": np.mean(ns_list),
            "QS": np.mean(qs_list),
        }
        if config.edit_lang == eval_lang:
            edit_to_edit = res
        else:
            edit_to_others.append(res)

    # Summarize results
    if len(edit_to_edit) != 0:
        log.info(f"Overall result of {config.edit_lang}-->{config.edit_lang}:")
        log.info(
            f"ES = {edit_to_edit['ES']}\tPS = {edit_to_edit['PS']}\tNS = {edit_to_edit['NS']}\tQS = {edit_to_edit['QS']}\n"
        )
    if len(edit_to_others) != 0:
        log.info(f"Overall result of {config.edit_lang}-->others:")
        avg_res = {
            "ES": np.mean([x["ES"] for x in edit_to_others]),
            "PS": np.mean([x["PS"] for x in edit_to_others]),
            "NS": np.mean([x["NS"] for x in edit_to_others]),
            "QS": np.mean([x["QS"] for x in edit_to_others]),
        }
        sum_res = {
            "ES": sum([x["ES"] for x in edit_to_others]),
            "PS": sum([x["PS"] for x in edit_to_others]),
            "NS": sum([x["NS"] for x in edit_to_others]),
            "QS": sum([x["QS"] for x in edit_to_others]),
        }
        log.info(
            f"Sum_ES = {sum_res['ES']}\tSum_PS = {sum_res['PS']}\tSum_NS = {sum_res['NS']}\tSum_QS = {sum_res['QS']}"
        )
        log.info(
            f"Avg_ES = {avg_res['ES']}\tAvg_PS = {avg_res['PS']}\tAvg_NS = {avg_res['NS']}\tAvg_QS = {avg_res['QS']}"
        )


if __name__ == "__main__":
    main()
