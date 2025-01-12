import json

import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())
from AMIG.knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)
from AMIG.knowledge_neurons.patch import register_hook


class Baseline_average_KnowledgeNeurons(KnowledgeNeurons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_avg_baseline_vector(
        self, dataset_path, output_path, lang, layer_idx, step_size, eval_ids_path=None,
    ):
        # Load dataset
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        # Load eval ids
        if eval_ids_path is not None:
            with open(eval_ids_path, "r") as f:
                eval_ids = [int(x.strip()) for x in f.readlines()]
            eval_ids = sorted(eval_ids)
            dataset = [dataset[x] for x in eval_ids]
            dataset = dataset[:100]     # qian 100
        # Extract prompts
        prompts = []
        for item in dataset:
            prompt: str = item["edit"][lang]["prompt"]
            prompt = prompt.format(item["edit"][lang]["subject"])
            prompts.append(prompt)
        # Obtain the embeddings
        embeddings = []
        for j in range(0, len(prompts), step_size):
            encodings = self.tokenizer(prompts[j], return_tensors="pt").to(self.device)
            mask_idx = -1
            _, baseline_activations = self.get_baseline_with_activations(
                encodings, layer_idx, mask_idx
            )
            embeddings.append(baseline_activations.detach().cpu().numpy().tolist())
        avg_baseline_vector = np.mean(embeddings, axis=0).tolist()
        print(
            f"avg_baseline_vector length is {len(avg_baseline_vector[0])}"
        )  # 8192
        # Save to file
        with open(output_path, "w") as f:
            json.dump(avg_baseline_vector, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--data_path", type=str, default="data/MLiKE_v2.json")
    parser.add_argument(
        "--eval_path", type=str, default="data/test_ids_2000.txt"
    )
    args = parser.parse_args()

    # model_name = "/netcache/huggingface/Meta-Llama-3-8B/"
    model_name = "ai-forever/mGPT"
    model_name = "/mnt/publiccache/huggingface/Qwen2.5-7B-Instruct/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = initialize_model_and_tokenizer(model_name)


    save_dir = f"baseline_vector_qwen2.5_1204/average_{args.lang}/"
    os.makedirs(save_dir, exist_ok=True)

    baseline_kn_ave = Baseline_average_KnowledgeNeurons(
        model, tokenizer, model_type=model_type(model_name), device=device
    )
    for i in range(baseline_kn_ave.n_layers()):
        baseline_kn_ave.generate_avg_baseline_vector(
            dataset_path=args.data_path,
            output_path=save_dir + f"layer{i}.json",
            lang=args.lang,
            layer_idx=i,
            step_size=200,
            eval_ids_path=args.eval_path,
        )
