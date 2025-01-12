from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

import utils.util_funcs as utf


class CustomizedHooker:
    def __init__(self, params_this_layer, model_type: str) -> None:
        self.model_type = model_type
        self.lang_specific_b: torch.Tensor = params_this_layer["lang_specific_b"]
        self.lang_specific_a: torch.Tensor = params_this_layer["lang_specific_a"]

        self.lang_independent_b: torch.Tensor = params_this_layer["lang_independent_b"]
        self.lang_independent_a: torch.Tensor = params_this_layer["lang_independent_a"]

        self.has_specific_neurons: bool = params_this_layer["has_specific_neurons"]
        self.has_independent_neurons: bool = params_this_layer[
            "has_independent_neurons"
        ]
        self.specific_mask: torch.Tensor = params_this_layer["specific_mask"]
        self.independent_mask: torch.Tensor = params_this_layer["independent_mask"]
        self.constraint_loss = 0.0

    def __call__(self, module, module_in, module_out):
        # module_in is a tuple with ONE element
        x = module_in[0]
        delta = self.get_delta()
        # if 'llama' in self.model_type:
        #     delta = delta.T
        module_out += (x @ delta.to(x.device))
        return module_out

    def get_delta(self) -> torch.Tensor:
        assert any([self.has_specific_neurons, self.has_independent_neurons])
        if self.has_specific_neurons and not self.has_independent_neurons:
            # print(f"sb.device {self.lang_specific_b.device} sa.device {self.lang_specific_a.device} sm.device {self.specific_mask.device}")
            delta = (
                (self.lang_specific_b)
                @ (self.lang_specific_a)
            ) * self.specific_mask.T 
        elif not self.has_specific_neurons and self.has_independent_neurons:
            # print("Independent")
            # print(self.lang_independent_b, self.lang_independent_a)
            delta = (
                self.lang_independent_b @ self.lang_independent_a
            ) * self.independent_mask.T 
        else:
            universal_delta = (
                self.lang_independent_b @ self.lang_independent_a
            ) * self.independent_mask.T
            specific_delta = (
                (self.lang_specific_b)
                @ (self.lang_specific_a)
            ) * self.specific_mask.T
            delta = (specific_delta + universal_delta)
        self.constraint_loss = torch.norm(delta)
        return delta


class NeuronBasedEditor:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        config,
        specific_neurons: Dict[int, List],
        independent_neurons: Dict[int, List],
        model_type: str,
    ) -> None:
        self.model_type = model_type
        if "llama" in model_type:
            self.mlp_c_proj_module = "model.layers.{}.mlp.down_proj"
            self.n_layers = model.config.num_hidden_layers
        elif "mgpt" in model_type:
            self.mlp_c_proj_module = "transformer.h.{}.mlp.c_proj"
            self.n_layers = model.config.n_layer
        self.config = config
        self.specific_neurons = specific_neurons
        self.independent_neurons = independent_neurons

        self.specific_b_matrices = []
        self.specific_a_matrices = []

        self.independent_b_matrices = []
        self.independent_a_matrices = []

        self.hooks = []
        self.handles = []
        self.edit_layers = []
        self.alphas = []
        for layer_idx in range(self.n_layers):
            self.register_hook_or_not(model, layer_idx)
        self.parameters = (
            self.specific_b_matrices
            + self.specific_a_matrices
            + self.independent_b_matrices
            + self.independent_a_matrices
        )

    def register_hook_or_not(self, model: AutoModelForCausalLM, layer_idx: int):
        w_name = self.mlp_c_proj_module.format(layer_idx) + ".weight"
        w = utf.get_attributes(model, w_name)
        has_specific_neurons = False
        has_independent_neurons = False
        if layer_idx in self.specific_neurons:
            has_specific_neurons = True
            specific_ids = self.specific_neurons[layer_idx]
            r1 = len(specific_ids)  # rank
            specific_b = torch.zeros(
                (w.shape[1], self.config.n_rank * r1), requires_grad=True, device=w.device,
            )
            specific_a = torch.randn(
                (self.config.n_rank * r1, w.shape[0]), requires_grad=True, device=w.device,
            )
            specific_mask = torch.zeros_like(w,)
            if 'mgpt' in self.model_type:
                specific_mask[specific_ids, :] = 1.0
            elif 'llama' in self.model_type:
                specific_mask[:, specific_ids] = 1.0   # transpose
            else:
                raise NotImplementedError
            self.specific_b_matrices.append(specific_b)
            self.specific_a_matrices.append(specific_a)
        else:
            specific_b = None
            specific_a = None
            specific_mask = None

        if layer_idx in self.independent_neurons:
            has_independent_neurons = True
            independent_ids = self.independent_neurons[layer_idx]
            r2 = len(independent_ids)  # rank
            independent_b = torch.zeros(
                (w.shape[1], self.config.n_rank * r2),  requires_grad=True, device=w.device,
            )
            independent_a = torch.randn(
                (self.config.n_rank * r2, w.shape[0]),  requires_grad=True, device=w.device,
            )
            independent_mask = torch.zeros_like(w)
            if 'mgpt' in self.model_type:
                independent_mask[independent_ids, :] = 1.0
            elif 'llama' in self.model_type:
                independent_mask[:, independent_ids] = 1.0   # transpose

            self.independent_b_matrices.append(independent_b)
            self.independent_a_matrices.append(independent_a)
        else:
            independent_b = None
            independent_a = None
            independent_mask = None

        params_this_layer = {
            "lang_specific_b": specific_b,
            "lang_specific_a": specific_a,
            "specific_mask": specific_mask,
            "lang_independent_b": independent_b,
            "lang_independent_a": independent_a,
            "independent_mask": independent_mask,
            "has_specific_neurons": has_specific_neurons,
            "has_independent_neurons": has_independent_neurons,
        }
        if has_specific_neurons or has_independent_neurons:
            # Register hook for this layer
            hook = CustomizedHooker(params_this_layer, self.model_type)
            module_name = self.mlp_c_proj_module.format(layer_idx)
            target_module = utf.get_attributes(model, module_name)
            handle = target_module.register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)
            self.edit_layers.append(layer_idx)

    def fit_on(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        no_print: bool = False,
    ):
        # if len(requests) > 1:
        #     raise ValueError("We only support single edit yet.")
        new_facts = []
        old_facts = []
        target_news = []
        target_trues = []
        for req in requests:
            target_new_str = " " + req["target_new"]
            target_true_str = " " + req["target_true"]
            prefix = req["prompt"].format(req["subject"])
            new_facts.append(prefix)
            old_facts.append(prefix)
            target_news.append(target_new_str)
            target_trues.append(target_true_str)

        if not no_print:
            print("---" * 10 + "Edit the following examples" + "---" * 10)
            for nf in new_facts:
                print(nf)

        # Optimization
        utf.freeze_model(model)
        print(f"lr = {self.config.editor.learning_rate}")
        # opt = torch.optim.Adam(self.parameters, lr=self.config.editor.learning_rate)
        opt = torch.optim.Adam(self.parameters, lr=self.config.editor.learning_rate)

        for _ in range(self.config.editor.num_steps):
            opt.zero_grad()
            new_fact_result = utf.forward_for_one_fact(
                model, tok, new_facts[0], target_news[0]
            )
            origin_fact_result = utf.forward_for_one_fact(
                model, tok, old_facts[0], target_trues[0]
            )
            old_probability = origin_fact_result["target_prob"]
            nll = new_fact_result["nll"]
            new_probability = new_fact_result["target_prob"]
            total_loss = (
                2.5 * nll
                + 2 * old_probability
            )
            if not no_print:
                print(f"Total loss = {total_loss.item():.6f}\tTarget Prob = {new_probability.item():.4f}")

            if total_loss.item() < 1e-2 or new_probability.item() > 0.96:   # 0.995
                break
            total_loss.backward()
            opt.step()
        
        # Update
        deltas = self._compute_unclamped_update()
        weights_copy = {}
        with torch.no_grad():
            for name, upd in deltas.items():
                w = utf.get_attributes(model, name)
                upd = upd.to(w.device)
                unclamped_norm = torch.norm(upd)
                # Clamp
                w_norm = torch.norm(w)
                scale = unclamped_norm / w_norm
                # scale = self._scale_with_sigmoid(scale)
                scale = 0.5   # == 0.4 0.3
                upd = scale * upd

                weights_copy[name] = w.detach().clone()
                w[...] += upd

        for h in self.handles:
            h.remove()
        return model, weights_copy

    def _compute_unclamped_update(self):
        delta = {}
        for ly_idx, h in zip(self.edit_layers, self.hooks):
            upd = h.get_delta().T
            w_name = self.mlp_c_proj_module.format(ly_idx) + ".weight"
            delta[w_name] = upd
        return delta

    def _compute_constraint_loss(self):
        losses = [hook.constraint_loss for hook in self.hooks]
        return sum(losses) / len(losses)

    def _scale_with_sigmoid(self, x):
        return 0.65 * torch.exp(-0.5 * torch.pow(x, 2)) + 0.4 * torch.sigmoid(2 * x)    # 0.65 0.4
