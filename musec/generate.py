"""Contains autoregressive generation code."""

import json
import argparse
import torch
import pyroll
from dataclasses import dataclass

from model import TransformerLM, ModelConfig
from utils import Tokenizer
from train import PretrainLM


def lazy_sample(
    prompt: list,
    model: TransformerLM,
    tokenizer: Tokenizer,
    temp: float = 0.9,
    piano_roll: bool = True,
):
    prompt_len = len(prompt)
    max_seq_len = tokenizer.model_config.max_seq_len
    res = tokenizer.encode(prompt).cuda()

    for _ in range(max_seq_len - prompt_len):
        logits = model(res.reshape(1, -1))[0, -1, :] / temp
        probs = torch.nn.functional.softmax(logits, dim=0)
        next = torch.multinomial(probs, num_samples=1)
        res = torch.cat((res, next), dim=-1)
        print(tokenizer.id_to_tok[next.item()])

    res = tokenizer.decode(res)
    if piano_roll is True:
        return pyroll.PianoRoll.from_seq(res)
    else:
        return res


def sampale_causal(model_path: str, prompt_path: str):
    # Load model
    model_config = ModelConfig()
    tokenizer = Tokenizer(model_config)
    model = get_torch_module(model_path).cuda()
    model.eval()

    # Load prompts if provided
    if prompt_path is None:
        prompts = [["<S>"]] * 10
    else:
        with open(prompt_path) as f:
            prompts = json.load(f)

    prompt_len = 100
    for i, prompt in enumerate(prompts):
        assert tokenizer.unk_tok not in tokenizer.decode(
            tokenizer.encode(prompt)
        ), "unk_tok present in prompt"

        res = lazy_sample(
            prompt=prompt[:prompt_len],
            model=model,
            tokenizer=tokenizer,
            piano_roll=False,
        )

        print(f"Done {i+1}/{len(prompts)}")
        res_p_roll = pyroll.PianoRoll.from_seq(res)
        prompt_p_roll = pyroll.PianoRoll.from_seq(prompt)
        res_mid = res_p_roll.to_midi()
        prompt_mid = prompt_p_roll.to_midi()
        res_mid.save(f"samples/res{i+1}.mid")
        prompt_mid.save(f"samples/prompt{i+1}.mid")


def get_torch_module(load_path: str):
    """Returns torch nn.Module from lightning module."""
    return PretrainLM.load_from_checkpoint(load_path).model


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_path", type=str, default="models/params.ckpt")
    argp.add_argument("--prompt_path")
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()
    sampale_causal(**kwargs)
