import sys
import os
import tqdm
import argparse
import json
from time import time
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
import yaml 
from safetensors import safe_open
import numpy as np
import random
from collections import defaultdict
from tiktoken.load import load_tiktoken_bpe
from model import FlashSTU, FlashSTUConfig
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.utils.random_utils import get_logger, save_yaml_config
import math
from typing import Union
import gc

logger = get_logger(__name__)
bpe_path = "./o200k_base.tiktoken"

def set_initial_random_seed(random_seed: int):
    if random_seed > 0:
        seed_offset = 0 #single gpu for now
        random_seed += seed_offset
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

def apply_compile(model: nn.Module) -> None:
    """
    Apply torch.compile to each layer. This makes compilation efficient
    due to repeated structure. Alternatively, one can just compile the whole model.
    """
    logger.info(f"Compiling each {model.__class__.__name__} layer with torch.compile...")
    start = time.perf_counter()
    for idx, layer in model.layers.named_children():
        compiled_layer = torch.compile(layer, mode="max-autotune", fullgraph=True)
        model.layers.register_module(idx, compiled_layer)
    end = time.perf_counter()
    logger.info(f"Finished compiling each {model.__class__.__name__} layer in {end - start:.4f} seconds.")


def load_stu_model(config_data, checkpoint_path: str, device: torch.device, futurefill_k: Union[None, int]):

    torch_dtype = getattr(torch, config_data["torch_dtype"])
    is_futurefill = futurefill_k is not None

    model_config = FlashSTUConfig(**config_data)
    model_config.torch_dtype = getattr(torch, config_data["torch_dtype"])
    is_futurefill = futurefill_k is not None

    spectral_filters = get_spectral_filters(model_config.seq_len, model_config.num_eigh, model_config.use_hankel_L, device, torch_dtype)
    model = FlashSTU(model_config, spectral_filters, future_fill = futurefill_k)
    model = model.to(device=device, dtype=torch_dtype)

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = {}
        start_time = time()

        if checkpoint_path.endswith(".safetensors"):
            with safe_open(checkpoint_path, framework="pt", device=device.type) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        elif checkpoint_path.endswith(".pt"):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        logger.info(f"Checkpoint loaded in {time() - start_time:.2f} seconds.")

        model.load_state_dict(state_dict, strict=True)
        logger.info("Model weights loaded successfully!")

    if is_futurefill:
        for idx, layer in enumerate(model.layers):
            if hasattr(layer, "stu"):
                logger.warning("Right now, FutureFill is intialized with the model ... So can't run multiple sequence length on the same run")
                model.layers[idx].stu.setup_ff(futurefill_k)


    if config_data["torch_compile"]:
        model = apply_compile(model)
    model.eval()

    return model, config_data

def generate_text(
    model,
    tokenizer,
    prompt,
    num_return_sequences=1,
    max_length=512,
    device="cuda",
    temperature=1.0,
    top_k=50,
    cache = True, 
    futurefill_k = None
):
    """
    Generate text from the given prompt using top-k sampling.

    Args:
        model: The FlashSTU model instance.
        tokenizer: The tokenizer used for encoding/decoding.
        prompt (str | torch.tensor): Input prompt text.
        num_return_sequences (int): How many sequences to return.
        max_length (int): Maximum length of generated tokens.
        device: torch device.
        temperature (float): Sampling temperature. Higher = more random.
        top_k (int): Top-K sampling parameter.

    Returns:
        list[str]: A list of generated text sequences.
    """

    # Encode prompt tokens.
    if isinstance(prompt, torch.Tensor):
        if prompt.numel() == 0:
            if tokenizer.name.lower() == "o200k_base":
                tokens = torch.tensor([[tokenizer.eot_token]], device=device)
            elif "gpt" in tokenizer.name.lower():
                tokens = torch.tensor([[tokenizer.bos_token_id]], device=device)
        else:
            tokens = prompt
    else:
        tokens = torch.tensor(
            [tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})],
            device=device,
        )
    
    seq_len = tokens.shape[1]
    tokens = tokens.repeat(num_return_sequences, 1)
    
    input_pos = torch.arange(seq_len, device=device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1746)

    eos_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]

    cur_token = seq_len
    with torch.no_grad():
        for idx in tqdm.tqdm(range(max_length - tokens.size(1))):
            with torch.amp.autocast(device_type="cuda", dtype=model.config.torch_dtype):

                # Fwd pass. Inspect logits here.
                if not cache and futurefill_k is None:
                    logits = model(tokens, input_pos = input_pos)
                elif idx != 0:
                    logits = model(tokens[:, -1:], input_pos = input_pos)     # shape: [batch, 1, vocab]
                else:
                    logits = model(tokens, input_pos = input_pos)     # shape: [batch, seq, vocab]
                logits = logits[:, -1, :]  # last token logits

                # Apply temperature scaling.
                if temperature > 0:
                    logits = logits / temperature

            # # Compute probabilities -> no need for proba
            # probs = F.softmax(logits, dim=-1)

            # # Top-K sampling
            # top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            # ix = torch.multinomial(top_k_probs, 1, generator=sample_rng)
            # next_token = torch.gather(top_k_indices, -1, ix)
            next_token = torch.argmax(logits, dim = -1, keepdim=True)

            # Append next token.
            tokens = torch.cat((tokens, next_token), dim=1)
            input_pos = torch.tensor([cur_token]).to(device)
            cur_token +=1 

            # # Stop if EOS token is generated -> we don't want to stop
            # if (next_token == eos_token_id).any():
            #     break

    return tokens

def generate_and_time(model, tokenizer, eval_config, save_path_for_this_exp, device):
    total_tokens = 0
    start_time = time()
    cache = eval_config.get("cache", False)
    max_length = eval_config.get("max_length", [32])
    input_length = eval_config.get("input_length", [32])
    BASE_TEMPERATURE = eval_config.get("temperature", 0.7)
    BASE_TOP_K = eval_config.get("top_k", 50)
    num_repeat = eval_config.get("repeat", 2)
    debug = eval_config.get("debug", False)
    futurefill_k = eval_config.get("futurefill_k", None)

    for i, input_length in enumerate(input_length, 1):
        runtimes = {}
        for j, final_length in enumerate(max_length):
            running_runtime = []
            for repeat in range(num_repeat):
                logger.info(f"Generating text for prompt {i} of length {input_length}. Max generation is {final_length}")
                if cache:
                    if not isinstance(futurefill_k, list):
                        model.setup_caches(batch_size = 1)

                if not cache and model.caches_are_enabled():
                    model.reset_caches()

                _dummy_prompt_ids = torch.full((1, input_length), 1)
                
                start_time = time()
                tokens = generate_text(
                    model,
                    tokenizer,
                    _dummy_prompt_ids,
                    num_return_sequences=1,
                    max_length=final_length,
                    device=device,
                    temperature=BASE_TEMPERATURE,
                    top_k=BASE_TOP_K,
                    cache = cache, 
                    futurefill_k = futurefill_k
                )

                end_time = time()
                current_runtime = end_time - start_time
                running_runtime.append(current_runtime)

                if cache:
                    if futurefill_k is None:
                        model.reset_caches()

                    elif futurefill_k is not None:
                        for idx, layer in enumerate(model.layers):
                            if hasattr(layer, "stu"):
                                model.layers[idx].stu.eff_plus.reset_cache()
                                if not model.config.use_hankel_L:
                                    model.layers[idx].stu.eff_minus.reset_cache()

                logger.info(f"Current runtime: {current_runtime}")

            mean_running_time = np.mean(running_runtime)
            mean_running_time_wo_first = np.mean(running_runtime[1:])
            runtimes[final_length] = mean_running_time
            runtimes[f"{str(final_length)}-wo-first"] = mean_running_time_wo_first
            logger.info(f"final_length: {final_length}")
            logger.info(f"Mean runtime: {mean_running_time}")
            logger.info(f"Mean runtime wo first: {mean_running_time_wo_first}")
            logger.info(f"std runtimes: {np.std(running_runtime)}")
            logger.info(f"std runtimes wo first: {np.std(mean_running_time_wo_first)}")
            logger.info(f"min runtimes: {min(running_runtime)}")
            logger.info(f"max runtimes: {max(running_runtime)}")
            logger.info("-----\n")

            if debug:
                generated_text = tokenizer.decode(tokens[0].tolist())
                logger.info(f"\nPrompt: {_dummy_prompt_ids}")
                # logger.info(f"Generated Text: {generated_text}\n")
                logger.info(f"Generated Tokens: {tokens}\n")
                total_tokens += len(tokenizer.encode(generated_text, allowed_special={"<|endoftext|>"}))

        logger.info("Runtimes in seconds: \n")
        logger.info(runtimes)
        logger.info("-----\n")

        # Save using output dir and model name and date
        sub_exp_name = f"runname={eval_config.get('run_name')}-input_seqlen={input_length}-max_output_seqlen={max_length[0]}-numlayer={model.config.num_layers}-dim={model.config.dim}-attn={model.config.use_attn}.json"

        with open(os.path.join(save_path_for_this_exp, sub_exp_name), "w") as file:
            json.dump(runtimes, file)
        
        
        

def main(args):
    # Load eval config
    eval_config = yaml.load(open(args.eval_path, 'r'), Loader=yaml.FullLoader)
    with open(args.config_path, "r") as f:
        model_config = json.load(f)

    # Create output dir; save yaml configs in there
    run_id = random.randint(0, 10 ** 6)
    sub_run_id = 0

    # Set random seed for reproducibility
    set_initial_random_seed(eval_config.get('random_seed', -1))

    for NEW_NUM_LAYERS in [8, 12, 16]:
        for NEW_MAX_LENGTH in [4096, 8192, 16384, 32768, 65536, 131072]:

            # save yaml configs in there
            save_path_for_this_exp = os.path.join(eval_config.get('save_dir'), str(run_id), str(sub_run_id))
            os.makedirs(save_path_for_this_exp, exist_ok=True)
            logger.info(f"For this experiment, saving path is: {save_path_for_this_exp}")


            # Overwrite some existing config
            model_config["num_layers"] = NEW_NUM_LAYERS
            eval_config["max_length"] = [NEW_MAX_LENGTH]
            eval_config["futurefill_k"] = ["None"]

            logger.info(f"eval_config: {eval_config}")
            logger.info(f"model_config: {model_config}")


            save_yaml_config(eval_config, save_path_for_this_exp, "eval_config.yaml")
            save_yaml_config(model_config, save_path_for_this_exp, "model_config.yaml")

            # Need to caclulcate futurefill K here ... TO BE IMPROVED
            futurefill_k = eval_config.get("futurefill_k", None)
            if futurefill_k is not None:
                if isinstance(futurefill_k[-1], str) and futurefill_k[-1] == "None":
                    generation_L = eval_config.get("max_length")[-1]
                    futurefill_k = int(math.sqrt(generation_L * math.log2(generation_L)))
                elif isinstance(futurefill_k[-1], int):
                    futurefill_k = futurefill_k[-1]

            # Load model and config.
            device = torch.device("cuda")
            model, config_data = load_stu_model(model_config, args.checkpoint_path, device, futurefill_k = futurefill_k)
            
            # Create tokenizer (for della)
            bpe_dict = load_tiktoken_bpe(bpe_path)
            tokenizer = tiktoken.Encoding(
                name="o200k_base",  # Name of the encoding
                pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""",
                mergeable_ranks=bpe_dict,
                special_tokens={
                    "<|endoftext|>": 199999,  # Custom special token example (modify as needed)
                    "<|endofprompt|>": 200018,
                }
            )

            generate_and_time(model, tokenizer, eval_config, save_path_for_this_exp, device)

            sub_run_id += 1

            del model
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    CHECKPOINT_PATH = "" #"./model_step-114000.safetensors"
    CONFIG_PATH = "./configs/test/config.json"
    EVAL_PATH = "./configs/test/eval.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_PATH, type=str)
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str)
    parser.add_argument('--eval_path', default=EVAL_PATH, type=str)
    
    args = parser.parse_args()

    main(args)