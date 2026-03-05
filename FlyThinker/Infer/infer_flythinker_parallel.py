import os
import json
import re
import copy
import random
import datetime as dt
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from utils.modeling_qwen2_flythinker import Qwen2ForCausalLM
from utils.flythinker import FlyThinker

# -----------------------------
# Prompt / args / utils
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Inference with Pipeline Parallel Setup")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--lambda_diff", type=float, required=True)
    parser.add_argument("--output_name", type=str, default="./outputs")
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_distributed():
    dist.init_process_group(backend="nccl", timeout=dt.timedelta(minutes=45))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


class SFTDataset(Dataset):
    def __init__(self, data_list):
        self.data_ = data_list

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):
        return self.data_[idx]
    

def prepare_data(test_dataset, tokenizer, max_len=5120):
    prepared_data = []
    for data in test_dataset:
        inp = data["inp"]
        profile = data["profile"]
        gt = data["target"]

        # truncate profile
        profile_split = profile.split(".")
        for ii in range(len(profile_split)):
            profile_split_ = profile_split[: -(ii + 1)]
            profile_ = ".".join(profile_split_)
            profile_split_id_len = len(tokenizer.encode(profile_))
            if profile_split_id_len < 1250:
                profile = profile_ + "."
                break

        # truncate inp
        inp_split = inp.split(".")
        for ii in range(len(inp_split)):
            inp_split_ = inp_split[: -(ii + 1)]
            inp_ = ".".join(inp_split_)
            inp_split_id_len = len(tokenizer.encode(inp_))
            if inp_split_id_len < 300:
                inp = inp_ + "."
                break

        input_text = profile + inp
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": input_text}], 
            add_generation_prompt=True, 
            tokenize=False
        )
        prepared_data.append({"prompt": prompt, "input": input_text, "gt": gt, "original_data": data})

    return prepared_data


def collate_fn(batch):
    """自定义 collate 函数"""
    return batch

import torch
import torch.distributed as dist


def parallel_generate_one_sample_new(
    rank: int,
    Reasoner,         # rank0: model; rank1: None
    Generator,             # rank1: model; rank0: None
    base_embed_table,     # rank1: embedding layer; rank0: None
    prompt_input_ids: torch.Tensor,      # [B,T] on *each rank's device*
    prompt_attention_mask: torch.Tensor, # [B,T] on *each rank's device*
    prompt_embeds_base: torch.Tensor,    # [B,T,H2] on rank1 device; rank0 can pass None
    prompt_embeds_reasoner: torch.Tensor,# [B,T,H1] on rank0 device; rank1 can pass None
    lambda_diff: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    pad_token_id: int,
):
    """
    Semantics remain: strict per-step ping-pong
      rank0: reasoner -> send hidden -> recv next token -> append
      rank1: recv hidden -> base -> sample -> send next token -> append
    """
    device = prompt_input_ids.device
    B, T0 = prompt_input_ids.shape
    input_ids = prompt_input_ids.clone()

    # KV caches
    past_r1 = None   # reasoner cache
    past_r2 = None   # base cache

    gen_id = None # rank1 to save the generated token

    # For norm scaling (rank1)
    if rank == 1:
        input_emb_norms = torch.norm(prompt_embeds_base, p=2, dim=-1, keepdim=False)  # [B,T]
    else:
        input_emb_norms = None

    # -------------------------------------------------
    # nucleus sampling (top-p)
    # -------------------------------------------------
    def sample_top_p(logits, top_p=0.9, temperature=1.0):
        if temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)  # [B,V]

        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        mask[..., 0] = False  # keep at least 1
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

        next_in_sorted = torch.multinomial(sorted_probs, 1)  # [B,1]
        next_token = sorted_idx.gather(-1, next_in_sorted).squeeze(1)  # [B]
        return next_token

    # ----------------------------
    # Precompute shapes once
    # ----------------------------
    if rank == 0:
        H1 = int(prompt_embeds_reasoner.shape[-1])
        hdim_msg = torch.tensor([H1, T0, B], device=device, dtype=torch.int64)
        dist.send(hdim_msg, dst=1)
    else:
        hdim_msg = torch.empty((3,), device=device, dtype=torch.int64)
        dist.recv(hdim_msg, src=0)
        H1, T0_recv, B_recv = map(int, hdim_msg.tolist())
        if T0_recv != T0 or B_recv != B:
            raise RuntimeError(
                f"Shape mismatch: rank{rank} local (B={B},T={T0}) "
                f"but got (B={B_recv},T={T0_recv}) from rank0."
            )

    # ----------------------------
    # Preallocate and reuse buffers
    # ----------------------------
    if rank == 1:
        # step0 to recv prompt hidden
        h_seq_buf = torch.empty((B, T0, H1), device=device, dtype=torch.bfloat16)
        # step>0 to recv last hidden
        h_last_buf = torch.empty((B, 1, H1), device=device, dtype=torch.bfloat16)
    else:
        h_seq_buf = None
        h_last_buf = None

    if rank == 0:
        # rank0 to recv token
        next_tok_buf = torch.empty((B,), device=device, dtype=torch.long)
    else:
        next_tok_buf = None

    # ----------------------------
    # EOS early-stop state (both ranks)
    # ----------------------------
    finished = torch.zeros((B,), device=device, dtype=torch.bool)
    # rank1 uses this to ensure EOS is sent at most once per sample (so rank0 can detect)
    sent_eos_once = torch.zeros((B,), device=device, dtype=torch.bool) if rank == 1 else None
    pad_const = torch.tensor(pad_token_id, device=device, dtype=torch.long)

    for step in range(max_new_tokens):
        # ---------------------------------------
        # step 0: rank0 sends full-seq hidden, rank1 injects on full prompt embeds
        # ---------------------------------------
        if step == 0:
            if rank == 0:
                out_r1 = Reasoner(
                    inputs_embeds=prompt_embeds_reasoner,
                    attention_mask=prompt_attention_mask,
                    use_cache=True,
                    past_key_values=past_r1,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_r1 = out_r1.past_key_values
                h_seq = out_r1.hidden_states[-1].contiguous()  # [B,T0,H1]

                if h_seq.dtype != torch.bfloat16:
                    h_seq = h_seq.to(torch.bfloat16)

                dist.send(h_seq, dst=1)

                dist.recv(next_tok_buf, src=1)

                # EOS bookkeeping on rank0
                finished |= (next_tok_buf == eos_token_id)

                input_ids = torch.cat([input_ids, next_tok_buf[:, None]], dim=-1)

                # batch-level early stop
                if bool(torch.all(finished).item()):
                    break
                continue

            else:
                # rank1
                dist.recv(h_seq_buf, src=0)

                hidden_norms = torch.norm(h_seq_buf, p=2, dim=-1)  # [B,T0]
                h_seq_scaled = h_seq_buf * (input_emb_norms / (hidden_norms + 1e-12))[:, :, None]

                r2_inp_emb = prompt_embeds_base.clone()  # [B,T0,H2]
                if r2_inp_emb.shape[-1] == h_seq_scaled.shape[-1]:
                    r2_inp_emb[:, 1:, :] = r2_inp_emb[:, 1:, :] + lambda_diff * h_seq_scaled[:, :-1, :]
                else:
                    r2_inp_emb[:, 1:, :h_seq_scaled.shape[-1]] = (
                        r2_inp_emb[:, 1:, :h_seq_scaled.shape[-1]] + lambda_diff * h_seq_scaled[:, :-1, :]
                    )

                out_r2 = Generator(
                    inputs_embeds=r2_inp_emb,
                    attention_mask=prompt_attention_mask,
                    use_cache=True,
                    past_key_values=past_r2,
                    return_dict=True,
                )
                past_r2 = out_r2.past_key_values

                logits = out_r2.logits[:, -1, :]  # [B,V]
                next_tok = sample_top_p(logits, top_p=top_p, temperature=temperature).to(torch.long)  # [B]

                # EOS early-stop on rank1:
                # - if a sample first hits eos: send eos this step (so rank0 can observe)
                # - if sample already finished and has sent eos before: send pad
                just_eos = (next_tok == eos_token_id)
                new_finished = finished | just_eos

                to_pad = new_finished & sent_eos_once
                next_tok = torch.where(to_pad, pad_const, next_tok)

                sent_eos_once |= just_eos
                finished = new_finished

                dist.send(next_tok, dst=0)

                input_ids = torch.cat([input_ids, next_tok[:, None]], dim=-1)
                gen_id = next_tok[:, None]  # [B,1]

                if bool(torch.all(finished).item()):
                    break
                continue

        # ---------------------------------------
        # step > 0: rank0 sends last hidden; rank1 injects into last token embed
        # ---------------------------------------
        if rank == 0:
            last_ids = input_ids[:, -1:]  # [B,1]
            out_r1 = Reasoner(
                input_ids=last_ids,
                use_cache=True,
                past_key_values=past_r1,
                output_hidden_states=True,
                return_dict=True,
            )
            past_r1 = out_r1.past_key_values
            h_last = out_r1.hidden_states[-1][:, -1:, :].contiguous()  # [B,1,H1]

            if h_last.dtype != torch.bfloat16:
                h_last = h_last.to(torch.bfloat16)

            dist.send(h_last, dst=1)

            dist.recv(next_tok_buf, src=1)

            # EOS bookkeeping on rank0
            finished |= (next_tok_buf == eos_token_id)

            input_ids = torch.cat([input_ids, next_tok_buf[:, None]], dim=-1)

            if bool(torch.all(finished).item()):
                break
            continue

        else:
            
            if step == 1:
                h_last_buf = h_seq_buf[:, -1:, :].contiguous()  # use last token hidden from step0 as "last hidden" for step1

            ref = input_emb_norms[:, -1:].contiguous()  # [B,1]
            hidden_norms = torch.norm(h_last_buf, p=2, dim=-1)  # [B,1]
            h_last_scaled = h_last_buf * (ref / (hidden_norms + 1e-12))[:, :, None]

            last_ids = input_ids[:, -1:]  # [B,1]
            r2_inp_emb = base_embed_table(last_ids)  # [B,1,H2]

            if r2_inp_emb.shape[-1] == h_last_scaled.shape[-1]:
                r2_inp_emb = r2_inp_emb + lambda_diff * h_last_scaled
            else:
                r2_inp_emb[:, :, :h_last_scaled.shape[-1]] = (
                    r2_inp_emb[:, :, :h_last_scaled.shape[-1]] + lambda_diff * h_last_scaled
                )

            out_r2 = Generator(
                inputs_embeds=r2_inp_emb,
                use_cache=True,
                past_key_values=past_r2,
                return_dict=True,
            )
            past_r2 = out_r2.past_key_values

            logits = out_r2.logits[:, -1, :]
            next_tok = sample_top_p(logits, top_p=top_p, temperature=temperature).to(torch.long)  # [B]

            # EOS early-stop on rank1
            just_eos = (next_tok == eos_token_id)
            new_finished = finished | just_eos

            to_pad = new_finished & sent_eos_once
            next_tok = torch.where(to_pad, pad_const, next_tok)

            sent_eos_once |= just_eos
            finished = new_finished

            dist.send(next_tok, dst=0)
            # rank1
            dist.recv(h_last_buf, src=0)
            
            input_ids = torch.cat([input_ids, next_tok[:, None]], dim=-1)
            gen_id = torch.cat([gen_id, next_tok[:, None]], dim=1) if gen_id is not None else next_tok[:, None]

            if bool(torch.all(finished).item()):
                break
            continue

    if rank == 1 and gen_id is not None:
        print(f'gen_id:{gen_id.shape}')

    # Keep original behavior: return generated tokens on each rank consistently
    return gen_id.squeeze() if gen_id is not None else gen_id


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    assert world_size == 2, "nproc_per_node=2"

    device = torch.device("cuda", local_rank)
    setup_seed(1)
    
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    Generator_path = './Qwen2.5-3B-Instruct' 
    Reasoner_path = './Qwen2.5-3B-Instruct'
    
    Generator = Qwen2ForCausalLM.from_pretrained(
        Generator_path,
        dtype=torch.bfloat16,#torch.float32,
        use_cache=True,
        trust_remote_code=True
    )
    flythinker = FlyThinker(Generator, reasoner_path = Reasoner_path)
    
    try:
        from safetensors.torch import load_file as safe_load
        sd = safe_load(f"{args.ckpt_path}/model.safetensors", device="cpu")
    except Exception:
        sd = torch.load(f"{args.ckpt_path}/pytorch_model.bin", map_location="cpu")

    missing, unexpected = flythinker.load_state_dict(sd, strict=False)
    
    if rank == 0:
        print("missing:", missing)
        print("unexpected:", unexpected)
        
    if rank == 0:
        Reasoner = flythinker.Reasoner.to(device).eval()
        for p in Reasoner.parameters():
            p.requires_grad = False
        Generator = None
        base_embed_table = None
    else:
        Generator = flythinker.Generator.to(device).eval()
        Reasoner = None
        base_embed_table = Generator.get_input_embeddings()
        
        
    paths = ["./topic_writing_user.json"]
    path_names = ["topic"]
    
    for idx, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            test_dataset = json.load(f)[:100]

        prepared_data = prepare_data(test_dataset, tokenizer)
        val_dataset = SFTDataset(prepared_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs, inputs, gts = [], [], []

        progress_bar = tqdm(val_dataloader, desc=f"Processing {path_names[idx]}", disable=(rank != 0))
        
        for ff, batch in enumerate(progress_bar):
            sample = batch[0]  # batch_size=1
            prompt = sample["prompt"]
            input_text = sample["input"]
            gt = sample["gt"]

            batch_tokens = tokenizer(
                [prompt],
                return_tensors="pt",
                padding="longest",
                padding_side="left",
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            
            # for time experiment
            # input_ids = batch_tokens.input_ids[:,:2].to(device)
            # attention_mask = batch_tokens.attention_mask[:,:2].to(device)
            
            input_ids = batch_tokens.input_ids.to(device)
            attention_mask = batch_tokens.attention_mask.to(device)

            # === rank1: base embeds，rank0: reasoner embeds === 
            if rank == 1:
                prompt_embeds_base = Generator.get_input_embeddings()(input_ids)
                prompt_embeds_reasoner = None
            else:
                prompt_embeds_reasoner = Reasoner.get_input_embeddings()(input_ids)
                prompt_embeds_base = None
                
            # === pipeline-parallel generate ===
            final_ids = parallel_generate_one_sample_new(
                rank=rank,
                Reasoner=Reasoner,
                Generator=Generator,
                base_embed_table=base_embed_table,
                prompt_input_ids=input_ids,
                prompt_attention_mask=attention_mask,
                prompt_embeds_base=prompt_embeds_base,
                prompt_embeds_reasoner=prompt_embeds_reasoner,
                lambda_diff=args.lambda_diff,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            if rank == 1:
                out_text = tokenizer.batch_decode(final_ids[None,:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                generated = out_text.strip()
                print(f"generated: {generated}")
                outputs.append(generated)
                inputs.append(input_text)
                gts.append(gt)
            
        if rank == 1:
            finals = [{"inp": inp, "pd": out, "gt": gt} for inp, out, gt in zip(inputs, outputs, gts)]
            output_file = f"./{args.output_name}_{path_names[idx]}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(finals, f, indent=4, ensure_ascii=False)
            print(f"Done! Results saved to {output_file}")

        dist.barrier()
        
    cleanup_distributed()

if __name__ == "__main__":
    main()
