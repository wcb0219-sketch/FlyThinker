from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import datetime as dt
import numpy as np
import random
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Config")

    # 基础参数
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="./outputs", 
    )
    
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_distributed():
    # dist.init_process_group(backend='nccl')
    dist.init_process_group(
        backend='nccl',
        timeout=dt.timedelta(minutes=45) 
    )
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

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
        
        profile_split = profile.split('.')
        for ii in range(len(profile_split)):
            profile_split_ = profile_split[:-(ii+1)]
            profile_ = '.'.join(profile_split_)
            profile_split_id = tokenizer.encode(profile_)
            profile_split_id_len = len(profile_split_id)
            if profile_split_id_len < 1250:
                profile = profile_ + '.'
                break   
        
        inp_split = inp.split('.')
        for ii in range(len(inp_split)):
            inp_split_ = inp_split[:-(ii+1)]
            inp_ = '.'.join(inp_split_)
            inp_split_id = tokenizer.encode(inp_)
            inp_split_id_len = len(inp_split_id)
            if inp_split_id_len < 300:
                inp = inp_ + '.'
                break   
        
        input_text = profile + inp
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": input_text}], 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        prepared_data.append({
            'prompt': prompt,
            'input': input_text,
            'gt': gt,
            'original_data': data
        })
    
    return prepared_data

def collate_fn(batch):
    return batch

def main():
    args = parse_args()
    local_rank, world_size = setup_distributed()
    setup_seed(1)


    model_path = args.ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()

    tokenizer.padding_side = "left"
    llm.eval()

    llm = DDP(llm, device_ids=[local_rank], output_device=local_rank)

    paths = ["./topic_writing_user.json"]
    path_names = ["topic"]
    
    for idx, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            test_dataset = json.load(f)
        print(f'path:{path}\n')
        if local_rank == 0:
            print(f'test_dataset:{len(test_dataset)}')
        
        has_print = False
        
        prepared_data = prepare_data(test_dataset, tokenizer)
        
        val_dataset = SFTDataset(prepared_data)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1, 
            sampler=val_sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
        
        outputs = []
        inputs = []
        gts = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc=f"Processing {path_names[idx]}", disable=(local_rank != 0))
            for ff, batch in enumerate(progress_bar):
                for sample in batch: 
                    prompt = sample['prompt']
                    input_text = sample['input']
                    gt = sample['gt']
                    
                    batch_tokens = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        padding="longest",
                        padding_side='left',
                        truncation=False,
                        add_special_tokens=False,
                        return_attention_mask=True
                    )
                    if not has_print:
                        print(batch_tokens.input_ids[0].tolist())
                        print(tokenizer.decode(batch_tokens.input_ids[0]))
                        has_print = True
                        
                    input_embeds = llm.module.get_input_embeddings()(batch_tokens.input_ids.cuda())
                    attention_mask = torch.tensor([[1] * input_embeds.shape[-2]]).to(input_embeds.device)
                    generated_output = llm.module.generate(
                        inputs_embeds = input_embeds,
                        attention_mask = attention_mask,
                        max_new_tokens=1536,
                        # min_new_tokens = 10,
                        do_sample=True,
                        temperature=0.8,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        top_p=0.95,
                        num_return_sequences=1,
                        return_dict_in_generate=True,
                    )

                    generate_ids = generated_output.sequences
                    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    generated = [text.strip() for text in output_text]
                    
                    print(f'generated:{local_rank},{generated}', flush=True)
                    generate_id_new = tokenizer.encode(generated[0])
                    print(f'generate_id_new:{local_rank},{len(generate_id_new)}')
                    
                    outputs.extend(generated)
                    inputs.append(input_text)
                    gts.append(gt)
                    
        dist.barrier()

        all_outputs = [None for _ in range(world_size)]
        all_inputs = [None for _ in range(world_size)]
        all_gts = [None for _ in range(world_size)]
        
        dist.all_gather_object(all_outputs, outputs)
        dist.all_gather_object(all_inputs, inputs)
        dist.all_gather_object(all_gts, gts)
        
        if local_rank == 0:
            final_outputs = []
            final_inputs = []
            final_gts = []
            
            for outputs_per_gpu, inputs_per_gpu, gts_per_gpu in zip(all_outputs, all_inputs, all_gts):
                final_outputs.extend(outputs_per_gpu)
                final_inputs.extend(inputs_per_gpu)
                final_gts.extend(gts_per_gpu)
            
            finals = [
                {
                    "inp": input_text,
                    "pd": output,
                    "gt": gt
                }
                for input_text, output, gt in zip(final_inputs, final_outputs, final_gts)
            ]
            
            output_file = f"./{args.output_name}_{path_names[idx]}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(finals, f, indent=4, ensure_ascii=False)
            
            print(f"Done! Results saved to {output_file}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()

