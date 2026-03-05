import torch
import datasets
import numpy as np

from tqdm import tqdm
from utils.templates import Qwen2PromptTemplate


class PersonalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                llm_tokenizer,
                max_length=2048,
                max_his_len=8,
                training=True
    ):
        self.main_dataset = main_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.cnt = 0
        
        self.pt = Qwen2PromptTemplate()

        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            profile = self.main_dataset[idx]["profile"]
            profile_split = profile.split('.')
            for ii in range(len(profile_split)):
                profile_split_ = profile_split[:-(ii+1)]
                profile_ = '.'.join(profile_split_)
                profile_split_id = llm_tokenizer.encode(profile_)
                profile_split_id_len = len(profile_split_id)
                if profile_split_id_len < 1250:
                    # print(f'profile_split_id_len:{profile_split_id_len}')
                    profile = profile_ + '.'
                    break   
            inp = self.main_dataset[idx]["inp"]   
            inp_split = inp.split('.')
            for ii in range(len(inp_split)):
                inp_split_ = inp_split[:-(ii+1)]
                inp_ = '.'.join(inp_split_)
                inp_split_id = llm_tokenizer.encode(inp_)
                inp_split_id_len = len(inp_split_id)
                if inp_split_id_len < 300:
                    # print(f'inp_split_id_len:{inp_split_id_len}')
                    inp = inp_ + '.'
                    break   
            response = self.main_dataset[idx]["target"]
            
            response_split = response.split('.')
            response_split = [item for item in response_split if item]
            for ii in range(len(response_split)):
                response_split_ = response_split[:-(ii+1)]
                response_split_orig = response_split[:1].copy()
                response_ = '.'.join(response_split_)
                response_split_id = llm_tokenizer.encode(response_)
                response_split_id_len = len(response_split_id)
                if response_split_id_len == 1:
                    # print(f'response_split_orig:{response_split_orig}')
                    response = '.'.join(response_split_orig) + '.'
                    break
                if response_split_id_len < 1000:            
                    response = response_ + '.'
                    break
                
            inp_str = profile + inp
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = 1250 + 300 + 1000
            out_str = response
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=1250 + 300,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=1000, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
            }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)
            

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_avg_profile_len(self):
        return self.cnt / self.total_len
    

    
def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data
    return datasets.Dataset.from_generator(gen)
