import os
import sys

import pandas as pd

from transformers import AutoTokenizer, set_seed
from FlyThinker.Train.dataset import PersonalDataset, convert_to_dataset
import json

os.environ["HF_DATASETS_CACHE"] = ''
os.environ['HF_HOME'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
set_seed(42)

llm_model_name = "your LLM"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"

with open('./train_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
personal_dataset = PersonalDataset(data,
                        llm_tokenizer=llm_tokenizer,
                        training = True
                        )

print(personal_dataset.get_avg_profile_len())
hf_dataset = convert_to_dataset(personal_dataset)
hf_dataset.save_to_disk("./data/dataset_train")


with open('./val_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
personal_dataset = PersonalDataset(data,
                        llm_tokenizer=llm_tokenizer,
                        training = True
                        )

print(personal_dataset.get_avg_profile_len())
hf_dataset = convert_to_dataset(personal_dataset)
hf_dataset.save_to_disk("./data/dataset_val")


