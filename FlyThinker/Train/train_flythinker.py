import os
import sys
import torch
import warnings
import torch.distributed as dist
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import Trainer,TrainingArguments
from transformers import set_seed
from FlyThinker.Train.flythinker import FlyThinker
from FlyThinker.Train.utils.modeling_qwen2 import Qwen2ForCausalLM
from transformers import TrainerCallback
import argparse

os.environ["HF_DATASETS_CACHE"] = ''
os.environ['HF_HOME'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Config")

    parser.add_argument(
        "--diff_lambda",
        type=float,
        required=True,
    )
    
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
    )

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)

    args, _ = parser.parse_known_args()
    return args

class EarlyStopAtEpoch(TrainerCallback):
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.stop_epoch:
            control.should_training_stop = True
        return control

set_seed(42)
world_size = int(os.environ.get("WORLD_SIZE", 1))
device_map = "auto"
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

class CustomTrainer(Trainer):
    def __init__(self, perslt_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.have_print = True
        self.perslt_args = perslt_args
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        position_ids = torch.clip(torch.cumsum(inputs['attention_mask'], dim=-1) - 1, min=0, max=None)
        
        inputs_plus = { "input_ids": inputs['input_ids'], 
                        "attention_mask":inputs["attention_mask"],
                        "labels": inputs['labels'],
                        "position_ids": position_ids,
                        "diff_lambda": self.perslt_args.diff_lambda,
                        "num_items_in_batch": num_items_in_batch
                   }
        
        loss = model(**inputs_plus)
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
    
        return loss

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
llm_model_name = "./Qwen2.5-7B-Instruct"
reasoner_path = "./Qwen2.5-1.5B-Instruct"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"

personal_model = Qwen2ForCausalLM.from_pretrained(
    llm_model_name,
    device_map=device_map,
    dtype=torch.bfloat16,
    trust_remote_code=True
)
personal_model = FlyThinker(personal_model, reasoner_path)

device_num = torch.cuda.device_count()

print_trainable_parameters(personal_model)
if not ddp and torch.cuda.device_count() > 1:
        personal_model.is_parallelizable = True
        personal_model.model_parallel = True 

perslt_args = parse_args()

training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir=f"{perslt_args.save_name}",
    logging_steps=10,
    save_strategy="epoch",
    # save_strategy="steps",          
    # save_steps=5,              
    # eval_strategy='epoch',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    learning_rate=5e-6,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    # deepspeed="deepspeed/ds_z1_config.json",
    deepspeed="deepspeed/ds_z3_config.json",
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False if ddp else None,
    run_name="demo-rag",
    save_safetensors=False, 
    # load_best_model_at_end=True,
)

dataset_train = load_from_disk(f"./data/dataset_train").remove_columns(["inp_str", "out_str"])
dataset_val = load_from_disk(f"./data/dataset_val").remove_columns(["inp_str", "out_str"])

for ii, data in enumerate(dataset_train):
    if ii < 1:
        print(data['input_ids'])
        print(llm_tokenizer.decode(data['input_ids']))
    
trainer = CustomTrainer(
    model=personal_model,
    args=training_args,
    perslt_args = perslt_args,
    train_dataset=dataset_train,
    eval_dataset = dataset_val,
    tokenizer=llm_tokenizer,
    callbacks=[EarlyStopAtEpoch(stop_epoch=5)]
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
)

print("train start")
trainer.train()
print("train done")
if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)