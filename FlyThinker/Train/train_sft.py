import os
import sys
import torch
import warnings
import torch.distributed as dist

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer,TrainingArguments
from transformers import set_seed
from transformers import TrainerCallback
os.environ["HF_DATASETS_CACHE"] = ''
os.environ['HF_HOME'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
warnings.filterwarnings("ignore")



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
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"
personal_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map=device_map,
    dtype=torch.bfloat16,
    trust_remote_code=True
)

device_num = torch.cuda.device_count()

print_trainable_parameters(personal_model)
if not ddp and torch.cuda.device_count() > 1:
        personal_model.is_parallelizable = True
        personal_model.model_parallel = True 

training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir=f"output_SFT",
    logging_steps=10,
    save_strategy="epoch",
    # eval_strategy='epoch',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    learning_rate=5e-6,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    deepspeed="deepspeed/ds_z1_config.json",
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False if ddp else None,
    run_name="demo-rag",
    # load_best_model_at_end=True,
)

dataset_train = load_from_disk(f"./data/dataset_train").remove_columns(["inp_str", "out_str"])
dataset_val = load_from_disk(f"./data/dataset_val").remove_columns(["inp_str", "out_str"])

for ii, data in enumerate(dataset_train):
    if ii < 1:
        print(data['input_ids'])
        print(llm_tokenizer.decode(data['input_ids']))

trainer = Trainer(
    model=personal_model,
    args=training_args,
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
