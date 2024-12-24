import os
import gc
import torch

from dataclasses import dataclass, field
from typing import Optional, Dict

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    IntervalStrategy,
    TrainingArguments,
)

from os.path import join
from peft import LoraConfig, PeftModel, get_peft_model

from trainer import SiglipTrainer

# import bitsandbytes as bnb
from datasets import Dataset

import warnings
import random
import sys

from tqdm import tqdm

from data import TrainDatasetForEmbedding, EmbedCollator
from arguments import DataArguments


# Define and parse arguments.
@dataclass
class ScriptArguments:
    experiment_name: str = field(
        metadata={"help": "experiment name"},
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "local rank of process"}
    )

class TrainingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)


parser = HfArgumentParser((DataArguments, ScriptArguments))
data_args, script_args = parser.parse_args_into_dataclasses()

local_rank = script_args.local_rank
experiment_name = script_args.experiment_name

experiment_config_path = f"/home/nikhil.kothari/all/sm/src/train/finetune/deepspeed/config/{experiment_name}.yml"
config = TrainingConfig.from_file(file_path=experiment_config_path)

RANDOM_SEED = config.random_seed
random.seed(RANDOM_SEED)

base_model_path = config.base_model_path
output_dir = join(
    config.output_dir,
    f"{config.base_model_path.rstrip('/').split('/')[-1]}-emb-experiment-{experiment_name}",
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

if tokenizer.pad_token_id is None:
    print("adding pad token..")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

tokenizer.padding_side = "right"

if config.emb_type == "eos_token" and tokenizer.eos_token is None:
    raise

eos_token = (
    tokenizer.eos_token
    if config.emb_type == "eos_token"
    and tokenizer("A")["input_ids"][-1] != tokenizer.eos_token_id
    else ""
)

# Data Collator
collator = EmbedCollator(
    tokenizer,
    query_max_len=config.query_max_len,
    passage_max_len=config.passage_max_len,
)

# Load Dataset
# Dataset Arguments
data_args: DataArguments
data_args.train_data = config.dataset_name_or_path
data_args.query_max_len = config.query_max_len
data_args.passage_max_len = config.passage_max_len
data_args.max_example_num_per_dataset = config.max_example_num_per_dataset
data_args.query_instruction_for_retrieval = config.query_instruction_for_retrieval
data_args.passage_instruction_for_retrieval = config.passage_instruction_for_retrieval
data_args.use_dataset_neg = config.use_dataset_neg
data_args.train_group_size = config.train_group_size

train_data = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
eval_data = list()

print(f"Training Data Size: {len(train_data)}")
print(f"Evaluation Data Size: {len(eval_data)}")
print(train_data[0])

# Training arguments
training_args = TrainingArguments(
    save_safetensors=False,
    remove_unused_columns=False,
    group_by_length=False,
    output_dir=output_dir,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=config.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # eval_strategy=IntervalStrategy.STEPS,
    eval_strategy="no",
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    logging_steps=config.logging_steps,
    deepspeed=config.deepspeed,
    max_steps=config.max_steps,
    load_best_model_at_end=False,
    lr_scheduler_type=getattr(config, "lr_scheduler", "cosine"),
    report_to="tensorboard",
    local_rank=local_rank,
    save_strategy="steps",
    save_total_limit=1,
    weight_decay=config.weight_decay,
    warmup_steps=getattr(config, "warmup_steps", -1),
    # warmup_ratio=config.warmup_ratio,
    # eval_steps=save_eval_steps,
    save_steps=500,
    # adam_epsilon=1e-6,
    # adam_beta2=0.98,
    logging_dir=f"/home/nikhil.kothari/.extension/tensorboard/{config.base_model_path.rstrip('/').split('/')[-1]}-emb-experiment-{experiment_name}-reentrant-false/",
)

# Model to fine-tune
"""
Load the model
"""
from transformers import AutoModel

model = AutoModel.from_pretrained(
    base_model_path  # , torch_dtype=torch.float16, trust_remote_code=True
)
# model.config.use_cache = False

# LoRA configuration
if config.use_peft:
    peft_config = LoraConfig(
        r=config.peft_lora_r,
        lora_alpha=config.peft_lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=[
            "k_proj",
            "gate_proj",
            "v_proj",
            "up_proj",
            "q_proj",
            "o_proj",
            "down_proj",
        ],
    )

else:
    peft_config = None


# Create trainer
trainer = SiglipTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collator,
    peft_config=peft_config,
    emb_type=config.emb_type,
)

# trainer.model.base_model.model.model.resize_token_embeddings(len(tokenizer))
# trainer.model.resize_token_embeddings(len(tokenizer))
trainer.train()

if config.use_peft:
    trainer.model = trainer.model.merge_and_unload()

trainer.save_model(output_dir)
