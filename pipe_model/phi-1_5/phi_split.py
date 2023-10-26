
import os 
import torch
from configuration_mixformer_sequential import MixFormerSequentialConfig
from pipe_mixformer_sequential import ClientSideMixFormerSequentialForCausalLM, ServerSideMixFormerSequentialForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import default_data_collator
from data_utils import transform_data_to_fedml_format, group_texts, tokenize_function
from functools import partial
from torch.utils.data import DataLoader

import math
import time
import numpy as np
import wandb
            
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For synchronous execution of CPU and GPU


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True
        
def print_trainable_params(model):
    for name, param in model.named_parameters():
        if(param.requires_grad):
            print(name, param.requires_grad)
        
def print_grad(model):
    for name, param in model.named_parameters():
        if(param.requires_grad):
            print(name, param.grad)
                
def main():
    LR = 5e-5
    MODEL_NAME = "microsoft/phi-1_5"
    DATASET_NAME = "wikitext"
    DATASET_NAME_CONFIG = "wikitext-2-raw-v1"
    LORA_RANK = 16
    EPOCH = 3
    lora_modules=["Wqkv"] 
    wandb.init(
        # set the wandb project where this run will be logged
        project="split-learning-measure-overhead",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": MODEL_NAME,
            "dataset": DATASET_NAME_CONFIG,
            "epochs": EPOCH,
            "LoRA rank": LORA_RANK,
            "LoRA module": lora_modules
        }
    )

    client_device = torch.device("cuda:2")
    server_device = torch.device("cuda:1")
    model_device = torch.device("cuda:0")

    config = MixFormerSequentialConfig()
    client_model = ClientSideMixFormerSequentialForCausalLM(config)
    server_model = ServerSideMixFormerSequentialForCausalLM(config)


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        cache_dir="/app/.huggingface_cache/model/"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir="/app/.huggingface_cache/model/",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print(model)
    
    lora_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=lora_modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    client_model = get_peft_model(client_model, lora_config)
    server_model = get_peft_model(server_model, lora_config)


    # resize embeddings
    embedding_size = client_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        client_model.resize_token_embeddings(len(tokenizer))

    # load raw data
    block_size = tokenizer.model_max_length
    raw_datasets = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir="/app/.huggingface_cache/dataset/",
        streaming=False
    )
    column_names = list(raw_datasets["train"].features)

    # data preprocessing 
    __tokenize_function = partial(tokenize_function, text_column_name="text", tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(
                        __tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                        desc="Running tokenizer on dataset",
                    )

    __group_texts = partial(group_texts, block_size=block_size)
    lm_datasets = tokenized_datasets.map(
                    __group_texts,
                    batched=True,
                    # num_proc=1,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    # desc=f"Grouping texts in chunks of {block_size}",
                )
    lm_datasets['train'].set_format("torch", device=client_device)
    train_dataloader = DataLoader(lm_datasets['train'], shuffle=True, collate_fn=default_data_collator, batch_size=1)
    print(lm_datasets['train'])
    print(train_dataloader)
    no_decay = ["bias", "layer_norm.weight"]
    client_grouped_parameters = [
        {
            "params": [p for n, p in client_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in client_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    server_grouped_parameters = [
        {
            "params": [p for n, p in server_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in server_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    model_grouped_paramters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    server_optimizer = torch.optim.AdamW(server_grouped_parameters, lr=0.001)
    client_optimizer = torch.optim.AdamW(client_grouped_parameters, lr=0.001)
    model_optimizer = torch.optim.AdamW(model_grouped_paramters, lr=0.001)

    # print("\n\n---- trainable params of server_model ----")
    # print_trainable_params(server_model)
    # print("\n\n---- trainable params of client_model ----")
    # print_trainable_params(client_model)
    # print(f"\n {len(train_dataloader)}")

    model.train()
    client_model.train()
    server_model.train()
    print(len(train_dataloader))

    latency = {
        "client": [],
        "server": [],
        "end-to-end": [],
        "model": []
    }

    for step, batch in enumerate(train_dataloader):
        print(f"step = {step} ")

        for key in batch.keys():
            batch[key] = batch[key].to(client_device)
        # batch = batch.to(device)
        # print(f"batch: {batch}")
        client_model, server_model, model = client_model.to(client_device), server_model.to(server_device), model.to(model_device)
        
        # Split training
        client_start_time = time.perf_counter()
        input_ids, past_key_values, attention_mask, labels, acts = client_model(**batch)
        client_end_time = time.perf_counter()
        acts.retain_grad()
        input_ids, attention_mask, labels, acts = input_ids.to(server_device), attention_mask.to(server_device), labels.to(server_device), acts.to(server_device)
        
        server_start_time = time.perf_counter()
        split_outputs = server_model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, labels=labels, hidden_layer_input=acts)
        server_end_time = time.perf_counter()
        
        split_loss = split_outputs.loss
        split_perplexity = math.exp(split_loss)
        split_loss.backward()
        
        server_optimizer.step()
        client_optimizer.step()
        
        # print("\n\n---- grad on server_model ----")
        # print_grad(server_model)    
        # print("\n\n---- grad on client_model ----")
        # print_grad(client_model)
        latency["client"].append(client_end_time-client_start_time)
        latency["server"].append(server_end_time-server_start_time)
        latency["end-to-end"].append(server_end_time-client_start_time)
        
        print(f"  - split_loss = {split_loss}")
        print(f"  - split_perplexity = {split_perplexity}")
        print(f"  - Latency (sec): Client = {np.average(latency['client'])}  ||  Server = {np.average(latency['server'])}  || End-to-end = {np.average(latency['end-to-end'])}")
        
        server_optimizer.zero_grad()
        client_optimizer.zero_grad()
        
        # Non-split training
        for key in batch.keys():
            batch[key] = batch[key].to(model_device)
        model_start_time = time.perf_counter()
        model_output = model(**batch)
        model_end_time = time.perf_counter()
        model_loss = model_output.loss
        model_perplexity = math.exp(model_loss)
        model_loss.backward()
        model_optimizer.step()    
        model_optimizer.zero_grad()
        latency['model'].append(model_end_time-model_start_time)
        print(f"  - model_loss = {model_loss}")
        print(f"  - model_perplexity = {model_perplexity}")
        print(f"  - Latency (sec): model = {np.average(latency['model'])}\n")

if __name__ == "__main__":
    main()