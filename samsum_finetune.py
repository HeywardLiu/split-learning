import os
os.environ['CUDA_VISIBLE_DEVICES']="2,3"

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
import evaluate
import numpy as np
import copy
import nltk
    
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{11264}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, 
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config



def find_all_linear_names(model):
    # cls = torch.nn.Linear
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def tokenize_batch(batch, tokenizer, max_length):  # Tokenizing a batch
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_function(sample, tokenizer, max_length=1024, padding="max_length", do_truncation=True, text_column='dialogue', summary_column='summary'):
    """
    Format various fields of the sample ('id', 'dialogue', 'summary')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionary
    """
    # clean data 
    inputs, targets = [], []
    for i in range(len(sample[text_column])):
        if sample[text_column][i] and sample[summary_column][i]:
            inputs.append(sample[text_column][i])
            targets.append(sample[summary_column][i])

    INTRO_BLURB = "Below is a short dialogue. Write a summary of the conversation.\n\n"
    INSTRUCTION_KEY = "### Dialogue:\n"
    RESPONSE_KEY = "\n\n### Summary:"

    # format input to alpaca style
    inputs = [INTRO_BLURB + INSTRUCTION_KEY + inp for inp in inputs]
    inputs = [inp + RESPONSE_KEY for inp in inputs]
    
    # tokenize to numerical type
    model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=do_truncation)
    labels = tokenizer(text_target=targets, max_length=max_length, padding=padding, truncation=do_truncation)
    print(len(inputs), len(targets))
    
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]\

    return model_inputs


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    # Add prompt to each sample
    print("Preprocessing dataset...")
    # _preprcessing_function = partial(create_prompt_formats, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "do_truncation": True},
        batched=True,
        remove_columns=["id", "dialogue", "summary"]
    )
        
    print(type(dataset[0]))
    for i in range(2):
        print(f"--- formatted data {i} ---")
        print(dataset[i].get('text'))        
        print("-----------\n\n\n")
        
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    # _preprocessing_function = partial(tokenize_batch, max_length=max_length, tokenizer=tokenizer)
    # tokenized_dataset = dataset.map(
    #     tokenize_batch,
    #     fn_kwargs={"max_length": max_length, "tokenizer": tokenizer},
    #     batched=True,
    #     remove_columns=["id", "dialogue", "summary"],
    # )

    # Filter out samples that have input_ids exceeding max_length
    # tokenized_dataset = tokenized_dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)

    return tokenized_dataset


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def eval(model, tokenizer, dataset):
    rougeL_list = []
    # nltk.download('punkt')
    data_cnt = len(dataset['test'])

    for data_idx in range(data_cnt):
        rouge = evaluate.load("rouge")
        input = dataset['test']['text'][data_idx]
        batch = tokenizer(input, return_tensors="pt")
        model.config.use_cache=False
        model.eval()
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=128)

        decoded_predictions = tokenizer.decode(output_tokens[0], skip_special_tokens=True, padding=True)
        decoded_labels = tokenizer.decode(dataset['test']['labels'][data_idx])
        # print(f"inference result:\n{decoded_predictions}\n")
        # print(f"reference-answer:\n{decoded_labels}")
        preds, targets = postprocess_text(preds=decoded_predictions, targets=decoded_labels)
        # result = rouge.compute(predictions=preds, references=targets, use_stemmer=True)
        result = rouge.compute(predictions=[decoded_predictions], references=[decoded_labels], use_stemmer=True)
        rougeL_list.append(copy.copy(result['rougeL']))
        mean_result = np.mean(np.array(rougeL_list))
        print(f"{data_idx}/{data_cnt}: {result['rougeL']} | Mean rouge-L: {mean_result}")
        

# def compute_metrics(eval_preds):
#     metric = evaluate.load("rouge")
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     # Replace -100s used for padding as we can't decode them
#     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Some simple post-processing
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     result = {k: round(v * 100, 4) for k, v in result.items()}
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     return result


def main():    
    # load model
    model_list = ("meta-llama/Llama-2-7b-hf", "gpt2") 
    MODEL_NAME = model_list[1]
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(MODEL_NAME, bnb_config)
    max_length = get_max_length(model)
    
    # load dataset
    dataset_list = ("e2e_nlg", "yahma/alpaca-cleaned", "samsum")
    DATASET = dataset_list[2]
    split_dataset = load_dataset(DATASET)
    
    # preprocess [train, test, validation] dataset
    for split_type in split_dataset.keys():
        print(f"formatting {split_type} dataset...")
        print(split_dataset[split_type])
        split_dataset[split_type] = split_dataset[split_type].map(
            preprocess_function,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "do_truncation": True},
            batched=True,
            remove_columns=["id", "dialogue", "summary"]
        )
    print(split_dataset)
        
    
    
    output_dir = f"results/{MODEL_NAME}/final_checkpoint"
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    lora_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        args=TrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # auto_find_batch_size=True,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            num_train_epochs=1,
            # max_steps=10,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=50,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
            eval_steps=1,
            eval_accumulation_steps=1,
            do_predict=True
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
        
    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    print(metrics)

    
    # evalute rouge score    
    eval(model, tokenizer, split_dataset)

if __name__ == "__main__":
    main()