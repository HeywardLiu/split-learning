import fedml
from fedml import FedMLRunner

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from data_utils import transform_data_to_fedml_format, group_texts, tokenize_function
from peft import LoraConfig, get_peft_model
from functools import partial


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
    
    
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True,
        cache_dir=args.model_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.model_cache_dir,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    lora_modules=["Wqkv"] 
    lora_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=lora_modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # load data
    block_size = tokenizer.model_max_length
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.data_cache_dir,
        streaming=args.dataset_streaming
    )
    column_names = list(raw_datasets["train"].features)
    
    # data preprocessing 
    __tokenize_function = partial(tokenize_function, text_column_name="text", tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(
                        __tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                        # load_from_cache_file=not data_args.overwrite_cache,
                        # desc="Running tokenizer on dataset",
                    )
    
    __group_texts = partial(group_texts, block_size=block_size)
    lm_datasets = tokenized_datasets.map(
                    __group_texts,
                    batched=True,
                    # num_proc=1,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    # desc=f"Grouping texts in chunks of {block_size}",
                )
    print(lm_datasets['train'])
    dataset = transform_data_to_fedml_format(args, train_dataset=lm_datasets['train'], test_dataset=lm_datasets['test'])
    print(dataset)
    # To do: Build SimulatorMPI of LLM + Split-learning 
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()