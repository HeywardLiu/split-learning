from datasets import Dataset
from fedml.arguments import Arguments
from itertools import chain
from transformers.testing_utils import CaptureLogger
import transformers
from torch.utils.data import DataLoader

def group_texts(examples, block_size):
    
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_function(examples, text_column_name, tokenizer):
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output




def transform_data_to_fedml_format(args: Arguments, train_dataset: Dataset, test_dataset: Dataset):
    """
    args: Dataset
    return: Dataloader
    """
    # TODO: scrutinize
    train_data_num = train_dataset.num_rows
    test_data_num = test_dataset.num_rows
    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    if args.rank == 0:
        # server data
        train_data_global = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        test_data_global = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
    else:
        # client data
        train_data_local_num_dict[args.rank - 1] = train_dataset.num_rows
        train_data_local_dict[args.rank - 1] = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        test_data_local_dict[args.rank - 1] = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
        
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        2
    )