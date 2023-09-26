from datasets import Dataset
from fedml.arguments import Arguments

def transform_data_to_fedml_format(args: Arguments, train_dataset: Dataset, test_dataset: Dataset):
    # TODO: scrutinize
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    if args.rank == 0:
        # server data
        test_data_global = test_dataset['text']
    else:
        # client data
        train_data_local_num_dict[args.rank - 1] = train_dataset['num_rows']
        train_data_local_dict[args.rank - 1] = train_dataset['text']
        test_data_local_dict[args.rank - 1] = test_dataset['text']
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