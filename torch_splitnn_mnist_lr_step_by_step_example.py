import fedml
from fedml import FedMLRunner

from transformers import AutoModelForCausalLM
from datasets import load_dataset
from data_utils import transform_data_to_fedml_format

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.data_cache_dir,
        streaming=args.dataset_streaming
    )
    dataset = transform_data_to_fedml_format(args, train_dataset=dataset['train'], test_dataset=dataset['test'])
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.model_cache_dir,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    print(dataset)
    # To do: Build SimulatorMPI of LLM + Split-learning 
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()