import argparse

from datasets import load_dataset
from inference import DecodeCardinalityPipeline
from utils import MetricLogger, load_config, load_model_from_checkpoint


def test(config):
    model_dict = load_model_from_checkpoint(
        config.io.checkpoint_dir,
        device_map=config.inference.device_map,
        use_bf16=config.inference.use_bf16,
        pad_left=True
    )
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    dataset = load_dataset(config.io.dataset_name, split="test")

    pipe = DecodeCardinalityPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=config.inference.max_length,
        decode_mode=config.io.mode,
        device_map=config.inference.device_map,
    )

    logger = MetricLogger(log_file=config.io.log_file, metric_name="qerror")

    for i, data in enumerate(dataset):
        outputs = pipe(data)
        print(
            logger.log_and_return_metric(
                outputs["true_cardinality"],
                outputs["estimated_cardinality"],
                outputs[["qerror"]],
            )
        )
        if (i + 1) % config.io.output_step == 0:
            logger.print_running_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config YAML file path", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    test(config)
