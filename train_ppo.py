import argparse
import torch
import wandb

from datasets import load_dataset
from easydict import EasyDict as edict
from inference import batch_decode_cardinality_and_calc_qerror
from models import load_trl_model_from_checkpoint
from tqdm import tqdm
from trl import PPOConfig, PPOTrainer, set_seed
from utils import load_config, populate_default_arguments_for_config
from models import load_reward_model_from_checkpoint
from transformers import pipeline

def train_ppo(config: edict):
    model_and_tokenizer = load_trl_model_from_checkpoint(
        checkpoint_dir=config.io.sft_checkpoint,
        use_bf16=config.io.use_bf16,
        device_map=config.io.device_map,
        pad_left=True,
    )
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    # Prepare the dataset.
    def tokenize(sample):
        sample["input_id"] = tokenizer.encode(sample["prompt"])
        sample["query"] = sample["prompt"]
        return sample

    dataset = load_dataset(config.io.dataset_prefix + config.io.mode, split="train")
    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch")

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Set seed before initializing value head for deterministic eval
    set_seed(config.misc.seed)

    # The reference model is just a deep copy of the SFT model.
    ref_model = load_trl_model_from_checkpoint(
        checkpoint_dir=config.io.sft_checkpoint,
        use_bf16=config.io.use_bf16,
        device_map=config.io.device_map,
        pad_left=True,
    )["model"]

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_config = PPOConfig(
        model_name=config.model.model_name,
        learning_rate=config.training.learning_rate,
        log_with="wandb",
        remove_unused_columns=False
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    # The arguments passed to the PPO generate function to generate the model
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": config.inference.max_length,
    }

    reward_model, reward_tokenizer = load_reward_model_from_checkpoint("checkpoints/reward-/reward_checkpoint")
    reward_pipeline = pipeline("reward_model", model = reward_model, tokenizer=reward_tokenizer)

    for _ in range(config.training.num_train_epochs):
        # TODO: see the epochs
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_id"]

            # Get response from the active and the reference model
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                generate_ref_response=True,
                **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

            # Compute sentiment score
            decoded_active_model_outputs = batch_decode_cardinality_and_calc_qerror(
                batch["response"], batch["true_cardinality"], config.io.mode
            )
            decoded_ref_model_outputs = batch_decode_cardinality_and_calc_qerror(
                batch["ref_response"], batch["true_cardinality"], config.io.mode
            )

            # The reward is defined as the proportion of reduced / increased qerror.
            # TODO 1: normalize the rewards (use training set to find the cardinality of upper bound)
            # TODO 2: replace decoded_ref_model_outputs with classical method / baseline methods (orcal model methods)
            # TODO 3: tune the weights between rl loss and KL divergence loss
            # rewards = (
                # decoded_ref_model_outputs["qerror"] - decoded_active_model_outputs["qerror"]
            # ) / decoded_ref_model_outputs["qerror"]
            # UPPER_BOUND = 460456072
            # rewards = (
            #     decoded_ref_model_outputs["qerror"] - decoded_active_model_outputs["qerror"]
            # ) / UPPER_BOUND

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            # rewards = [torch.FloatTensor([r]) for r in rewards]
            batch["rewards"] = rewards

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, batch["rewards"])
            ppo_trainer.log_stats(
                stats,
                batch,
                batch["rewards"],
                columns_to_log=["query", "response", "rewards"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="training config YAML file path", required=True
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = populate_default_arguments_for_config(config)
    wandb.init(name=config.io.run_name)
    train_ppo(config)
