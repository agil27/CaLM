import pandas as pd
import datasets
import argparse
from utils import load_config


def encode_decimal(cardinality: int):
    return str(cardinality)


def encode_binary(cardinality: int):
    return bin(cardinality)[2:]


def encode_science(cardinality: int):
    num_digits = len(str(cardinality))
    decimal_format = encode_decimal(cardinality)
    before_dot = decimal_format[0]
    if len(decimal_format) == 1:
        return before_dot + ".0e1"

    # Otherwise, more than one digit
    post_dot = decimal_format[1:]
    return before_dot + "." + post_dot + "e" + str(num_digits - 1)


def encode_cardinality(cardinality: int, mode: str):
    assert mode in ["decimal", "binary", "science"]
    if mode == "decimal":
        return encode_decimal(cardinality)
    elif mode == "binary":
        return encode_binary(cardinality)
    else:
        # mode == "science"
        return encode_science(cardinality)


def generate_dataset(
    token: str, dataset_prefix: str, encode_mode: str, input_csv_path: str
):
    # Process the data into HuggingFace dataset
    data = pd.read_csv(input_csv_path, sep="#", header=None)
    # A generator yielding texts from the training set.
    def generate_texts():
        for _, row in data.iterrows():
            projections = "SELECT " + row[0]
            predicates = []
            if isinstance(row[1], str) and len(row[1]) > 0:
                predicates.extend(row[1].split(","))
            if isinstance(row[2], str) and len(row[2]) > 0:
                range_predicate_str = row[2].replace(",<,", " < ")
                range_predicate_str = range_predicate_str.replace(",>,", " > ")
                range_predicate_str = range_predicate_str.replace(",<=,", " <= ")
                range_predicate_str = range_predicate_str.replace(",>=,", " >= ")
                range_predicate_str = range_predicate_str.replace(",=,", " = ")
                predicates.extend(range_predicate_str.split(","))
            predicates_str = " FROM T WHERE " + " AND ".join(predicates)
            prompt = """Below is a task to predict the cardinality
of a imagined database from the query.
### Query: %s
### Cardinality: """ % (
                projections + predicates_str
            )
            prompt = prompt.replace("\n", " ")
            true_cardinality = row[3]
            yield {
                "text": prompt + encode_cardinality(true_cardinality, mode),
                "prompt": prompt,
                "true_cardinality": true_cardinality,
            }
    dataset = datasets.Dataset.from_generator(generate_texts).train_test_split(test_size=0.2)
    dataset.push_to_hub(dataset_prefix + encode_mode, token=token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="config YAML file",
        default="configs/datagen_configs/gen.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for mode in ["binary", "decimal", "science"]:
        generate_dataset(
            config.token, config.dataset_prefix, mode, config.input_csv_path
        )
