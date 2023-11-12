import pandas as pd
import datasets

# Use token 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

from lib import TOKEN


def generate_decimal_dataset():
    # Process the data into HuggingFace dataset
    data = pd.read_csv("train.csv", sep="#", header=None)

    # A generator yielding texts from the training set.
    def generate_texts():
        for i, row in data.iterrows():
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
            yield {
                "text": (
                    """Below is a task to predict the cardinality
    of a imagined database from the query.
    ### Query: %s
    ### Cardinality: %d"""
                    % (projections + predicates_str, row[3])
                ).replace("\n", "")
            }

    dataset = datasets.Dataset.from_generator(generate_texts)
    dataset.push_to_hub("vic0428/imdb-card-pred-decimal", token=TOKEN)


def generate_binary_dataset():
    # Process the data into HuggingFace dataset
    data = pd.read_csv("train.csv", sep="#", header=None)

    # A generator yielding texts from the training set.
    def generate_texts():
        for i, row in data.iterrows():
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
            sample = {
                "text": (
                    """Below is a task to predict the cardinality
    of a imagined database from the query.
    ### Query: %s
    ### Cardinality: %s"""
                    % (projections + predicates_str, bin(row[3])[2:])
                ).replace("\n", "")
            }
            yield sample

    dataset = datasets.Dataset.from_generator(generate_texts)
    dataset.push_to_hub("vic0428/imdb-card-pred-binary", token=TOKEN)


def generate_scientific_notation_dataset():
    # Process the data into HuggingFace dataset
    data = pd.read_csv("train.csv", sep="#", header=None)

    # A generator yielding texts from the training set.
    def generate_texts():
        for i, row in data.iterrows():
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
            sample = {
                "text": (
                    """Below is a task to predict the cardinality
    of a imagined database from the query.
    ### Query: %s
    ### Cardinality: %s"""
                    % (projections + predicates_str, format(row[3], "e"))
                ).replace("\n", "")
            }
            yield sample

    dataset = datasets.Dataset.from_generator(generate_texts)
    dataset.push_to_hub("vic0428/imdb-card-pred-science", token=TOKEN)


if __name__ == "__main__":
    generate_binary_dataset()
    generate_scientific_notation_dataset()
