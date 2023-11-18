# CaLM

Cardinality prediction language model

## Usage

1. Provide a config YAML file and run training script:

```bash
python3 train.py -c <your-config-file>
```

The training script will generate a snapshot directory (under the output director you configure in the YAML file) and will generate a test config YAML file under that snapshot directory.

1. Run the test script:

```bash
python3 test.py -c <generated-test-config-file>
```