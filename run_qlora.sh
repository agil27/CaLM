# We know that batch size 20 can converge
batch_size=(20)

# Iterate over array elements
for bs in "${batch_size[@]}"; do
    echo "Try batch size $bs"
    python3 train.py vic0428/imdb-card-pred-binary results_nov10_binary-bs$bs results_nov10_binary-bs$bs --batchSize $bs
    python3 train.py vic0428/imdb-card-pred-decimal results_nov10_decimal-bs$bs results_nov10_decimal-bs$bs --batchSize $bs
    python3 train.py vic0428/imdb-card-pred-science results_nov10_science-bs$bs results_nov10_science-bs$bs --batchSize $bs
    done
