#! /bin/bash

#######################################
# This is a basic test running example configs on fake data.
# The test stops at the first error and exits with non-zero status.
# Please run this script in the LibMultilabel directory.
# Usage:
#   tests/basic.sh [filters...]
# filters restricts the tested configs that have paths containing all values.
#
# Examples:
#   Test all configs:
#       tests/basic.sh
#   Test KimCNN on all datasets:
#       tests/basic.sh kim
#   Test all models on rcv1:
#       tests/basic.sh rcv1
#######################################

result_dir="/tmp/libmultilabel-test-$(whoami)/runs"
datasets=(
    MIMIC-50
    rcv1
    EUR-Lex
    EUR-Lex-57k
    Wiki10-31K
    MIMIC
)
datasets_with_val=(
    MIMIC
    MIMIC-50
    EUR-Lex-57k
)
datasets_with_embed=(
    MIMIC
    MIMIC-50
)
datasets_with_vocab=(
    MIMIC
    MIMIC-50
)
pretrained_embed=(
    charngram.100d
    fasttext.en.300d
    fasttext.simple.300d
    glove.42B.300d
    glove.840B.300d
    glove.twitter.27B.25d
    glove.twitter.27B.50d
    glove.twitter.27B.100d
    glove.twitter.27B.200d
    glove.6B.50d
    glove.6B.100d
    glove.6B.200d
    glove.6B.300d
)

if [[ -d data ]]; then
    echo "Please move the directory data, $0 overwrites it"
    exit 1
fi

function cleanup()
{
    status=$?
    rm -r $result_dir
    rm -r data
    exit $status
}
trap cleanup EXIT

lml_labels=$(seq -s ' ' 100)
lml_text=$(printf 'lorem %.0s' {1..10})
lml_data=$lml_labels$'\t'$lml_text
svm_labels=$(seq -s ',' 100)
svm_features=$(echo {1..10}':0.1')
svm_data="$svm_labels $svm_features"
for d in ${datasets[@]}; do
    mkdir -p "data/$d"
    for i in {1..10}; do
        echo "$lml_data" >> "data/$d/train.txt"
        echo "$lml_data" >> "data/$d/test.txt"
        echo "$svm_data" >> "data/$d/train.svm"
        echo "$svm_data" >> "data/$d/test.svm"
    done
done

for d in ${datasets_with_val[@]}; do
    for i in {1..10}; do
        echo "$lml_data" >> "data/$d/valid.txt"
    done
done

for d in ${datasets_with_embed[@]}; do
    echo 'lorem 0.1' > "data/$d/processed_full.embed"
    echo 'lorem 0.1' > "data/$d/processed_full.embed"
done

for d in ${datasets_with_vocab[@]}; do
    echo 'lorem' > "data/$d/vocab.csv"
    echo 'lorem' > "data/$d/vocab.csv"
done

for e in ${pretrained_embed[@]}; do
    echo lorem $(seq -f '0.%g' 300) > "data/$e.txt"
done

tmp=("${@/#/-regex .*}")
filters=${tmp[@]/%/.*}
find example_config -name "*.yml" ${filters[@]} -type f -print0 |
    while IFS= read -r -d '' config; do
        if [[ $config == *tune.yml ]]; then
            script=search_params.py
            continue # parameter search is not ready for testing
        else
            script=main.py
        fi
        echo "Running $config"
        stderr=$(python $script --config "$config" --epochs 1 \
            --result_dir "$result_dir" --embed_cache_dir data \
            --cpu 2>&1 > /dev/null)
        if [[ $? -ne 0 ]]; then
            echo "$stderr" >&2
            exit 1
        fi
    done
