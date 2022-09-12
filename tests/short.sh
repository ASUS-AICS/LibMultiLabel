#! /bin/bash

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

find example_config -name "*.yml" -type f -print0 |
    while IFS= read -r -d '' config; do
        if [[ $config == *tune.yml ]]; then
            script=search_params.py
            continue
        else
            script=main.py
        fi
        echo "Running $config"
        stderr=$(python $script --config "$config" --epochs 1 \
            --result_dir "$result_dir" 2>&1 > /dev/null)
        if [[ $? -ne 0 ]]; then
            echo "$stderr"
            exit 1
        fi
    done
