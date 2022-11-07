#!/bin/bash

BRANCH_TO_TEST=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
RESULT_DIR="/tmp/libmultilabel-test-$(whoami)/runs"
REPORT_PATH="test_report.txt"

#######################################
# Run the command and get the test results in the latest log file.
# Arguments:
#   $1: Path prefix of the log file.
#   $2: Command to run.
#######################################
get_test_results() {
  prefix=$1
  command="$2"

  echo "Running $command ..."
  $command

  log_file=$(ls -t $prefix*[0-9]*/logs.json | head -1)
  echo "Getting test results from $log_file ..."
  while IFS="=" read -r key value; do
      results[$key]=$value
  done < <(jq -r '.test[0]|to_entries|map("\(.key)=\(.value)")|.[]' $log_file)
}

#######################################
# Compare results between current branch and master branch.
# Arguments:
#   $1: Data name such as EUR-Lex, MIMIC-50, or rcv1.
#   $2: Network name defined in libmultilabel/nn/networks/*
#   $3: Command template to run.
#######################################
run_and_compare() {
  data_name=$1
  network_name=$2
  command=$(echo "$3" | sed "s/%s/$data_name/" | sed "s/%s/$network_name/")
  echo "Testing command: $command" >> $REPORT_PATH

  declare -A results actual_results
  prefix="${RESULT_DIR}/${data_name}_${network_name}_"

  # Get actual results in current branch
  get_test_results $prefix "$command"
  for i in "${!results[@]}"; do actual_results[$i]=${results[$i]}; done

  # Record the checkpoint directory for current branch
  current_dir=$(ls -t $prefix*[0-9]*/logs.json | head -1 | sed "s/logs.json//")

  # Get expected results in master branch
  git checkout master
  get_test_results $prefix "$command"

  # Record the checkpoint directory for master branch
  master_dir=$(ls -t $prefix*[0-9]*/logs.json | head -1 | sed "s/logs.json//")

  # Compare each API component
  if [ "$network_name" != "l2svm" ]; then
    python3 tests/compare_components.py --current_dir $current_dir --master_dir $master_dir >> $REPORT_PATH
  fi

  # Compare the results between current branch and master.
  cmp=$(echo ${results[@]} ${actual_results[@]} | tr ' ' '\n' | sort | uniq -u | wc -l)
  is_passed=$(echo $cmp | grep -q "0" && echo "PASSED" || echo "FAILED")
  echo "Test $is_passed!" & echo "results $is_passed" >> $REPORT_PATH &

  echo "Switch to $BRANCH_TO_TEST ..."
  git checkout $BRANCH_TO_TEST # back to current branch
}

main() {
  # Initialize the log directory and log file.
  mkdir -p $RESULT_DIR
  rm $REPORT_PATH

  TEST_COMMAND_TEMPLATES=(
    # Run 20% of the training data, 20% of the validation data, and 1% of the test data for 2 epochs.
    "python3 run_and_store_results.py --config example_config/%s/%s.yml --result_dir $RESULT_DIR --limit_train_batches 0.2 --limit_val_batches 0.2 --limit_test_batches 0.01 --epochs 2"
  )
  for template in "${TEST_COMMAND_TEMPLATES[@]}"; do
    run_and_compare "EUR-Lex" "kim_cnn" "$template"
    run_and_compare "MIMIC-50" "caml" "$template"
  done

  TEST_COMMAND_TEMPLATES=(
    # Run default linear 1vsrest
    "python3 main.py --config example_config/%s/%s.yml --result_dir $RESULT_DIR"
  )
  for template in "${TEST_COMMAND_TEMPLATES[@]}"; do
    run_and_compare "rcv1" "l2svm" "$template"
  done

  # Print the test results and remove the intermediate files.
  all_tests=$(grep 'PASSED\|FAILED' $REPORT_PATH | wc -l)
  passed_tests=$(grep "PASSED" $REPORT_PATH | wc -l)
  echo "All tests finished ($passed_tests/$all_tests) on $BRANCH_TO_TEST. See $REPORT_PATH for details."
  rm -r $RESULT_DIR
}

#######################################
# Please run this script in the LibMultilabel directory.
# Usage:
#   bash tests/autotest.sh
#######################################
if $(echo $(pwd) | grep -q "tests"); then
  echo "Please run this script in the LibMultilabel directory."
  echo "Go to the LibMultilabel directory and run: bash tests/autotest.sh"
else
  main
fi
