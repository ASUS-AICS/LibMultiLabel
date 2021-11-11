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
# Test results between current branch and master branch.
# Arguments:
#   $1: Data name such as MIMIC-50 or rcv1.
#   $2: Nework name defined in libmultilabel/nn/networks/*
#   $3: Command template to run.
#######################################
run_test() {
  data_name=$1
  network_name=$2
  command=$(echo "$3" | sed "s/%s/$data_name/" | sed "s/%s/$network_name/") # sscanf...

  declare -A results actual_results
  prefix="${RESULT_DIR}/${data_name}_${network_name}_"

  # Get actual results in current branch
  get_test_results $prefix "$command"
  for i in "${!results[@]}"; do actual_results[$i]=${results[$i]}; done

  # Get expected results in master branch
  git checkout auto-test
  get_test_results $prefix "$command"

  # Compare the results between current branch and master.
  cmp=$(echo ${results[@]} ${actual_results[@]} | tr ' ' '\n' | sort | uniq -u | wc -l)
  is_passed=$(echo $cmp | grep -q "0" && echo "PASSED" || echo "FAILED")
  echo "Test $is_passed!" & echo "$is_passed   $command" >> $REPORT_PATH &

  echo "Switch to $BRANCH_TO_TEST ..."
  git checkout $BRANCH_TO_TEST # back to current branch
}

main() {
  # Initialize the log directory and log file.
  mkdir -p $RESULT_DIR
  rm $REPORT_PATH

  # Run the tests.
  TEST_COMMAND_TEMPLATES=(
    "python3 main.py --config example_config/%s/%s.yml --result_dir $RESULT_DIR --limit_train_batches 0.2 --limit_val_batches 0.2 --limit_test_batches 0.01 --epochs 2"
  )
  for template in "${TEST_COMMAND_TEMPLATES[@]}"; do
    run_test "MIMIC-50" "caml" "$template"
    run_test "rcv1" "kim_cnn" "$template"
  done

  all_tests=$(less $REPORT_PATH | wc -l)
  passed_tests=$(grep "PASSED" $REPORT_PATH | wc -l)
  echo "All tests finished ($passed_tests/$all_tests) on $BRANCH_TO_TEST. See $REPORT_PATH for details."
  rm -r $RESULT_DIR
}

main