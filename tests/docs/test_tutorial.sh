#!/bin/bash

BRANCH_TO_TEST=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
REPORT_PATH="test_tutorial_report.txt"
LOG_PREFIX="test_tutorial"

update_libmultilabel() {
  pip install -U libmultilabel
}

run_with_logs() {
  branch=$1
  prefix=$2
  command="$3"

  log_path="$prefix_$branch.log"
  echo "Test $command and save logs to ${log_path}."
  command="$command > ${log_path} &"
  $command
}

#######################################
# Compare logs between current branch and master branch. (out_$branch.log)
# Arguments:
#   $1: Command to run.
#######################################
run_and_compare_logs() {
  command="$1"

  run_with_logs $BRANCH_TO_TEST $LOG_PREFIX "$command"
  git checkout master
  run_with_logs "master" $LOG_PREFIX "$command"
  git checkout $BRANCH_TO_TEST # back to current branch

  n=$(tail -${tail_n} ${LOG_PREFIX}_master.log | grep -n "Test metric" | cut -c1)
  if [ -z $n ]; then n=1; else n=$((10-$n)); fi
  branch_test_result=$(tail -$n ${LOG_PREFIX}_${BRANCH_TO_TEST}.log)
  master_test_result=$(tail -$n ${LOG_PREFIX}_master.log)
  is_passed=$([ "$branch_test_result" = "$master_test_result" ] && echo "PASSED" || echo "FAILED")
  echo "Test $is_passed!" & echo "results $is_passed" >> $REPORT_PATH &

  # remove logs
  # rm out_${BRANCH_TO_TEST}.log
  # rm out_master.log
}

main() {
  rm $REPORT_PATH
  TEST_FILES=(
    "linear_quickstart.py"
    # "kimcnn_quickstart.py"
    # "bert_quickstart.py"
  )
  for file_name in "${TEST_FILES[@]}"; do
    command="python3 docs/examples/${file_name}"
    run_and_compare_logs "$command"
  done
}

#######################################
# Please run this script in the LibMultilabel directory.
# Usage:
#   bash tests/docs/test_tutorial.sh
#######################################
if $(echo $(pwd) | grep -q "tests"); then
  echo "Please run this script in the LibMultilabel directory."
  echo "Go to the LibMultilabel directory and run: bash tests/autotest.sh"
else
  # update_libmultilabel
  main
fi
