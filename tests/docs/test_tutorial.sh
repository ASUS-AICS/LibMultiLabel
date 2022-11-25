#!/bin/bash

BRANCH_TO_TEST=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
REPORT_PATH="test_tutorial_report.txt"

update_libmultilabel() {
  pip install -U libmultilabel
}

#######################################
# Compare results between current branch and master branch.
# Arguments:
#   $1: Command to run.
#######################################
run_and_compare_logs() {
  command=$1
  echo "Testing command: $command"
  eval "$command > out_${BRANCH_TO_TEST}.log &"
  git checkout master
  eval "$command > out_master.log &"
  git checkout $BRANCH_TO_TEST # back to current branch

  # Compare log results
  n=$(tail -${tail_n} out_master.log | grep -n "Test metric" | cut -c1)
  if [ -z $n ]; then n=1; else n=$((10-$n)); fi
  branch_test_result=$(tail -$n out_${BRANCH_TO_TEST}.log)
  master_test_result=$(tail -$n out_master.log)
  is_passed=$([ "$branch_test_result" = "$master_test_result" ] && echo "PASSED" || echo "FAILED")
  echo "Test $is_passed!" & echo "results $is_passed" >> $REPORT_PATH &

  # remove logs
  rm out_${BRANCH_TO_TEST}.log
  rm out_master.log
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
