#!/bin/bash

BRANCH_TO_TEST=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
REPORT_PATH="test_doc_examples_report.txt"
LOG_PREFIX="out"

update_libmultilabel() {
  pip3 install -U libmultilabel
}

#######################################
# Compare logs between current branch and master branch.
# Arguments:
#   $1: Command to run.
#######################################
run_and_compare_logs() {
  command="$1"
  $command >> ${LOG_PREFIX}_${BRANCH_TO_TEST}.log
  git checkout master
  $command >> ${LOG_PREFIX}_master.log
  git checkout $BRANCH_TO_TEST # back to current branch

  # Compare the test results in logs. We grep the last 10 lines to ignore tqdm above.
  n=$(tail -10 ${LOG_PREFIX}_master.log | grep -n "Test metric" | cut -c1)
  if [ -z $n ]; then n=1; else n=$((10-$n)); fi
  branch_res=$(tail -$n "${LOG_PREFIX}_${BRANCH_TO_TEST}.log")
  master_res=$(tail -$n "${LOG_PREFIX}_master.log")
  is_passed=$([ "$branch_res" = "$master_res" ] && echo "PASSED" || echo "FAILED")
  echo "$is_passed  $command" >> $REPORT_PATH &

  # Remove temporary files.
  rm ${LOG_PREFIX}_${BRANCH_TO_TEST}.log
  rm ${LOG_PREFIX}_master.log
}

#######################################
# Check if error occur.
# Arguments:
#   $1: Command to run.
#######################################
run() {
  command="$1"
  # Simply run the code and test if error occur.
  $command
  is_passed=$([ $? = 0 ] && echo "PASSED" || echo "FAILED")
  echo "$is_passed  $command" >> $REPORT_PATH &
}

main() {
  rm $REPORT_PATH
  TEST_FILES_WithoutOutput=(
    "plot_linear_gridsearch_tutorial.py"
    "plot_dataset_tutorial.py"
  )
  for file_name in "${TEST_FILES_WithoutOutput[@]}"; do
    command="python3 docs/examples/${file_name}"
    run "$command"
  done

  TEST_FILES_WithOutput=(
    "plot_linear_quickstart.py"
    "plot_KimCNN_quickstart.py"
    "plot_bert_quickstart.py"
  )
  for file_name in "${TEST_FILES_WithOutput[@]}"; do
    echo ${file_name}
    command="python3 docs/examples/${file_name}"
    run_and_compare_logs "$command"
  done

  # Print the test results and remove the intermediate files.
  all_tests=$(grep 'PASSED\|FAILED' $REPORT_PATH | wc -l)
  passed_tests=$(grep "PASSED" $REPORT_PATH | wc -l)
  echo "All tests finished ($passed_tests/$all_tests) on $BRANCH_TO_TEST. See $REPORT_PATH for details."
}

#######################################
# Please run this script in the LibMultilabel directory.
# Usage:
#   bash tests/docs/test_examples.sh
#######################################
if $(echo $(pwd) | grep -q "tests"); then
  echo "Please run this script in the LibMultilabel directory."
  echo "Go to the LibMultilabel directory and run: bash tests/autotest.sh"
else
  update_libmultilabel
  main
fi
