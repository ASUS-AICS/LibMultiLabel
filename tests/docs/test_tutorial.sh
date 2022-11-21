#!/bin/bash

REPORT_PATH="test_tutorial_report.txt"

update_libmultilabel() {
    pip install -U libmultilabel
}

main() {
  # Initialize the report file
  rm $REPORT_PATH

  TEST_COMMANDS=(
    "python3 docs/examples/linear_quickstart.py"
    "python3 docs/examples/kimcnn_quickstart.py"
    "python3 docs/examples/bert_quickstart.py"
  )

  for command in "${TEST_COMMANDS[@]}"; do
    echo "Testing command: $command"
    if $command; then
      is_passed="PASSED"
    else
      is_passed="FAILED"
    fi
    # is_passed=$(echo $command | grep -q "0" && echo "PASSED" || echo "FAILED")
    echo "${is_passed}    $command " >> $REPORT_PATH &
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
  update_libmultilabel
  main
fi
