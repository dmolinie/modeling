#!/bin/bash
# Create the `tests` folder, remove the files is may contain and enter is
mkdir -p tests/ && rm -f -r tests/* && cd tests/;
# Search for any `test*.py` file in the project and execute them
# from the `tests` folder (so possibly generated files will be there)
for test_file in $( find .. -iname test*.py ); do
    echo -e "\033[1m    Executing file $test_file \033[0m";
    python "$test_file";
    echo;
done;

