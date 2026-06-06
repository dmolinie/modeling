#!/bin/bash
# Search for any `__pycache__.pyc` file in the project and remove them
for pyc_folder in $( find .. -iname __pycache__ ); do
    rm -r "$pyc_folder";
done;
# Also remove the `tests` folder
rm -f -r tests;

