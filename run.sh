#/bin/bash

python -m data.preprocess  --tokenize --embed --extract --mode 'test' --test_in $1
python -m test $1 $2
