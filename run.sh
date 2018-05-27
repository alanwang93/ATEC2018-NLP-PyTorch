#/bin/bash

python -m data.preprocess --mode 'test' --test_in $1
python test.py $1 $2
