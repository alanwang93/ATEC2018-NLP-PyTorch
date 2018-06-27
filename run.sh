#/bin/bash

python -m data.process  --mode 'test' --test_in $1
python -m test $1 $2
