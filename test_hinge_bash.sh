#!/bin/sh
for i in 1 2 3 4 5
do
  python3 test_hinge.py $1
done
python3 test_hinge.py --write $1 
echo "All done"