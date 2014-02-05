#!/bin/sh

python data_to_vw.py train.csv > train.vw
python data_to_vw.py test-full.csv > test.vw

###vw train.vw -k -c -f model --passes 10
vw train.vw -k -c -f model --passes 50 --l1 0.00001

###vw train.vw -k -c -f model --passes 200 --l1 0.00005 -q cl -qct -q lt
###vw -d train.vw -k -c -f model --cubic clt --passes 60 --l1 1.5e-6 --cubic ttt -q tt

vw -k -c -t test.vw -i model -p test.predict
python mae.py test-full.csv test.predict 
