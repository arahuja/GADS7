#!/bin/sh


#python parse_to_vw.py
vw -d ${1} --lda 10 \
--lda_alpha 0.1 \
--lda_rho 0.1 \
--lda_D 1980686 \
--minibatch 2056 \
-b 16 \
--power_t 0.5 \
--initial_t 1 \
-p ${1}-predictions.dat \
--readable_model ${1}-topics.dat
