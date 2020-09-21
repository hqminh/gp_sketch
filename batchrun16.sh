#!/bin/bash

nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 2603 2>&1 > ./results/abalone/abalone_vaegp_16_s1.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 411 2>&1 > ./results/abalone/abalone_vaegp_16_s2.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 807 2>&1 > ./results/abalone/abalone_vaegp_16_s3.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 1002 2>&1 > ./results/abalone/abalone_vaegp_16_s4.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 1008 2>&1 > ./results/abalone/abalone_vaegp_16_s5.log &
