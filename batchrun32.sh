#!/bin/bash

nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 2603 2>&1 > ./results/abalone/abalone_vaegp_32_s1.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 411 2>&1 > ./results/abalone/abalone_vaegp_32_s2.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 807 2>&1 > ./results/abalone/abalone_vaegp_32_s3.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 1002 2>&1 > ./results/abalone/abalone_vaegp_32_s4.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 1008 2>&1 > ./results/abalone/abalone_vaegp_32_s5.log &
