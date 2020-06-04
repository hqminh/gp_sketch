#!/bin/bash

nohup python -u vaegp.py ./results/abalone/ vaegp_16 abalone 0 2>&1 > ./results/abalone/abalone_vaegp_16_rmseopt.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_32 abalone 0 2>&1 > ./results/abalone/abalone_vaegp_32_rmseopt.log &
nohup python -u vaegp.py ./results/abalone/ vaegp_512 abalone 1 2>&1 > ./results/abalone/abalone_vaegp_512_rmseopt.log &
