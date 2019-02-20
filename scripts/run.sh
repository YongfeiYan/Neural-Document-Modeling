#!/bin/bash

PATH="/home/yyf/anaconda3/bin:$PATH"
echo "PATH:"
echo "$PATH"
echo "python path:"
which python

export PYTHONPATH=`pwd`:$PYTHONPATH


function mydisown(){
	# The function should be called 
	pid=$!
	if [ -n "$1" ]; then
		pid=$1
	fi
	echo "disown pid: $pid"
	disown $pid
	echo "jobs:"
	jobs
}


OUTPUT="`pwd`/data/output"
ls -l .
echo "========================Begin run.sh=========================="

mkdir -p data/exp
mkdir -p data/output
mkdir -p data/nips

export PYTHONPATH="`pwd`":"$PYTHONPATH"


###############################
# NTM
###############################

log=data/output/ntm_`date | sed 's/[ :]/_/g'`_$RANDOM
# export CUDA_VISIBLE_DEVICES=0
# nvdm
# python -u experiments/ntm.py -F data/exp/nvdm with config_file='data/config/nvdm.yaml' &> $log &
# ntm
# python -u experiments/ntm.py -F data/exp/ntm with config_file='data/config/ntm.yaml' &> $log &
# gsm
# python -u experiments/ntm.py -F data/exp/gsm with config_file='data/config/gsm.yaml' &> $log &
# ntmr
# python -u experiments/ntm.py -F data/exp/ntmr with config_file='data/config/ntmr.yaml' &> $log &
# my
python -u experiments/ntm.py -F data/exp/my with config_file='data/config/my.yaml' &> $log &
# 
mydisown

tail -f $log
echo "Finished run.sh"


# sed '/^[\/ ]/d' data/output/output | less



