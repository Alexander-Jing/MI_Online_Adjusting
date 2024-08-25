#!/bin/bash

export PYTHONPATH="${PYTHONPATH}/home/jyt/workspace/MI_Online"
cd ${PYTHONPATH}/Offline_synthesizing_results/
python synthesize_hypersearch_for_a_subject.py  --experiment_dir ${PYTHONPATH}/Offline_experiments/Jyt 