#!/bin/bash

INSTALLATION_FOLDER="installation_files"

GSIM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


cd $GSIM_DIR

cp $INSTALLATION_FOLDER"/run_experiment.py" .. 
cp $INSTALLATION_FOLDER"/gsim_conf-base.py" .. 
cp $INSTALLATION_FOLDER"/gsim_conf-base.py" "../gsim_conf.py"

mkdir -p ../experiments
cp $INSTALLATION_FOLDER"/example_experiments.py" ../experiments/
