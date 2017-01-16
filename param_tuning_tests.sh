#!/bin/bash          

# testing what text preprocessing works best
python main.py 100 0 0 0 0 0 0 >> results/params_tuning.txt 
python main.py 100 0 0 0 1 0 0 >> results/params_tuning.txt 
python main.py 100 0 0 1 0 0 0 >> results/params_tuning.txt 
python main.py 100 0 0 1 1 0 0 >> results/params_tuning.txt 

# testing stoplists
python main.py 100 0 1 0 0 0 0 >> results/params_tuning.txt 
python main.py 100 0 1 0 1 0 0 >> results/params_tuning.txt 

# testing stemming
python main.py 100 1 0 0 0 0 0 >> results/params_tuning.txt 
python main.py 100 1 0 0 1 0 0 >> results/params_tuning.txt 

# testing both stoplists and stemming
python main.py 100 1 1 0 0 0 0 >> results/params_tuning.txt 
python main.py 100 1 1 0 1 0 0 >> results/params_tuning.txt

echo "Finished."
