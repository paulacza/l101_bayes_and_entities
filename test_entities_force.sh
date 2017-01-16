#!/bin/bash     

python main.py 20 0 0 0 0 1 1 >> results/entity_tests_force.txt 
python main.py 50 0 0 0 0 1 1 >> results/entity_tests_force.txt
python main.py 100 0 0 0 0 1 1 >> results/entity_tests_force.txt 
python main.py 500 0 0 0 0 1 1 >> results/entity_tests_force.txt
python main.py 1000 0 0 0 0 1 1 >> results/entity_tests_force.txt 
python main.py 2000 0 0 0 0 1 1 >> results/entity_tests_force.txt
python main.py 10000 0 0 0 0 1 1 >> results/entity_tests_force.txt 

echo "finished"
