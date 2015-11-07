#!/bin/sh
rm -rf *_read_*.out

n=9

echo "experiment_read_001"
for i in $(seq 0 $n); do 
    python experiment_read_001.py >> experiment_read_001.out
done

echo "experiment_read_002"
for i in $(seq 0 $n); do 
    python experiment_read_002.py >> experiment_read_002.out
done

echo "experiment_read_003"
for i in $(seq 0 $n); do 
    python experiment_read_003.py >> experiment_read_003.out
done


