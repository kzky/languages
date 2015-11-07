#!/bin/sh
rm -rf *_write_*.out

n=9

echo "experiment_write_001"
for i in $(seq 0 $n); do 
    python experiment_write_001.py >> experiment_write_001.out
done

echo "# experiment_write_002"
for i in $(seq 0 $n); do 
    python experiment_write_002.py >> experiment_write_002.out
done

echo "experiment_write_003"
for i in $(seq 0 $n); do 
    python experiment_write_003.py >> experiment_write_003.out
done


