#!/bin/bash

echo "Compute with copyng to host"
for n in $(seq 1 100); do
	m=$(($n * 1000000))
	sort.out $m
done	
