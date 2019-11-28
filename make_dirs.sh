#!/bin/bash

cd experimental_results

mkdir cur_run$1
cd cur_run$1

for i in `seq 0 5`;
do
	mkdir curriculum$i
done

cd ../..
cd saves/models
mkdir cur_run$1
cd cur_run$1

for i in `seq 0 5`;
do
	mkdir curriculum$i
done	
