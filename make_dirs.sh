#!/bin/bash
cd experimental_results/all_tasks

declare -a tasks=("parts_task" "similarity_objects" "count" "select")

for t in "${tasks[@]}"
do
    if [ -d t ]; then
        cd $t
    else
        mkdir $t
        cd $t
    fi
    mkdir run$1
    cd run$1
    for i in `seq 1 5`;
    do
        mkdir curriculum$i
    done
    cd ../..
done