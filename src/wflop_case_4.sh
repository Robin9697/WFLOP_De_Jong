#!/bin/sh

# run program
echo "Starting program"
RUN_NUM_FILE='run_num_case_4.txt'
NUM_OLD=0
NUM_NEW=$((NUM_OLD+1))
for i in {1..1}
 do
 for num in {1..1}
  do 
  python Case_4_run.py &
  sleep 60
  sed -i "s/$NUM_OLD/$NUM_NEW/" $RUN_NUM_FILE
  NUM_OLD=$((NUM_OLD+1))  
  NUM_NEW=$((NUM_NEW+1)) 
 done
 # wait untill all scripts are done
 wait
done