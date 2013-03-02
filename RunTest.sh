#!/bin/bash

NVPROF=nvprof  # Name of profiling cmmand
PROFFLAGS="--profile-from-start-off --devices 0 --events"
EXEC=main      # Name of executable

# Check that 
if [ ! -f $EXEC ]; then
    echo "FILE $EXEC DOES NOT EXIST"
    echo "PLEASE RUN MAKE"
    exit 
fi

function profile {
    SHELLCOMMAND=$1
    NVCOMMANDS="$NVPROF $PROFFLAGS"
    
    # Get L1 load hit and miss
    FLAGS="l1_global_load_miss,l1_global_load_hit"
    COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
#    echo $COMMANDS
    eval RESULT='$('$COMMANDS')'

    L1_AVG_HIT=$( echo "$RESULT" | grep hit | awk --field-separator=" " '{print $2 }')
    L1_AVG_MISS=$( echo "$RESULT" | grep miss | awk --field-separator=" " '{print $2 }')
  
    # Get total number of load and store requests
    FLAGS="gld_request,gst_request"
    COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
    eval RESULT='$('$COMMANDS')'

    GLD_AVG=$( echo "$RESULT" | grep gld | awk --field-separator=" " '{print $2 }')
    GST_AVG=$( echo "$RESULT" | grep gst | awk --field-separator=" " '{print $2 }')

    # Get total number of store transactions
    FLAGS="global_store_transaction"
    COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
    eval RESULT='$('$COMMANDS')'

    STORE_TRANS_AVG=$( echo "$RESULT" | grep global | awk --field-separator=" " '{print $2 }')

    echo "Global load request: " $GLD_AVG
    echo "L1 Hit: " $L1_AVG_HIT " L1 Miss: " $L1_AVG_MISS
    L1_TOTAL_TRANF=$(echo "$L1_AVG_HIT+$L1_AVG_MISS" | bc)
    echo "Actual loads: " $L1_TOTAL_TRANF  " L1 hit rate " $(echo "scale=1; ($L1_AVG_HIT/$L1_TOTAL_TRANF)*100" | bc)"%"
    LOAD_RATIO=$(echo "scale=2; $L1_TOTAL_TRANF/$GLD_AVG" | bc)
    echo "Ratio: " $LOAD_RATIO 
    echo " "

    echo "Global store requests: " $GST_AVG
    echo "Actual stores: $STORE_TRANS_AVG"
    STORE_RATIO=$(echo "scale=2; $STORE_TRANS_AVG/$GST_AVG" | bc)
    echo "Ratio: $STORE_RATIO"
    echo " "
}

# The important metric from nvprof are:
# Actual load
# Actual store
# Requested load
# Requested store

# Run Test 1
echo "RUNNING TEST 1"
profile "./$EXEC"

#METRIC="l1_global_load_miss,l1_global_load_hit"
#COMMANDS="$NVPROF $PROFFLAGS l1_global_load_miss,l1_global_load_hit ./$EXEC"
#eval L1='$('$COMMANDS')'

#L1_AVG_HIT=$( echo "$L1" | grep hit | awk --field-separator=" " '{print $2 }')
#L1_AVG_MISS=$( echo "$L1" | grep miss | awk --field-separator=" " '{print $2 }')

#echo "L1 Hit: " $L1_AVG_HIT " L1 Miss: " $L1_AVG_MISS
#L1_TOTAL_TRANF=$(echo "$L1_AVG_HIT+$L1_AVG_MISS" | bc)
#echo "Total memory load tranferes: $L1_TOTAL_TRANF"


#$($NVPROF --profile-from-start-off --devices 0 --events l1_global_load_miss,l1_global_load_hit ./$EXEC )
#$(NVPROF) --profile-from-start-off --devices 0 --events gld_request,gst_request,global_store_transaction ./$(OUTPUT) 
#	 GLOBAL_STORE_REQUEST=$($(NVPROF) --profile-from-start-off --devices 0 --events global_store_transaction ./$(OUTPUT) | grep global)