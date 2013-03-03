#!/bin/bash

NVPROF=nvprof  # Name of profiling cmmand
PROFFLAGS="--profile-from-start-off --devices 0"
EXEC=main      # Name of executable
DOPROFTYPE=1   # 1 is event counters
               # 2 is kernel run time

# Check that 
if [ ! -f $EXEC ]; then
    echo "FILE $EXEC DOES NOT EXIST"
    echo "PLEASE RUN MAKE"
    exit 
fi

OFFSET=$1

function profile {
    local SHELLCOMMAND=$1
    local PROFTYPE=$2
    local NVCOMMANDS="$NVPROF $PROFFLAGS"
    
    # Profile events
    if [ $PROFTYPE -eq 1 ]; then
        # Get L1 load hit and miss
	NVCOMMANDS="$NVCOMMANDS --events"
	local FLAGS="l1_global_load_miss,l1_global_load_hit"
	local COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
	eval RESULT='$('$COMMANDS')'
	
	local L1_AVG_HIT=$( echo "$RESULT" | grep hit | awk --field-separator=" " '{print $2 }')
	local L1_AVG_MISS=$( echo "$RESULT" | grep miss | awk --field-separator=" " '{print $2 }')
	
        # Get total number of load and store requests
	local FLAGS="gld_request,gst_request"
	local COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
	eval RESULT='$('$COMMANDS')'
	
	local GLD_AVG=$( echo "$RESULT" | grep gld | awk --field-separator=" " '{print $2 }')
	local GST_AVG=$( echo "$RESULT" | grep gst | awk --field-separator=" " '{print $2 }')
	
        # Get total number of store transactions
	local FLAGS="global_store_transaction"
	local COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
	eval RESULT='$('$COMMANDS')'
	
	local STORE_TRANS_AVG=$( echo "$RESULT" | grep global | awk --field-separator=" " '{print $2 }')
	
	echo "Global load request: " $GLD_AVG
	echo "L1 Hit: " $L1_AVG_HIT " L1 Miss: " $L1_AVG_MISS
	local L1_TOTAL_TRANF=$(echo "$L1_AVG_HIT+$L1_AVG_MISS" | bc)
	echo "Actual loads: " $L1_TOTAL_TRANF  " L1 hit rate " $(echo "scale=1; ($L1_AVG_HIT/$L1_TOTAL_TRANF)*100" | bc)"%"
	if [ $GLD_AVG -ne 0 ]; then
	    local LOAD_RATIO=$(echo "scale=2; $L1_TOTAL_TRANF/$GLD_AVG" | bc)
	else
	    local LOAD_RATIO=0
	fi
	echo "Ratio: " $LOAD_RATIO 
	echo " "
	
	echo "Global store requests: " $GST_AVG
	echo "Actual stores: $STORE_TRANS_AVG"
	if [ $GST_AVG -ne 0 ]; then
	    local STORE_RATIO=$(echo "scale=2; $STORE_TRANS_AVG/$GST_AVG" | bc)
	else 
	    local STORE_RATIO=0
	fi
	echo "Ratio: $STORE_RATIO"
	echo " "
    fi

    # Profile time
    if [ $PROFTYPE -eq 2 ]; then
        # Get L1 load hit and miss
	local FLAGS="-s"
	local COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
	eval RESULT='$('$COMMANDS')'
	
	local RUNTIME=$( echo "$RESULT" | grep cu | awk --field-separator=" " '{print $2 }')
	echo $RUNTIME
    fi
}

# The important metric from nvprof are:
# Actual load
# Actual store
# Requested load
# Requested store

# Run Test 1
echo "RUNNING TEST 1 - Constant zero offset"
profile "./$EXEC $OFFSET" $DOPROFTYPE

exit
echo "RUNNING TEST 2 - Changing the offset by 2"
for OFFSET in `seq 1 2 31`
do
    echo $OFFSET
    profile "./$EXEC $OFFSET" $DOPROFTYPE
done
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