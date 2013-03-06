#!/bin/bash

NVPROF=nvprof  # Name of profiling cmmand
PROFFLAGS="--profile-from-start-off --devices 0"
EXEC=main      # Name of executable
DOPROFTYPE=1   # 1 is event counters
               # 2 is kernel run time
INPUT=$#       # Number of input
DOGP=0         # Produce gnuplot figure?

# Check that 
if [ ! -f $EXEC ]; then
    echo "FILE $EXEC DOES NOT EXIST"
    echo "PLEASE RUN MAKE"
    exit 
fi

CFLAGS=" "
TESTNO=0       # Default: Run all tests
OFFSET=0

# Load Test number if given
if [ $INPUT -gt 0 ]; then
    TESTNO=$1
fi

if [ $TESTNO -eq 1 ] && [ $INPUT -gt 1 ]; then
    OFFSET=$2
fi

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
	echo $COMMANDS
	eval RESULT='$('$COMMANDS')'
	
	local RUNTIME=$( echo "$RESULT" | grep cu | awk --field-separator=" " '{print $2 }')
	echo $RUNTIME
	echo "$OFFSET $RUNTIME" >> OffsetRuntime.data
    fi
}

# The important metric from nvprof are:
# Actual load
# Actual store
# Requested load
# Requested store

# Run Test 1
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 1 ]; then
    echo "RUNNING TEST 1 - Constant offset of $OFFSET"
    CFLAGS="--Offset $OFFSET"
    profile "./$EXEC $CFLAGS" 1
fi

# Run Test 2
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 2 ]; then
    echo "RUNNING TEST 2 - Changing the offset by 2"
    echo " " > OffsetRuntime.data
    for OFFSET in `seq 0 2 32`
    do
	echo "TESTING OFFSET $OFFSET"
	CFLAGS="--Gridx 1000 --Offset $OFFSET"
	profile "./$EXEC $CFLAGS" 2 #$DOPROFTYPE
    done
fi

# Run Test 3
# Only some threads will copy values
# Memory should still be coacelsed
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 3 ]; then
    echo "TESTING WHEN NOT ALL THREADS COPY"
    CFLAGS="--Min 36 --Max 45"
    profile "./$EXEC $CFLAGS" 1
fi