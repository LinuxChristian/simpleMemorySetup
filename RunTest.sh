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
    echo "NOW RUNNING MAKE"
    eval "make > /dev/null 2>&1"
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

	if [ $# -eq 3 ]; then
	    #printf "%i \t %i \t %i \t %i \t %i\n" $L1_AVG_HIT $L1_AVG_MISS $GLD_AVG $GST_AVG $STORE_TRANF_AVG >> $3
	    printf "%f \t %f \n" $LOAD_RATIO $STORE_RATIO >> $3
	fi
    fi

    # Profile time
    if [ $PROFTYPE -eq 2 ]; then
        # Get L1 load hit and miss
	local FLAGS="-s"
	local COMMANDS="$NVCOMMANDS $FLAGS $SHELLCOMMAND"
	eval RESULT='$('$COMMANDS')'
	
	local RUNTIME=$( echo "$RESULT" | grep cu | awk --field-separator=" " '{print $2 }')
	echo $RUNTIME

	if [ $# -eq 3 ] && [ $TESTNO -ne 7 ] && [ $TESTNO -ne 8 ]; then 
	    echo "$OFFSET $RUNTIME" >> $3
	else
	    printf "%s \t " $RUNTIME >> $3
	fi
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
    OUTPUTFILE="OffsetRuntime.data"
    echo " " > $OUTPUTFILE
    for OFFSET in `seq 0 2 32`
    do
	echo "TESTING OFFSET $OFFSET"
	CFLAGS="--Gridx 1000 --Blockx 1024 --Offset $OFFSET"
	profile "./$EXEC $CFLAGS" 2 $OUTPUTFILE
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


# Run Test 4
# Does a simple finite difference using global memory
# and no tricks
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 4 ]; then
    echo "TESTING A SIMPLE FINITE DIFFERENCE STENCIL USING GLOBAL MEMORY"
    CFLAGS="--Gridx 128 --Gridy 128 --Blockx 32 --Blocky 32 -t 2 --xdim 10000 --ydim 10000"
    profile "./$EXEC $CFLAGS" 1
fi

# Run Test 5
# Does a simple finite difference using shared memory
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 5 ]; then
    echo "TESTING A SIMPLE FINITE DIFFERENCE STENCIL USING SHARED MEMORY"

    # Force padding to be zero and recompile
    PADDINGLINENO=$(awk '/#define PADDING/{print FNR}' main.cu) # Get the line number of the padding definition
    LINECONTENT=$(awk '/#define PADDING/{print $0}' main.cu) # Get the line number of the padding definition
    eval "perl -pi -e 's/$LINECONTENT/#define PADDING 0 / if $. == $PADDINGLINENO' main.cu"
    eval "make clean > /dev/null 2>&1 ; make > /dev/null 2>&1"
    
    if [ ! -f $EXEC ]; then
	echo "FAILED TO COMPILE $EXEC"
	exit
    fi

    
    CFLAGS="--Gridx 128 --Gridy 128 --Blockx 32 --Blocky 32 -t 3 --xdim 10000 --ydim 10000"
    profile "./$EXEC $CFLAGS" 1
fi

# Run Test 6
# Runs test 1 for a lot of different grid sizes. What the test should show it that the GPU is only
# able to achive a ratio of 2 when the problem size get large.
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 6 ]; then
    echo "RUNNING TEST 6 - TEST 1 WITH CHANGING DIMENSIONS"
    OUTPUTFILE="MemCpyEffency.data"
    #printf "GDIM \t L1_HIT \t L1_MISS \t GLD_REQ \t GST_REQ \t GLOBAL_ST \n" > $OUTPUTFILE
    printf "THREADS \t LD_RATIO \t ST_RATIO \n" > $OUTPUTFILE
    for GSIZE in `seq 1 1 30`
    do
	echo "TESTING 1D GRID DIMENSIONS $GSIZE"
	CFLAGS="--Gridx $GSIZE"
	THREADS=$(echo "scale=2; $GSIZE*128" | bc) # Note: 128 is the default threads pr. block
	printf "%i \t " $THREADS >> $OUTPUTFILE
	profile "./$EXEC $CFLAGS" 1 $OUTPUTFILE
    done
fi

# Run Test 7
# Runs test 5 for different grid sizes with and without padding
# This should show the effect of the ucoalesced memory access by the halo nodes
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 7 ]; then
    echo "RUNNING TEST 7 - TEST 5 WITH CHANGING DIMENSIONS AND PADDING"

    # Init flags
    PROFFLAGS="$PROFFLAGS -u s" # Set time unit to seconds
    GSIZE=128
    CFLAGS="--Gridx $GSIZE --Gridy $GSIZE --xdim 10000 --ydim 10000 -t 3"

    for PADDINGSIZE in `seq 0 1 1`
    do
        # Replace padding size and recompile
	PADDINGLINENO=$(awk '/#define PADDING/{print FNR}' main.cu) # Get the line number of the padding definition
	LINECONTENT=$(awk '/#define PADDING/{print $0}' main.cu) # Get the line number of the padding definition
	eval "perl -pi -e 's/$LINECONTENT/#define PADDING $PADDINGSIZE / if $. == $PADDINGLINENO' main.cu"
	eval "make clean > /dev/null 2>&1 ; make > /dev/null 2>&1"

	if [ ! -f $EXEC ]; then
	    echo "FAILED TO COMPILE $EXEC"
	    exit
	fi

	OUTPUTFILE="PaddingEffency.data"
	OUTPUTFILE="$PADDINGSIZE$OUTPUTFILE"
	printf "MEMSIZE \t TIME  \n" > $OUTPUTFILE
	for GSIZE in `seq 2 8 256` 
	do
	    echo "TESTING GRID DIMENSIONS $GSIZE x $GSIZE"
	    GDIM=$(echo "$GSIZE*32+100" | bc) # Note: Make grid a bit bigger
	    CFLAGS="--Gridx $GSIZE --Gridy $GSIZE --xdim $GDIM --ydim $GDIM -t 3"
	    echo $CFLAGS
	    THREADS=$(echo "scale=2; $GSIZE*$GSIZE*32*32*8" | bc) # Note: 32x32 is the default threads pr. block * 8 bytes pr thread
	    printf "%i \t " $THREADS >> $OUTPUTFILE
	    profile "./$EXEC $CFLAGS" 2 $OUTPUTFILE

	    # Read line and compute memory throughput
	    TIME=$(cat $OUTPUTFILE | tail -n 1 | awk --field-separator=" " '{print $2 }' | perl -pi -e 's/e/*10^/') # Unit is important and bc does not know (e) exponant
	    #THROUGHPUT=$(echo "scale=15; ($THREADS/(10^9))/($TIME)" | bc) # See CUDA Best Practice (2.2.2)
	    echo "Runtime was $TIME" # and memory throughput was ${THROUGHPUT:0:5}"
	    #printf "%s " $THROUGHPUT >> $OUTPUTFILE
	    printf "\n" >> $OUTPUTFILE
	done
	echo $CFLAGS
	# Just to show ratio
	profile "./$EXEC $CFLAGS" 1
    done
fi

# Run Test 8
# A rerun of test 4 (Global memory access) but this time the runtime is 
# extracted so it can be comparied with the shared runtime
if [ $TESTNO -eq 0 ] || [ $TESTNO -eq 8 ]; then
    echo "TESTING A SIMPLE FINITE DIFFERENCE STENCIL USING GLOBAL MEMORY AND DIFFERENT GRID SIZES"

    # Init flags
    PROFFLAGS="$PROFFLAGS -u s" # Set time unit to seconds
    OUTPUTFILE="GlobalEffency.data"
    printf "MEMSIZE \t TIME  \n" > $OUTPUTFILE

    for GSIZE in `seq 2 8 256` 
    do
	echo "TESTING GRID DIMENSIONS $GSIZE x $GSIZE"
	GDIM=$(echo "$GSIZE*32+100" | bc) # Note: Make grid a bit bigger
	CFLAGS="--Gridx $GSIZE --Gridy $GSIZE --Blockx 32 --Blocky 32 --xdim $GDIM --ydim $GDIM -t 2"
	echo $CFLAGS
	THREADS=$(echo "scale=2; $GSIZE*$GSIZE*32*32*8" | bc) # Note: 32x32 is the default threads pr. block * 8 bytes pr thread
	printf "%i \t " $THREADS >> $OUTPUTFILE
	profile "./$EXEC $CFLAGS" 2 $OUTPUTFILE
	
	TIME=$(cat $OUTPUTFILE | tail -n 1 | awk --field-separator=" " '{print $2 }' | perl -pi -e 's/e/*10^/') # Unit is important and bc does not know (e) exponant
	#THROUGHPUT=$(echo "scale=15; ($THREADS/(10^9))/($TIME)" | bc) # See CUDA Best Practice (2.2.2)
	echo "Runtime was $TIME" # and memory throughput was ${THROUGHPUT:0:5}"
	#printf "%s " $THROUGHPUT >> $OUTPUTFILE
	printf "\n" >> $OUTPUTFILE
    done
fi