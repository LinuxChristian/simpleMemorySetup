# This is a automatic generated file for
# plotting residual of SSIM runs.
set term png
set title "Kernel runtime as a function of offset" 
set xlabel "Offset"
set ylabel "Runtime (micro seconds)" 
plot "OffsetRuntime.data" using 1:2 title "Runtime" with lines
