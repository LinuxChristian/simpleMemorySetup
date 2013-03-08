# This is a automatic generated file for
# plotting residual of SSIM runs.
set term png
set title "Kernel runtime as a function of total domain size" 
set xlabel "Gridsize (NxN)"
set ylabel "Runtime (s)" 
set log x
set log y
plot "0PaddingEffency.data" using 1:2 title "Zero padding" with lines, "1PaddingEffency.data" using 1:2 title "One padding" with lines, "GlobalEffency.data" using 1:2 title "Global copy" with lines
