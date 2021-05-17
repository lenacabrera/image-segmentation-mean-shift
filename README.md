# Image Segmentation Using Mean-Shift Algorithm

This project contains an implementation of the mean-shift algorithm and two speedups to improve the computation 
efficiency. The implementation is structured into 4 different python scripts.

main.py contains
- main function to execute the entire code
- function performing image segmentation "image_segmentation"

mean_shift.py contains
- peak searching function "find_peak"
- plain implementation of mean-shift procedure in function "ms_no_speedup" (calls "find_peak")
- first speedup of mean-shift algorithm in function "ms_speedup1" (calls "find_peak")
- second speedup of mean-shift algorithm in function "ms_speedup2" (calls "find_peak_opt")

utils.py contains
- functions to load data and images
- functions for preprocessing of images

plotcluster.py contains
- function to plot found clusters in 3D space
