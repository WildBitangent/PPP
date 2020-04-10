#!/bin/bash

ml intel/2020a HDF5/1.10.6-intel-2020a

# domain sizes
declare -a sizes=(256 512 1024 2048 4096)

# generate input files
for size in ${sizes[*]} 
do
  echo "Generating input data ${size}x${size}..."
  ../build/data_generator -o input_data_${size}.h5 -N ${size} -H 100 -C 20
done
