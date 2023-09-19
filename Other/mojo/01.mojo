from time import now
# import torch
# import cv2
# import numpy as npfrom benchmark import Benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota
from python import Python
from runtime.llcl import num_cores, Runtime
from algorithm import parallelize, vectorize
from tensor import Tensor

def f(n):
    var count: Float64 = 0.0

    # Main processing
    for i in range(n):
        for j in range(i):
            for k in range(j):
                count += 1.0

    return count

# execute main
def main():

    # Execute the function
    # Start timing
    let time_start = now()

    # main
    let n: Int64 = 2_000  # Hardcode the number of loops
    count = f(n)

    # Elapsed time (seconds)
    let eval_end = now()
    let execution_time_sequential = Float64(eval_end - time_start)
    print("Time:", execution_time_sequential / 1000_000_000)
    # count
    print(count)
