
import sys
import time


def f(n):
    # Main processing
    count = 0.0
    for i in range(n):
        for j in range(i):
            for _ in range(j):
                count += 1.0
    # print(count)


    return count

if __name__ == "__main__":
    # Number of loop iterations from command line argument
    n = 2_000
    # print("Number of iterations: ", n)
    n = int(n)

    # Execute the function
    # Start timing
    time_start = time.time()

    # main
    count = f(n)

    # Elapsed time (seconds)
    elapsed_time = time.time() - time_start
    print("Time:", str(elapsed_time))

    print(count)
