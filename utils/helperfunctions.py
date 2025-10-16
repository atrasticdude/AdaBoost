import numpy as np

def amount_of_say_calc(total_error):
    epsilon = 1e-10
    return 0.5 * np.log2((1 - total_error + epsilon) / (total_error + epsilon))


def prob_array(arr):
    for i in range(1,len(arr)):
       arr[i] = arr[i] + arr[i - 1]
    return np.array(arr)

def find_medians(a):
    arr = np.unique(np.sort(a))
    medians = []
    for i in range(len(arr) - 1):
        mid = (arr[i] + arr[i + 1]) / 2
        medians.append(mid)
    return medians