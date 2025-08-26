import numpy as np

array_1d = np.arange(1, 11)
print(f"The 5th element of the first array: {array_1d[4]}")
print(f"A slice of the array showing elements from the 3rd to the 8th position: {array_1d[2:8]}")

random_array = np.random.randint(10, 50, 6)
print(f"Elements greater than 30 from the random array: {random_array[random_array > 30]}")
print(f"Selected elements (2nd, 4th, 6th) from the random array using fancy indexing: {random_array[[1, 3, 5]]}")


