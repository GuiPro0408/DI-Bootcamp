my_list = [2, 3, 1, 2, "four", 42, 1, 5, 3, "imanumber"]


# Sum of array
def sum_of_array(arr):
    total = 0  # Declaration of total variable

    for item in arr:
        try:
            total += item
        except TypeError as e:
            print(f"Skipping item '{item}': {e}")

    return total


# Usage:
result = sum_of_array(my_list)
print(f"The sum of the array is: {result}")

from pizza import make_pizza

# Example usage of the pizza module
make_pizza(12, "pepperoni", "mushrooms", "green peppers")