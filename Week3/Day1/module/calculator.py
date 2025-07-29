from operators import add_operator, divide_operator


# Usage example
def calculate_sum(x, y):
    """
    Calculates the sum of two numbers using the add_operator function.

    Parameters:
    x (int, float): The first number.
    y (int, float): The second number.

    Returns:
    int, float: The sum of x and y.
    """
    return add_operator(x, y)


def calculate_division(x, y):
    """
    Calculates the division of two numbers using the divide_operator function.

    Parameters:
    x (int, float): The numerator.
    y (int, float): The denominator.

    Returns:
    float: The result of the division.

    Raises:
    ZeroDivisionError: If y is zero.
    """
    return divide_operator(x, y)


print(calculate_sum(1, 2))
print(calculate_division(10, 2))

# Example usage of the calculator module
if __name__ == "__main__":
    x = 10
    y = 5
    z = 0

    print(f"Sum of {x} and {y} is: {calculate_sum(x, y)}")

    try:
        print(f"Division of {x} by {y} is: {calculate_division(x, y)}")
        print(f"Division of {x} by {z} is: {calculate_division(x, z)}")
    except ZeroDivisionError as e:
        print(e)
