def add_operator(x, y):
    """
    Adds two numbers together.

    Parameters:
    x (int, float): The first number.
    y (int, float): The second number.

    Returns:
    int, float: The sum of x and y.
    """
    return x + y


def divide_operator(x, y):
    """
    Divides the first number by the second.

    Parameters:
    x (int, float): The numerator.
    y (int, float): The denominator.

    Returns:
    float: The result of the division.

    Raises:
    ZeroDivisionError: If y is zero.
    """
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return x / y
