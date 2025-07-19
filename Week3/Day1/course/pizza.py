def make_pizza(size, *toppings):
    """
    Prepares a pizza of the specified size with the provided toppings.

    This function prints the size of the pizza and the list of toppings
    that will be added to the pizza.

    :param size: The size of the pizza in inches.
    :type size: int
    :param toppings: Variable number of toppings to be added to the pizza.
    :type toppings: str
    :return: None
    """
    print(f"\n Making a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)
