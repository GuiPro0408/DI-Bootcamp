# 🌟 Exercise 1: Favorite Numbers
# Key Python Topics:
#
# Sets
# Adding/removing items in a set
# Set concatenation (using union)
#
#
# Instructions:
#
# Create a set called my_fav_numbers and populate it with your favorite numbers.
# Add two new numbers to the set.
# Remove the last number you added to the set.
# Create another set called friend_fav_numbers and populate it with your friend’s favorite numbers.
# Concatenate my_fav_numbers and friend_fav_numbers to create a new set called our_fav_numbers.
# Note: Sets are unordered collections, so ensure no duplicate numbers are added.

my_fav_numbers = {3, 7, 21, 42}  # Initial favorite numbers
my_fav_numbers.add(99)
my_fav_numbers.add(88)
my_fav_numbers.remove(88)

friend_fav_numbers = {5, 10, 15}  # Friend's favorite numbers
our_fav_numbers = my_fav_numbers.union(friend_fav_numbers)  # Concatenating sets
print("My favorite numbers:", my_fav_numbers)
print("Friend's favorite numbers:", friend_fav_numbers)
print("Our favorite numbers:", our_fav_numbers)

# 🌟 Exercise 2: Tuple
# Key Python Topics:
#
# Tuples (immutability)
#
#
# Instructions:
#
# Given a tuple of integers, try to add more integers to the tuple.
# Hint: Tuples are immutable, meaning they cannot be changed after creation. Think about why you can’t add more integers to a tuple.

my_tuple = (1, 2, 3, 4, 5)
try:
    my_tuple += (6, 7)  # This creates a new tuple by concatenation
    print("Updated tuple:", my_tuple)
except TypeError as e:
    print("Error:", e)
    print("Tuples are immutable, so we cannot change them directly.")

# 🌟 Exercise 3: List Manipulation
# Key Python Topics:
#
# Lists
# List methods: append, remove, insert, count, clear
#
#
# Instructions:
#
# You have a list: basket = ["Banana", "Apples", "Oranges", "Blueberries"]
# Remove "Banana" from the list.
# Remove "Blueberries" from the list.
# Add "Kiwi" to the end of the list.
# Add "Apples" to the beginning of the list.
# Count how many times "Apples" appear in the list.
# Empty the list.
# Print the final state of the list.

basket = ["Banana", "Apples", "Oranges", "Blueberries"]
basket.remove("Banana")  # Remove "Banana"
basket.remove("Blueberries")  # Remove "Blueberries"
basket.append("Kiwi")  # Add "Kiwi" to the end
basket.insert(0, "Apples")  # Add "Apples" to the beginning
apple_count = basket.count("Apples")  # Count "Apples"
basket.clear()  # Empty the list
print("Final state of the basket:", basket)

# 🌟 Exercise 4: Floats
# Key Python Topics:
#
# Lists
# Floats and integers
# Range generation
#
#
# Instructions:
#
# Recap: What is a float? What’s the difference between a float and an integer?
# Create a list containing the following sequence of mixed floats and integers:
# 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.
# Avoid hard-coding each number manually.
# Think: Can you generate this sequence using a loop or another method?

my_range = range(3, 11)
float_list = [x / 2 for x in my_range]  # Generate the sequence using a list comprehension
print("List of mixed floats and integers:", float_list)

# 🌟 Exercise 5: For Loop
# Key Python Topics:
#
# Loops (for)
# Range and indexing
#
#
# Instructions:
#
# Write a for loop to print all numbers from 1 to 20, inclusive.
# Write another for loop that prints every number from 1 to 20 where the index is even.

for i in range(1, 21):
    print(i)  # Print all numbers from 1 to 20

for i in range(1, 21):
    if i % 2 == 0:
        print(i)  # Print numbers where the index is even

# 🌟 Exercise 6: While Loop
# Key Python Topics:
#
# Loops (while)
# Conditionals
#
#
# Instructions:
#
# Write a while loop that keeps asking the user to enter their name.
# Stop the loop if the user’s input is your name.

while True:
    user_input = input("Please enter your name: ")
    if user_input.lower() == "guillaume juste":
        print("Welcome, Guillaume Juste!")
        break
    print("That's not my name, try again.")

# 🌟 Exercise 7: Favorite Fruits
# Key Python Topics:
#
# Input/output
# Strings and lists
# Conditionals
#
#
# Instructions:
#
# Ask the user to input their favorite fruits (they can input several fruits, separated by spaces).
# Store these fruits in a list.
# Ask the user to input the name of any fruit.
# If the fruit is in their list of favorite fruits, print:
# "You chose one of your favorite fruits! Enjoy!"
# If not, print:
# "You chose a new fruit. I hope you enjoy it!"

favorite_fruits = input("Enter your favorite fruits (separated by spaces): ").split()
fruit_choice = input("Enter the name of a fruit: ")
if fruit_choice in favorite_fruits:
    print("You chose one of your favorite fruits! Enjoy!")
else:
    print("You chose a new fruit. I hope you enjoy it!")

# 🌟 Exercise 8: Pizza Toppings
# Key Python Topics:
#
# Loops
# Lists
# String formatting
#
#
# Instructions:
#
# Write a loop that asks the user to enter pizza toppings one by one.
# Stop the loop when the user types 'quit'.
# For each topping entered, print:
# "Adding [topping] to your pizza."
# After exiting the loop, print all the toppings and the total cost of the pizza.
# The base price is $10, and each topping adds $2.50.

toppings = []
while True:
    topping = input("Enter a pizza topping (or type 'quit' to finish): ")
    if topping.lower() == 'quit':
        break
    toppings.append(topping)
    print(f"Adding {topping} to your pizza.")

base_price = 10
topping_price = 2.50
total_cost = base_price + (len(toppings) * topping_price)
print(f"Your pizza toppings: {', '.join(toppings)}")
print(f"Total cost of your pizza: ${total_cost:.2f}")

# 🌟 Exercise 9: Cinemax Tickets
# Key Python Topics:
#
# Conditionals
# Lists
# Loops
#
#
# Instructions:
#
# Ask for the age of each person in a family who wants to buy a movie ticket.
# Calculate the total cost based on the following rules:
# Free for people under 3.
# $10 for people aged 3 to 12.
# $15 for anyone over 12.
# Print the total ticket cost.
#
#
# Bonus:
#
# Imagine a group of teenagers wants to see a restricted movie (only for ages 16–21).
# Write a program to:
# Ask for each person’s age.
# Remove anyone who isn’t allowed to watch.
# Print the final list of attendees.

total_cost = 0
while True:
    age_input = input("Enter the age of the person (or type 'done' to finish): ")
    if age_input.lower() == 'done':
        break
    try:
        age = int(age_input)
        if age < 3:
            print("Free ticket for under 3 years old.")
        elif 3 <= age <= 12:
            total_cost += 10
            print("Ticket cost: $10")
        else:
            total_cost += 15
            print("Ticket cost: $15")
    except ValueError:
        print("Please enter a valid age or 'done' to finish.")

print(f"Total ticket cost: ${total_cost}")

# Bonus: Restricted movie attendees
restricted_attendees = []
while True:
    age_input = input("Enter the age of the person (or type 'done' to finish): ")
    if age_input.lower() == 'done':
        break
    try:
        age = int(age_input)
        if 16 <= age <= 21:
            restricted_attendees.append(age)
            print(f"Age {age} is allowed to watch the restricted movie.")
        else:
            print(f"Age {age} is not allowed to watch the restricted movie.")
    except ValueError:
        print("Please enter a valid age or 'done' to finish.")

print("Final list of attendees for the restricted movie:", restricted_attendees)

# 🌟 Exercise 10: Sandwich Orders
# Key Python Topics:
#
# Lists
# Loops (while)
#
#
# Instructions:
#
# Using the list:
# sandwich_orders = ["Tuna", "Pastrami", "Avocado", "Pastrami", "Egg", "Chicken", "Pastrami"]
# The deli has run out of “Pastrami”, so use a loop to remove all instances of “Pastrami” from the list.
# Prepare each sandwich, one by one, and move them to a list called finished_sandwiches.
# Print a message for each sandwich made, such as: "I made your Tuna sandwich."
# Print the final list of all finished sandwiches.

sandwich_orders = ["Tuna", "Pastrami", "Avocado", "Pastrami", "Egg", "Chicken", "Pastrami"]
finished_sandwiches = []
while "Pastrami" in sandwich_orders:
    sandwich_orders.remove("Pastrami")  # Remove all instances of "Pastrami"

for sandwich in sandwich_orders:
    finished_sandwiches.append(sandwich)  # Move to finished sandwiches
    print(f"I made your {sandwich} sandwich.")

print("All finished sandwiches:", finished_sandwiches)
