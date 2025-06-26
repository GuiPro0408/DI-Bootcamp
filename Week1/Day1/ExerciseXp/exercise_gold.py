# Exercise 1 : Hello World-I love Python
# Instructions
# Print the following output in one line of code:
#
# Hello world
# Hello world
# Hello world
# Hello world
# I love python
# I love python
# I love python
# I love python

print("Hello world\nHello world\nHello world\nHello world\nI love python\nI love python\nI love python\nI love python")

# Exercise 2 : What is the Season ?
# Instructions
# Ask the user to input a month (1 to 12).
# Display the season of the month received :
# Spring runs from March (3) to May (5)
# Summer runs from June (6) to August (8)
# Autumn runs from September (9) to November (11)
# Winter runs from December (12) to February (2)

month = int(input("Enter a month (1 to 12): "))

if month in [3, 4, 5]:
    print("Spring")
elif month in [6, 7, 8]:
    print("Summer")
elif month in [9, 10, 11]:
    print("Autumn")
elif month in [12, 1, 2]:
    print("Winter")
else:
    print("Invalid month. Please enter a number between 1 and 12.")
