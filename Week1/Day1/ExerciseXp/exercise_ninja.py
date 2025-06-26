# Exercise 3 : Outputs
# Instructions
# Predict the output of the following code snippets:
#
#     >>> 3 <= 3 < 9
#     >>> 3 == 3 == 3
#     >>> bool(0)
#     >>> bool(5 == "5")
#     >>> bool(4 == 4) == bool("4" == "4")
#     >>> bool(bool(None))
#     x = (1 == True)
#     y = (1 == False)
#     a = True + 4
#     b = False + 10
#
#     print("x is", x)
#     print("y is", y)
#     print("a:", a)
#     print("b:", b)

# Predictions:
# 3 <= 3 < 9: True (3 is equal to 3 and less than 9)
# 3 == 3 == 3: True (all three values are equal)
# bool(0): False (0 is considered False in Python)
# bool(5 == "5"): False (5 is an int and "5" is a str, so they are not equal)
# bool(4 == 4) == bool("4" == "4"): True (both comparisons are True)
# bool(bool(None)): False (None is considered False, so bool(None) is False)

x = (1 == True)  # True, because 1 is considered True
y = (1 == False)  # False, because 1 is not equal to False
a = True + 4  # 1 + 4 = 5, because True is treated as 1
b = False + 10  # 0 + 10 = 10, because False is treated as 0

print("x is", x)  # x is True
print("y is", y)  # y is False
print("a:", a)  # a: 5
print("b:", b)  # b: 10

# Exercise 4 : How many characters in a sentence ?
# Instructions
# Use python to find out how many characters are in the following text, use a single line of code (beyond the establishment of your my_text variable).

my_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.  sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
character_count = len(my_text)
print(f"The number of characters in the text is: {character_count}")

# Exercise 5: Longest word without a specific character
# Instructions
# Keep asking the user to input the longest sentence they can without the character “A”.
# Each time a user successfully sets a new longest sentence, print a congratulations message.

longest_sentence = ""

while True:
    user_input = input("Please enter a sentence without the character 'A': ")

    # Fallback to quit the loop if the user inputs 'exit'
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    # Check if the input contains the character 'A' or 'a'
    if 'A' in user_input or 'a' in user_input:
        print("Your sentence contains the character 'A'. Please try again.")
        continue

    if len(user_input) > len(longest_sentence):
        longest_sentence = user_input
        print(f"Congratulations! Your new longest sentence is: '{longest_sentence}'")
    else:
        print("Your sentence is not longer than the current longest sentence.")
