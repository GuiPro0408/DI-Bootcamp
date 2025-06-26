# Instructions:
# 1. Ask for User Input:
#
# The string must be exactly 10 characters long.
# 2. Check the Length of the String:
#
# If the string is less than 10 characters, print: "String not long enough."
# If the string is more than 10 characters, print: "String too long."
# If the string is exactly 10 characters, print: "Perfect string" and proceed to the next steps.
# 3. Print the First and Last Characters:
#
# Once the string is validated, print the first and last characters.
# 4. Build the String Character by Character:
#
# Using a for loop, construct and print the string character by character. Start with the first character, then the first two characters, and so on, until the entire string is printed.
# Hint: You can create a loop that goes through the string, adding one character at a time, and print it progressively.
#
# 5. Bonus: Jumble the String (Optional)
#
# As a bonus, try shuffling the characters in the string and print the newly jumbled string.
# Hint: You can use the random.shuffle function to shuffle a list of characters.

user_input = input("Please enter a string (exactly 10 characters long): ")

if len(user_input) < 10:
    print("String not long enough.")
elif len(user_input) > 10:
    print("String too long.")
else:
    print("Perfect string")
    print(f"First character: {user_input[0]}")
    print(f"Last character: {user_input[-1]}")

    # Print the string character by character
    # It start by printing from index 0 to index i (which is the current iteration)
    for i in range(1, len(user_input) + 1):
        print(user_input[:i])

    # Bonus: Jumble the string
    import random
    char_list = list(user_input) # Convert string to list of characters
    random.shuffle(char_list) # Shuffle the list of characters
    jumbled_string = ''.join(char_list) # Convert the list back to a string

    print(f"Jumbled string: {jumbled_string}")