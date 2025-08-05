"""
Anagram Checker User Interface Module

This module provides the user interface for the Anagram Checker application.
It handles user input validation, menu display, and result formatting.

Functions:
    validate_input(user_input): Validates and cleans user input
    display_results(word, checker): Displays anagram results for a word
    main(): Main program loop with menu system

Dependencies:
    - anagram_checker.py: Contains the AnagramChecker class

Usage:
    Run this module directly to start the Anagram Checker application:
    $ python anagrams.py

Author: Developer
Date: August 2025
"""

from anagram_checker import AnagramChecker


def validate_input(user_input):
    """
    Validate the user's input and return cleaned input or None if invalid.

    Performs the following validations:
    1. Removes leading and trailing whitespace
    2. Checks if input is not empty
    3. Ensures input is a single word (no spaces)
    4. Verifies input contains only alphabetic characters

    Args:
        user_input (str): The raw input string from the user

    Returns:
        str or None: The cleaned input string if valid, None if invalid.
                    Error messages are printed for invalid inputs.
    """
    # Remove whitespace from start and end
    cleaned_input = user_input.strip()

    # Check if input is empty
    if not cleaned_input:
        print("Error: Please enter a word.")
        return None

    # Check if it's a single word (no spaces)
    if ' ' in cleaned_input:
        print("Error: Please enter only a single word.")
        return None

    # Check if it contains only alphabetic characters
    if not cleaned_input.isalpha():
        print("Error: Please enter only alphabetic characters (no numbers or special characters).")
        return None

    return cleaned_input


def display_results(word, checker):
    """
    Display the results for a given word in a formatted manner.

    Shows whether the word is valid and lists any anagrams found.
    The output format follows the specification:
    - Displays the word in uppercase with quotes
    - Shows validity status
    - Lists anagrams if found, or indicates none were found

    Args:
        word (str): The validated word to check for anagrams
        checker (AnagramChecker): An instance of AnagramChecker class
                                 used to validate words and find anagrams

    Returns:
        None: This function only prints output, no return value

    Example Output:
        YOUR WORD: "MEAT"
        This is a valid English word.
        Anagrams for your word: mate, tame, team.

        OR

        YOUR WORD: "XYZ"
        This is not a valid English word.
    """
    print(f'\nYOUR WORD: "{word.upper()}"')

    # Check if it's a valid English word
    if checker.is_valid_word(word):
        print("This is a valid English word.")

        # Find anagrams
        anagrams = checker.get_anagrams(word)

        if anagrams:
            anagrams_str = ", ".join(anagrams)
            print(f"Anagrams for your word: {anagrams_str}.")
        else:
            print("No anagrams found for your word.")
    else:
        print("This is not a valid English word.")


def main():
    """
    Main function to run the anagram checker program.

    Provides a menu-driven interface that allows users to:
    1. Input a word to check for anagrams
    2. Exit the program

    The function creates an AnagramChecker instance and runs in a loop
    until the user chooses to exit. Input validation is performed on
    user entries, and results are displayed in a formatted manner.

    Menu Options:
        1: Input a word - Prompts for word input and shows anagram results
        2: Exit - Terminates the program with a goodbye message

    Error Handling:
        - Invalid menu choices display an error message
        - Word validation errors are handled by validate_input()
        - AnagramChecker initialization errors are handled in that class

    Returns:
        None: This function manages the program flow and user interaction
    """
    print("Welcome to the Anagram Checker!")
    print("================================")

    # Create an instance of AnagramChecker
    checker = AnagramChecker()

    while True:
        print("\nMenu:")
        print("1. Input a word")
        print("2. Exit")

        choice = input("Please choose an option (1 or 2): ").strip()

        if choice == '1':
            user_word = input("Please enter a word: ")
            validated_word = validate_input(user_word)

            if validated_word:
                display_results(validated_word, checker)

        elif choice == '2':
            print("Thank you for using the Anagram Checker! Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
