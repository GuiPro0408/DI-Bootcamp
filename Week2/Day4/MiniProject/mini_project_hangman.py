# Pseudocode:
# 1. Select a random word from a predefined list.
# 2. Mask all alphabetic characters in the word with '*', keeping non-letters unchanged.
# 3. Initialize an empty set for guessed letters, a fixed number of attempts, and a list of body parts for hangman.
# 4. While the word is not fully guessed and attempts remain:
#     a. Display current masked word, guessed letters, remaining attempts, and hangman progress.
#     b. Prompt user for a letter guess.
#     c. Validate input: must be a single alphabetic character not already guessed.
#     d. If guess is correct, reveal all instances of the letter in the word.
#     e. If guess is wrong, reduce attempts and add a hangman body part.
# 5. At game end, display result: win if all letters guessed, lose if attempts run out.

import random  # Import the random module to select a random word from the list

# Declarations
wordslist = [
    'correction', 'childish', 'beach', 'python', 'assertive',
    'interference', 'complete', 'share', 'credit card', 'rush', 'south'
]
word = random.choice(wordslist).lower()  # Randomly select a word from the list and convert it to lowercase
display = ['*' if c.isalpha() else c for c in word]  # Create a display list with '*' for each letter in the word, keeping non-letter characters as they are
guessed = set()
attempts = 6
parts = ['head', 'body', 'left arm', 'right arm', 'left leg', 'right leg']


# Function to print the current status of the game
def print_status():
    # Create a divider line for better visual separation
    divider = '=' * 40

    print(divider)
    print(f"🎮 HANGMAN GAME STATUS 🎮")
    print(divider)

    # Display the word with colorful formatting
    print(f"📝 Word: {' '.join(display)}")

    # Show guessed letters or a message if none
    if guessed:
        print(f"🔤 Letters guessed: {', '.join(sorted(guessed))}")
    else:
        print(f"🔤 Letters guessed: None yet")

    # Show remaining attempts with visual indicator
    health = '❤️' * attempts + '🖤' * (6 - attempts)
    print(f"💪 Attempts remaining: {attempts}/6 {health}")

    # Show hangman progress
    if 6 - attempts > 0:
        print(f"☠️ Hangman parts: {', '.join(parts[:6 - attempts])}")

    print(divider)


# Function to print the game result in a prettier way
def print_result(is_winner):
    divider = '*' * 50

    print("\n" + divider)
    if is_winner:
        print("""
        🎉🎉🎉 CONGRATULATIONS! 🎉🎉🎉
        🏆 You've guessed the word correctly! 🏆
        """)
        print(f"        The word was: ✨ {word.upper()} ✨")
    else:
        print("""
        💀 GAME OVER 💀
        Better luck next time!
        """)
        print(f"        The word was: ✨ {word.upper()} ✨")
        print(f"        You had {attempts} attempts left.")

    # Statistics section
    total_letters = len(
        set(letter for letter in word if letter.isalpha())
    )
    correct_guesses = sum(1 for letter in guessed if letter in word)
    accuracy = correct_guesses / len(guessed) * 100 if guessed else 0

    print("\n        📊 GAME STATISTICS 📊")
    print(f"        Total unique letters in word: {total_letters}")
    print(f"        Letters you guessed: {len(guessed)}")
    print(f"        Correct guesses: {correct_guesses}")
    print(f"        Accuracy: {accuracy:.1f}%")
    print(divider)


# Function to check if the guess matches any letters in the word and update the display
def check_match(guess, word, display):
    found = False
    for i, c in enumerate(word):
        if c == guess:
            display[i] = guess
            found = True
    return found


# Main game loop
while attempts > 0 and '*' in display:
    print_status()
    guess = input('Guess a letter: ').lower()

    # Validate the input
    if not (guess.isalpha() and len(guess) == 1):
        print('Invalid input. Enter a single letter.')
        continue

    # Check if the letter has already been guessed
    if guess in guessed:
        print('Already guessed.')
        continue

    # Add the guess to the set of guessed letters
    guessed.add(guess)

    # Check if the guess is in the word
    if check_match(guess, word, display):
        print('Good guess!')
    else:
        print(f'Wrong! Adding {parts[6 - attempts]}')
        attempts -= 1

# End of the game
print_status()
is_winner = '*' not in display
print_result(is_winner)
