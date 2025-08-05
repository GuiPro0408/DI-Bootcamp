import random


class Game:
    """A class to represent a single game of Rock Paper Scissors."""

    # Class variables - shared across all instances
    CHOICES = ['rock', 'paper', 'scissors']
    WINNING_COMBINATIONS = {
        'rock': 'scissors',  # rock beats scissors
        'paper': 'rock',  # paper beats rock
        'scissors': 'paper'  # scissors beats paper
    }

    @classmethod
    def get_user_item(cls):
        """Ask the user to select an item (rock/paper/scissors) with validation."""
        while True:
            user_choice = input("Select (rock/paper/scissors): ").lower().strip()
            if user_choice in cls.CHOICES:
                return user_choice
            else:
                print("Invalid choice. Please select rock, paper, or scissors.")

    @classmethod
    def get_computer_item(cls):
        """Select rock/paper/scissors at random for the computer."""
        return random.choice(cls.CHOICES)

    @classmethod
    def get_game_result(cls, user_item, computer_item):
        """Determine the result of the game.

        Args:
            user_item (str): The user's chosen item
            computer_item (str): The computer's chosen item

        Returns:
            str: 'win', 'draw', or 'loss'
        """
        if user_item == computer_item:
            return 'draw'

        if cls.WINNING_COMBINATIONS[user_item] == computer_item:
            return 'win'
        else:
            return 'loss'

    def play(self):
        """Play a single game of rock-paper-scissors.

        Returns:
            str: The result of the game ('win', 'draw', or 'loss')
        """
        # Get user's choice
        user_item = self.get_user_item()

        # Get computer's choice
        computer_item = self.get_computer_item()

        # Determine the result
        result = self.get_game_result(user_item, computer_item)

        # Print the game outcome
        print(f"\nYou selected {user_item}. The computer selected {computer_item}.", end=" ")

        if result == 'win':
            print("You win!")
        elif result == 'loss':
            print("You lose!")
        else:
            print("You drew!")

        return result
