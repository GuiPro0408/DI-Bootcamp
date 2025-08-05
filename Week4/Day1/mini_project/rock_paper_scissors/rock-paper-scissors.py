from game import Game


def get_user_menu_choice():
    """Display the menu and get the user's choice with validation.

    Returns:
        str: The user's menu choice
    """
    print("\nMenu:")
    print("(g) Play a new game")
    print("(x) Show scores and exit")
    print("(q) Quit")

    choice = input("What would you like to do? ").lower().strip()
    return choice


def print_results(results):
    """Print the results of all games played.

    Args:
        results (dict): Dictionary containing win, loss, and draw counts
    """
    print("\nGame Results")
    print("*" * 30)
    print(f"You won {results['win']} times")
    print(f"You lost {results['loss']} times")
    print(f"You drew {results['draw']} times")
    print("\nThank you for playing!")


def main():
    """Main function to run the Rock Paper Scissors game."""
    print("Welcome to Rock Paper Scissors!")

    # Initialize results dictionary
    results = {
        'win': 0,
        'loss': 0,
        'draw': 0
    }

    while True:
        choice = get_user_menu_choice()

        if choice == 'g':
            # Play a new game
            game = Game()
            result = game.play()

            # Update results
            results[result] += 1

        elif choice == 'x':
            # Show scores and exit
            print_results(results)
            break

        elif choice == 'q':
            # Quit without showing scores
            print("Thanks for playing!")
            break

        else:
            print("Invalid choice. Please select 'g', 'x', or 'q'.")


if __name__ == "__main__":
    main()
