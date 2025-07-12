# Pseudocode:
# 1. Initialize a 3x3 empty board.
# 2. Alternate turns between Player X and Player O:
#    a. Display the current board.
#    b. Ask the current player to select a cell (row, column).
#       - Validate input (row/col in range, cell is empty).
#    c. Place the player's mark (X/O) in the chosen cell.
#    d. Check for a win:
#       - If the current player has three in a row (row, column, or diagonal), declare win and end.
#    e. Check for a tie:
#       - If the board is full and no winner, declare a tie and end.
#    f. Switch to the other player and continue.
# 3. End when win or tie occurs.

# Mini Project: Tic Tac Toe Game

# Display a Tic Tac Toe board  (receives a 2D list as parameter e.g. [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
def display_board(board):
    divider = '*' * 25

    print("\nTIC TAC TOE\n" + divider)
    for row in board:
        print("* " + " | ".join(row) + " *")
    print(divider)


# Function to handle player input for row and column selection (return e.g. (0, 1) for row 1, column 2)
def player_input(board):
    while True:
        try:
            # Prompt the player for row and column input
            row = int(input(f"Enter row: ")) - 1
            col = int(input(f"Enter column: ")) - 1

            # Validate the input
            if (
                    0 <= row < 3 and
                    0 <= col < 3
            ):
                if board[row][col] == ' ':
                    return row, col
                else:
                    print("Cell is occupied. Try again.")
            else:
                print("Row/Col must be 1, 2, or 3.")
        except ValueError:
            print("Enter numbers only.")


# Function to check if a player has won
def check_win(board, player):
    """
    Checks if a player has won the game by achieving a winning combination in
    a Tic Tac Toe board. The board is checked for all possible row, column,
    and diagonal win conditions.

    :param board: A 2D list representing the current state of the Tic Tac Toe board.
    :type board: list[list]
    :param player: The identifier for the player to check for a win condition. It
                   represents the player's token (e.g., 'X', 'O').
    :type player: str
    :return: A boolean indicating whether the specified player has a winning
             combination on the board.
    :rtype: bool
    """
    win_states = [
        # Rows
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Cols
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diags
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]  # Define all possible winning states

    # Check if the player has any winning combination
    return [player] * 3 in win_states


# Function to check if the board is full (no empty spaces)
def board_full(board):
    return all(cell != ' ' for row in board for cell in row)


# Main function to play the Tic Tac Toe game
def play():
    board = [
        [' '] * 3 for _ in range(3)
    ]  # Initialize a 3x3 board with empty spaces (e.g. [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])

    current = 'X'  # Start with Player X

    # Main game loop
    while True:
        display_board(board)  # Display the current state of the board

        print(f"Player {current}'s turn...")
        row, col = player_input(board)  # Get valid input from the player
        board[row][col] = current  # Place the player's mark on the board

        # Check for a win or tie after the player's move
        if check_win(board, current):
            display_board(board)
            print(f"Player {current} wins!")
            break

        # Check if the board is full (tie condition)
        if board_full(board):
            display_board(board)
            print("It's a tie!")
            break

        # Switch to the other player
        current = 'O' if current == 'X' else 'X'


# Entry point for the game
if __name__ == "__main__":
    play()
