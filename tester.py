import subprocess
import sys
from tqdm import tqdm  # For progress bar

def run_game():
    # Run the play script with specific agents
    result = subprocess.run([
        './play',
        '--player1', 'student_agent',
        '--player2', 'random_agent',
        '--display', 'false',
        '--verbose', 'false'  # Set verbose to false to avoid extra output
    ], capture_output=True, text=True)
    
    # Combine stdout and stderr
    output = result.stdout + result.stderr

    # Uncomment the following line to print the entire output for debugging
    # print(output)
    
    # Check if there was an error
    if result.returncode != 0:
        print("Error running the game:")
        print(output)
        return None  # Treat as a draw or error
    
    # Parse the output to determine winner
    if "Player 1 (Blue) wins!" in output:
        return True  # student_agent won
    elif "Player 2 (Brown) wins!" in output:
        return False  # student_agent lost
    elif "Game ends in a draw" in output:
        return None  # Draw
    else:
        print("Unexpected output:")
        print(output)
        return None  # Treat as a draw or error

def main():
    num_games = 50
    wins = 0
    draws = 0
    
    print("Running games: student_agent vs test_agent")
    print("-" * 60)
    
    # Run games with progress bar
    for game_num in tqdm(range(1, num_games + 1)):
        result = run_game()
        if result is True:
            wins += 1
            print(f"Game {game_num}: Win")
        elif result is False:
            print(f"Game {game_num}: Loss")
        elif result is None:
            draws += 1
            print(f"Game {game_num}: Draw")
        
    print("-" * 60)
    print(f"Number of wins: {wins}")
    print(f"Number of draws: {draws}")
    print(f"Number of losses: {num_games - wins - draws}")
    print(f"Number of games: {num_games}")
    print(f"Total win rate: {(wins/num_games)*100:.2f}%")
    print(f"Draw rate: {(draws/num_games)*100:.2f}%")
    print(f"Loss rate: {(num_games - wins - draws)/num_games*100:.2f}%")
    print(f"Total non-defeat rate: {(wins + draws)/num_games*100:.2f}%")

if __name__ == "__main__":
    main()
