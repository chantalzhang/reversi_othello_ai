import subprocess
import sys

def run_game():
    # Run the play script with specific agents
    result = subprocess.run([
        './play',
        '--player1', 'minimax_with_gpt_greedy_corners',
        '--player2', 'gpt_greedy_corners',
        '--display', 'false',
        '--verbose', 'true'  # Add verbose flag to show moves
    ], capture_output=True, text=True)
    
    # Parse the output to determine winner
    output = result.stdout
    
    # Print all moves from the output
    for line in output.split('\n'):
        if "Player 1 plays at" in line or "Player 2 plays at" in line:
            print(line.strip())
            
    if "Player 1 (Blue) wins!" in output:
        return True  # minimax won
    elif "Player 2 (Brown) wins!" in output:
        return False  # minimax lost
    else:
        return None  # Draw or error

def main():
    num_games = 50
    wins = 0
    draws = 0
    
    print("Running games: Minimax_GPT_Greedy_Corners vs GPT_Greedy_Corners")
    print("-" * 60)
    
    # Run games without progress bar
    for game_num in range(1, num_games + 1):
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
    print(f"Total winrate against gpt_greedy_corners_agent: {(wins/num_games)*100:.2f}%")
    print(f"Draw rate: {(draws/num_games)*100:.2f}%")
    print(f"Loss rate: {(num_games - wins - draws)/num_games*100:.2f}%")
    print(f"Total non-defeat rate: {(wins + draws)/num_games*100:.2f}%")


if __name__ == "__main__":
    main() 


