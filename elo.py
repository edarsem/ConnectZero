import argparse
import csv
import random
import math

def update_elo(r_a, r_b, s_a, K=32):
    """Update Elo ratings for a single match.
    s_a is the score of player A (1 win, 0.5 draw, 0 loss).
    """
    # Expected score for A
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    # Update ratings
    r_a_new = r_a + K * (s_a - e_a)
    r_b_new = r_b + K * ((1 - s_a) - (1 - e_a))
    return r_a_new, r_b_new

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Elo ratings from a tournament result CSV.")
    parser.add_argument("--input", type=str, required=True, help="Path to the tournament CSV file.")
    parser.add_argument("--output", type=str, default="elo_ratings.csv", help="Output CSV for final Elo ratings.")
    parser.add_argument("--kfactor", type=float, default=32, help="K-factor for Elo updates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # The first row is the header with player labels
    header = rows[0][1:]
    player_rows = rows[1:]

    players = header

    row_players = [r[0] for r in player_rows]

    # 1000 as initial Elo
    elo = {p: 1000.0 for p in set(players).union(set(row_players))}

    # Format: (player_A, player_B, score_A)
    matches = []

    # Parse the table
    # cell format: "W:D:L"
    # Example: "1:0:7" means 1 win, 0 draws, 7 losses from row_player vs col_player perspective
    # "-" means no matches or self
    for i, row_data in enumerate(player_rows):
        row_player = row_players[i]
        for j, cell in enumerate(row_data[1:], start=1):
            col_player = players[j-1]
            if cell.strip() == '-' or cell.strip() == '':
                continue
            # Parse W:D:L
            parts = cell.split(':')
            if len(parts) != 3:
                continue
            w, d, l = parts
            try:
                w = int(w)
                d = int(d)
                l = int(l)
            except ValueError:
                continue

            # For wins of row_player
            for _ in range(w):
                matches.append((row_player, col_player, 1.0))
            # For draws
            for _ in range(d):
                matches.append((row_player, col_player, 0.5))
            # For losses of row_player
            for _ in range(l):
                matches.append((row_player, col_player, 0.0))

    # Shuffle matches to ensure fairness as Elo is sequential
    random.shuffle(matches)

    for (a, b, s_a) in matches:
        r_a = elo[a]
        r_b = elo[b]
        r_a_new, r_b_new = update_elo(r_a, r_b, s_a, K=args.kfactor)
        elo[a] = r_a_new
        elo[b] = r_b_new

    # Output final Elo ratings
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["player", "elo"])
        for player, rating in sorted(elo.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([player, f"{rating:.2f}"])

    print(f"Elo ratings saved to {args.output}")

if __name__ == "__main__":
    main()