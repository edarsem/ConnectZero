import os
import csv
import pickle
import jax
from typing import List, Tuple
from fire import Fire

from utils import import_class
from play import agent_vs_agent_multiple_games, PlayResults


def load_agents_from_directory(
    directory: str,
    game_class: str,
    agent_class: str,
    limit: int = None,
):
    """
    Load all models from a directory, assumed to be named like model_iteration_X.pkl.
    Return a list of (iteration, agent).
    Sorted by iteration.
    """
    env_class = import_class(game_class)
    env = env_class()
    AgentClass = import_class(agent_class)

    model_files = [f for f in os.listdir(directory) if f.startswith("model_iteration_")]
    def get_iter(f):
        # Filename pattern: model_iteration_XX.pkl
        base = os.path.splitext(f)[0]
        iter_str = base.split("_")[-1]
        return int(iter_str)

    model_files = sorted(model_files, key=get_iter)
    if limit is not None:
        model_files = model_files[:limit]

    agents = []
    for f in model_files:
        iteration = get_iter(f)
        with open(os.path.join(directory, f), "rb") as fp:
            state_dict = pickle.load(fp)
        agent = AgentClass(
            input_dims=env.observation().shape,
            num_actions=env.num_actions()
        ).load_state_dict(state_dict).eval()
        agents.append((iteration, agent))
    return agents


def run_tournament(
    agents: List[Tuple[int, object]],
    game_class: str,
    num_games: int = 2,
    num_simulations_per_move: int = 32,
    disable_mcts: bool = False,
):
    """
    Run a double round-robin tournament.
    `num_games` as agent_i vs agent_j and `num_games` as agent_j vs agent_i.
    Scoring:
    - win = 1 point
    - draw = 0.5 points
    - loss = 0 points

    Also returns a matrix of results (wins, draws, losses) for the scenario "agent_i as first player vs agent_j".
    For i<j this is taken from result_1. For i>=j, we have no direct scenario, so it's "-".
    """
    env_class = import_class(game_class)
    env = env_class()

    num_agents = len(agents)
    scores = {agents[i][0]: 0.0 for i in range(num_agents)}

    # Create a matrix to store W:D:L results for agent_i as first vs agent_j
    # Initialize with '-' for no data
    matchup_matrix = [["-" for _ in range(num_agents)] for _ in range(num_agents)]

    rng = jax.random.PRNGKey(0)

    # Sort agents by iteration so i,j indices match iteration order
    agents_sorted = sorted(agents, key=lambda x: x[0])
    iteration_to_index = {agents_sorted[i][0]: i for i in range(num_agents)}

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            iter_i, agent_i = agents_sorted[i]
            iter_j, agent_j = agents_sorted[j]

            # agent_i as first player, agent_j as second player
            rng, rng1 = jax.random.split(rng)
            result_1: PlayResults = agent_vs_agent_multiple_games(
                agent_i, agent_j, env, rng1,
                disable_mcts=disable_mcts,
                num_simulations_per_move=num_simulations_per_move,
                num_games=num_games,
            )
            # agent_j as first player, agent_i as second player
            rng, rng2 = jax.random.split(rng)
            result_2: PlayResults = agent_vs_agent_multiple_games(
                agent_j, agent_i, env, rng2,
                disable_mcts=disable_mcts,
                num_simulations_per_move=num_simulations_per_move,
                num_games=num_games,
            )

            # Update scores
            # From agent_i perspective (result_1): wins=+1, draws=+0.5, losses=0
            score_i = (result_1.win_count * 1.0) + (result_1.draw_count * 0.5)
            score_j = (result_1.loss_count * 1.0) + (result_1.draw_count * 0.5)

            # From agent_j perspective (result_2)
            score_j += (result_2.win_count * 1.0) + (result_2.draw_count * 0.5)
            score_i += (result_2.loss_count * 1.0) + (result_2.draw_count * 0.5)

            scores[iter_i] += float(score_i)
            scores[iter_j] += float(score_j)

            # Store matchups (i as first player vs j)
            wins = int(result_1.win_count)
            draws = int(result_1.draw_count)
            losses = int(result_1.loss_count)
            matchup_matrix[i][j] = f"{wins}:{draws}:{losses}"

    return scores, agents_sorted, matchup_matrix


def main(
    game_class: str = "games.connect_four_game.Connect4Game",
    agent_class: str = "policies.resnet_policy.ResnetPolicyValueNet",
    directory: str = "./all_models/normal/models_by_iteration",
    num_games: int = 2,
    num_simulations_per_move: int = 32,
    disable_mcts: bool = False,
    limit: int = None,
    output_csv: str = "tournament_results.csv",
):
    """
    Run a tournament among agents from a directory and produce:
    - A printed scoreboard of total points.
    - A CSV file (tournament_results.csv by default) with a cross table of W:D:L results for "agent_i as first vs agent_j".
    """
    agents = load_agents_from_directory(directory, game_class, agent_class, limit=limit)
    if len(agents) < 2:
        print("Not enough agents to run a tournament.")
        return

    scores, agents_sorted, matchup_matrix = run_tournament(
        agents,
        game_class,
        num_games=num_games,
        num_simulations_per_move=num_simulations_per_move,
        disable_mcts=disable_mcts,
    )

    # Print scores
    print("Results:")
    for iter_num, _ in agents_sorted:
        print(f"Iteration {iter_num}: {scores[iter_num]} points")

    # Write CSV
    # First row: Iterations as headers
    iters = [a[0] for a in agents_sorted]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([""] + iters)
        # Rows
        for i, it_i in enumerate(iters):
            row = [it_i] + matchup_matrix[i]
            writer.writerow(row)

    print(f"Cross table saved to {output_csv}")


if __name__ == "__main__":
    Fire(main)