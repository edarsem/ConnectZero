import os
import csv
import pickle
import jax
from typing import List, Tuple
from fire import Fire

from utils import import_class
from play import agent_vs_agent_multiple_games, PlayResults


def load_agents_from_directories(
    directories: List[str],
    game_class: str,
    agent_class: str,
    limit: int = None,
):
    env_class = import_class(game_class)
    env = env_class()
    AgentClass = import_class(agent_class)

    agents = []
    for directory in directories:
        model_files = [f for f in os.listdir(directory) if f.startswith("model_iteration_")]

        def get_iter(f):
            base = os.path.splitext(f)[0]
            iter_str = base.split("_")[-1]
            return int(iter_str)

        model_files = sorted(model_files, key=get_iter)
        if limit is not None:
            model_files = model_files[:limit]

        for f in model_files:
            iteration = get_iter(f)
            file_path = os.path.join(directory, f)
            with open(file_path, "rb") as fp:
                state_dict = pickle.load(fp)
            agent = AgentClass(
                input_dims=env.observation().shape,
                num_actions=env.num_actions()
            ).load_state_dict(state_dict).eval()
            agents.append((iteration, agent, directory))
    return agents


def run_tournament(
    agents: List[Tuple[int, object, str]],
    game_class: str,
    num_games: int = 2,
    num_simulations_per_move: int = 32,
    disable_mcts: bool = False,
):
    """
    Run a double round-robin tournament among the given agents.
    Scoring:
    - win = 1 point
    - draw = 0.5 points
    - loss = 0 points
    """
    env_class = import_class(game_class)
    env = env_class()

    # Sort agents by (method + iteration) to have a consistent order
    agents_sorted = sorted(agents, key=lambda x: (x[2], x[0]))
    num_agents = len(agents_sorted)
    scores = {f"{a[2]}_iter_{a[0]}": 0.0 for a in agents_sorted}

    matchup_matrix = [["-" for _ in range(num_agents)] for _ in range(num_agents)]
    rng = jax.random.PRNGKey(0)

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            iter_i, agent_i, method_i = agents_sorted[i]
            iter_j, agent_j, method_j = agents_sorted[j]

            label_i = f"{method_i}_iter_{iter_i}"
            label_j = f"{method_j}_iter_{iter_j}"

            rng, rng1 = jax.random.split(rng)
            result_1: PlayResults = agent_vs_agent_multiple_games(
                agent_i, agent_j, env, rng1,
                disable_mcts=disable_mcts,
                num_simulations_per_move=num_simulations_per_move,
                num_games=num_games,
            )

            rng, rng2 = jax.random.split(rng)
            result_2: PlayResults = agent_vs_agent_multiple_games(
                agent_j, agent_i, env, rng2,
                disable_mcts=disable_mcts,
                num_simulations_per_move=num_simulations_per_move,
                num_games=num_games,
            )

            score_i = (result_1.win_count * 1.0) + (result_1.draw_count * 0.5)
            score_j = (result_1.loss_count * 1.0) + (result_1.draw_count * 0.5)

            score_j += (result_2.win_count * 1.0) + (result_2.draw_count * 0.5)
            score_i += (result_2.loss_count * 1.0) + (result_2.draw_count * 0.5)

            scores[label_i] += float(score_i)
            scores[label_j] += float(score_j)

            # Store results for agent_i first vs agent_j second (upper triangle)
            wins_ij = int(result_1.win_count)
            draws_ij = int(result_1.draw_count)
            losses_ij = int(result_1.loss_count)
            matchup_matrix[i][j] = f"{wins_ij}:{draws_ij}:{losses_ij}"

            # Store results for agent_j first vs agent_i second (lower triangle)
            wins_ji = int(result_2.win_count)
            draws_ji = int(result_2.draw_count)
            losses_ji = int(result_2.loss_count)
            matchup_matrix[j][i] = f"{wins_ji}:{draws_ji}:{losses_ji}"

    return scores, agents_sorted, matchup_matrix


def test_mode(directories: List[str], game_class: str, agent_class: str, limit: int = None):
    agents = load_agents_from_directories(directories, game_class, agent_class, limit=limit)
    if len(agents) < 1:
        print("No agents found.")
        return

    print(f"Found {len(agents)} agents across directories:")
    agents_sorted = sorted(agents, key=lambda x: x[0])
    for iteration, _, directory in agents_sorted:
        print(f"Iteration {iteration} from {directory}")


def main(
    game_class: str = "games.connect_four_game.Connect4Game",
    agent_class: str = "policies.resnet_policy.ResnetPolicyValueNet",
    directory: str = "./all_models/normal/models_by_iteration ./all_models/frozen/models_by_iteration ./all_models/vs_all/models_by_iteration",
    num_games: int = 2,
    num_simulations_per_move: int = 32,
    disable_mcts: bool = False,
    limit: int = None,
    output_csv: str = "tournament_results.csv",
    test: bool = False,
):
    """
    Run a tournament or test agent discovery.
    """
    directories = directory.split()

    if test:
        test_mode(directories, game_class, agent_class, limit)
        return

    agents = load_agents_from_directories(directories, game_class, agent_class, limit=limit)
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

    print("Results:")
    for iter_num, _, method in agents_sorted:
        label = f"{method}_iter_{iter_num}"
        print(f"{label}: {scores[label]} points")

    labels = [f"{a[2]}_iter_{a[0]}" for a in agents_sorted]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + labels)
        for i, label_i in enumerate(labels):
            row = [label_i] + matchup_matrix[i]
            writer.writerow(row)

    print(f"Cross table saved to {output_csv}")


if __name__ == "__main__":
    Fire(main)