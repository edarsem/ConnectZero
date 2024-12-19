import os
import csv
import pickle
import jax
from typing import List, Tuple
from fire import Fire

from utils import import_class


def load_agents_from_directory(
    directory: str,
    game_class: str,
    agent_class: str,
    limit: int = None,
):
    """
    Load all models from a directory, assumed to be named like model_iteration_X.pkl.
    Return a list of (iteration, agent, file_path).
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
        file_path = os.path.join(directory, f)
        with open(file_path, "rb") as fp:
            state_dict = pickle.load(fp)
        agent = AgentClass(
            input_dims=env.observation().shape,
            num_actions=env.num_actions()
        ).load_state_dict(state_dict).eval()
        agents.append((iteration, agent, file_path))
    return agents


def test_mode(directory: str, game_class: str, agent_class: str, limit: int = None):
    """
    Load and print agent iterations and their paths without running any games.
    """
    agents = load_agents_from_directory(directory, game_class, agent_class, limit=limit)
    if len(agents) < 1:
        print("No agents found.")
        return

    print(f"Found {len(agents)} agents in '{directory}':")
    for iteration, _, file_path in agents:
        print(f"Iteration {iteration}: {file_path}")


def main(
    game_class: str = "games.connect_four_game.Connect4Game",
    agent_class: str = "policies.resnet_policy.ResnetPolicyValueNet",
    directory: str = "./all_models/normal/models_by_iteration",
    num_games: int = 2,
    num_simulations_per_move: int = 32,
    disable_mcts: bool = False,
    limit: int = None,
    output_csv: str = "tournament_results.csv",
    test: bool = False,
):
    """
    Run a tournament or test agent discovery.
    
    If `test=True`, the script will only load and print information about agents without running games.
    """
    if test:
        test_mode(directory, game_class, agent_class, limit)
        return

    # Rest of the tournament logic...
    agents = load_agents_from_directory(directory, game_class, agent_class, limit=limit)
    if len(agents) < 2:
        print("Not enough agents to run a tournament.")
        return

    # Tournament logic goes here...
    # (not included to keep the test mode change concise)


if __name__ == "__main__":
    Fire(main)