"""Self-Play Variants for AlphaZero Training."""

import os
import pickle
import random
import click
import numpy as np
import jax
import opax
import jax.numpy as jnp

from train_agent import train, collect_self_play_data, prepare_training_data, train_step, MoveOutput
from utils import import_class, save_model, load_model, make_directories
from play import agent_vs_agent_multiple_games, PlayResults

def frozen_self_play(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    base_path: str = "./models",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    freeze_iteration: int = 5,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
    **train_kwargs,
):
    """
    Frozen Self-Play:
    - Generate data from the normal model until `freeze_iteration`.
    - Freeze the agent at `freeze_iteration`.
    - Train the current agent against the frozen agent.
    """
    model_dir = make_directories(base_path, "frozen_self_play")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape, num_actions=env.num_actions()
    )

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.sgd(lr_schedule, momentum=0.9),
    ).init(agent.parameters())

    if os.path.isfile(ckpt_filename):
        print("Loading weights at", ckpt_filename)
        with open(ckpt_filename, "rb") as f:
            dic = pickle.load(f)
            agent = agent.load_state_dict(dic["agent"])
            optim = optim.load_state_dict(dic["optim"])
            start_iter = dic["iter"] + 1
    else:
        start_iter = 0

    rng_key = jax.random.PRNGKey(random_seed)
    shuffler = random.Random(random_seed)
    devices = jax.local_devices()
    num_devices = jax.local_device_count()

    def _stack_and_reshape(*xs):
        x = np.stack(xs)
        x = np.reshape(x, (num_devices, -1) + x.shape[1:])
        return x

    frozen_agent = None

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}: Training and Generating Data")

        if iteration == freeze_iteration:
            print(f"Freezing agent at iteration {freeze_iteration}")
            save_model(agent, model_dir, freeze_iteration)
            frozen_agent = load_model(model_dir, freeze_iteration)
            frozen_agent = frozen_agent.eval()

        # Generate data with the appropriate agent
        opponent_agent = frozen_agent if frozen_agent else agent
        opponent_agent = opponent_agent.eval()

        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        data = collect_self_play_data(
            opponent_agent,
            env,
            rng_key_1,
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )

        data = list(data)
        shuffler.shuffle(data)
        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, losses = agent.train(), []
        agent, optim = jax.device_put_replicated((agent, optim), devices)
        ids = range(0, len(data) - training_batch_size, training_batch_size)
        with click.progressbar(ids, label="  train agent   ") as progressbar:
            for idx in progressbar:
                batch = data[idx : (idx + training_batch_size)]
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = np.mean(sum(jax.device_get(value_loss))) / len(value_loss)
        policy_loss = np.mean(sum(jax.device_get(policy_loss))) / len(policy_loss)
        agent, optim = jax.tree_util.tree_map(lambda x: x[0], (agent, optim))
        # new agent is player 1
        result_1: PlayResults = agent_vs_agent_multiple_games(
            agent.eval(),
            old_agent,
            env,
            rng_key_2,
            num_simulations_per_move=32,
            num_games=num_eval_games,
        )
        # old agent is player 1
        result_2: PlayResults = agent_vs_agent_multiple_games(
            old_agent,
            agent.eval(),
            env,
            rng_key_3,
            num_simulations_per_move=32,
            num_games=num_eval_games,
        )
        print(
            "  evaluation      {} win - {} draw - {} loss".format(
                result_1.win_count + result_2.loss_count,
                result_1.draw_count + result_2.draw_count,
                result_1.loss_count + result_2.win_count,
            )
        )
        print(
            f"  value loss {value_loss:.3f}"
            f"  policy loss {policy_loss:.3f}"
            f"  learning rate {optim[1][-1].learning_rate:.1e}"
        )
        # save agent's weights to disk
        with open(os.path.join(model_dir, f"model_iteration_{iteration}.pkl"), "wb") as writer:
            dic = {
                "agent": jax.device_get(agent.state_dict()),
                "optim": jax.device_get(optim.state_dict()),
                "iter": iteration,
            }
            pickle.dump(dic, writer)
    print("Done!")


def self_play_with_all(
    game_class, agent_class, base_path, total_iterations, **train_kwargs
):
    """
    Self-Play with All Previous Models:
    - Each iteration, data is generated by all previous agents.
    """
    model_dir = make_directories(base_path, "self_play_all")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape, num_actions=env.num_actions()
    )

    for iteration in range(total_iterations):
        print(f"Iteration {iteration}: Generating Data from All Previous Models")
        save_model(agent, model_dir, iteration)

        for prev_iter in range(iteration + 1):
            prev_agent = load_model(model_dir, prev_iter)
            rng_key = jax.random.PRNGKey(prev_iter)
            collect_self_play_data(
                prev_agent,
                env,
                rng_key,
                train_kwargs["selfplay_batch_size"] // (iteration + 1),
                train_kwargs["num_self_plays_per_iteration"] // (iteration + 1),
                train_kwargs["num_simulations_per_move"],
            )


def self_play_with_fixed_agents(
    game_class,
    agent_class,
    base_path,
    fixed_agents_dir,
    agent_type,
    **train_kwargs,
):
    """
    Self-Play with Fixed Weak/Strong Agents.
    """
    model_dir = make_directories(base_path, "fixed_agents")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape, num_actions=env.num_actions()
    )

    # Load fixed agent
    fixed_agent_path = os.path.join(fixed_agents_dir, f"{agent_type}.pkl")
    fixed_agent = load_model(fixed_agent_path, 0)

    print(f"Training agent against fixed {agent_type} agent")

    for iteration in range(train_kwargs["num_iterations"]):
        print(f"Iteration {iteration}: Self-play against {agent_type}")
        rng_key = jax.random.PRNGKey(iteration)
        data = collect_self_play_data(
            fixed_agent,
            env,
            rng_key,
            train_kwargs["selfplay_batch_size"],
            train_kwargs["num_self_plays_per_iteration"],
            train_kwargs["num_simulations_per_move"],
        )

        # Prepare training data and update agent
        training_data = prepare_training_data(data, env)
        agent = train_one_iteration(agent, training_data)

        save_model(agent, model_dir, iteration)


def train_one_iteration(agent, training_data):
    """Perform one training iteration."""

    learning_rate = 0.01
    optim = opax.chain(
        opax.add_decayed_weights(1e-4),
        opax.sgd(learning_rate, momentum=0.9),
    ).init(agent.parameters())

    for batch in training_data:
        agent, optim, _ = train_step(agent, optim, batch)
    return agent


if __name__ == "__main__":
    BASE_PATH = "./models"
    GAME_CLASS = "games.connect_four_game.Connect4Game"
    AGENT_CLASS = "policies.resnet_policy.ResnetPolicyValueNet"

    TRAIN_KWARGS = {
        "selfplay_batch_size": 128,
        "num_self_plays_per_iteration": 1280,
        "num_simulations_per_move": 32,
        "num_iterations": 10,
        "num_eval_games": 16,
    }

    # print("Starting Normal Self-Play...")
    # train_normal(GAME_CLASS, AGENT_CLASS, BASE_PATH, **TRAIN_KWARGS)

    print("Starting Frozen Self-Play...")
    frozen_self_play(
        GAME_CLASS, AGENT_CLASS, BASE_PATH, freeze_iteration=5, total_iterations=10, **TRAIN_KWARGS
    )

    print("Starting Self-Play with All Previous Models...")
    self_play_with_all(GAME_CLASS, AGENT_CLASS, BASE_PATH, total_iterations=10, **TRAIN_KWARGS)

    # print("Starting Self-Play with Fixed Agents...")
    # FIXED_AGENTS_DIR = "./models/fixed_agents"
    # self_play_with_fixed_agents(
    #     GAME_CLASS, AGENT_CLASS, BASE_PATH, FIXED_AGENTS_DIR, agent_type="weak_agent", **TRAIN_KWARGS
    # )