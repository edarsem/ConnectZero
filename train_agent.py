"""
AlphaZero training script.

Contains utilities and subfunctions for training, as well as the
train_normal and train_frozen functions.
"""

import os
import pickle
import random
from functools import partial
from typing import Optional, List, Tuple

import chex
import click
import fire
import jax
import jax.numpy as jnp
import numpy as np
import opax
import optax
import pax

from games.env import Enviroment
from play import PlayResults, agent_vs_agent_multiple_games
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import batched_policy, env_step, import_class, replicate, reset_env, save_model, load_model

EPSILON = 1e-9  # a very small positive value


@chex.dataclass(frozen=True)
class TrainingExample:
    """AlphaZero training example.

    state: the current state of the game.
    action_weights: the target action probabilities from MCTS policy.
    value: the target value from self-play result.
    """

    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class MoveOutput:
    """The output of a single self-play move.

    state: the current state of game.
    reward: the reward after executing the action.
    terminated: whether the current state is terminated.
    action_weights: the action probabilities from MCTS policy.
    """

    state: chex.Array
    reward: chex.Array
    terminated: chex.Array
    action_weights: chex.Array


@partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=(3, 4))
def collect_batched_self_play_data(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    batch_size: int,
    num_simulations_per_move: int,
):
    """Collect a batch of self-play data using mcts."""

    def single_move(prev, _):
        env, rng_key, step = prev
        rng_key, rng_key_next = jax.random.split(rng_key, 2)
        state = jax.vmap(lambda e: e.canonical_observation())(env)
        terminated = env.is_terminated()
        policy_output = improve_policy_with_mcts(
            agent,
            env,
            rng_key,
            recurrent_fn,
            num_simulations_per_move,
        )
        env, reward = jax.vmap(env_step)(env, policy_output.action)
        return (env, rng_key_next, step + 1), MoveOutput(
            state=state,
            action_weights=policy_output.action_weights,
            reward=reward,
            terminated=terminated,
        )

    env = reset_env(env)
    env = replicate(env, batch_size)
    step = jnp.array(1)
    _, self_play_data = pax.scan(
        single_move,
        (env, rng_key, step),
        None,
        length=env.max_num_steps(),
        time_major=False,
    )
    return self_play_data


def prepare_training_data(data: MoveOutput, env: Enviroment) -> List[TrainingExample]:
    """Preprocess the data collected from self-play."""
    buffer = []
    num_games = len(data.terminated)
    for i in range(num_games):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        num_steps = len(is_terminated)
        value: Optional[chex.Array] = None
        for idx in reversed(range(num_steps)):
            if is_terminated[idx]:
                continue
            if value is None:
                value = reward[idx]
            else:
                value = -value
            s = np.copy(state[idx])
            a = np.copy(action_weights[idx])
            for augmented_s, augmented_a in env.symmetries(s, a):
                buffer.append(
                    TrainingExample(
                        state=augmented_s,
                        action_weights=augmented_a,
                        value=np.array(value, dtype=np.float32),
                    )
                )
    return buffer


def collect_self_play_data(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    batch_size: int,
    data_size: int,
    num_simulations_per_move: int,
) -> List[TrainingExample]:
    """Collect self-play data for training."""
    num_iters = data_size // batch_size
    devices = jax.local_devices()
    num_devices = len(devices)
    rng_key_list = jax.random.split(rng_key, num_iters * num_devices)
    rng_keys = jnp.stack(rng_key_list).reshape((num_iters, num_devices, -1))
    data = []

    with click.progressbar(range(num_iters), label="  self play     ") as bar:
        for i in bar:
            batch = collect_batched_self_play_data(
                agent,
                env,
                rng_keys[i],
                batch_size // num_devices,
                num_simulations_per_move,
            )
            batch = jax.device_get(batch)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[2:])), batch
            )
            data.extend(prepare_training_data(batch, env=env))
    return data


def loss_fn(net, data: TrainingExample):
    """Sum of value loss and policy loss."""
    net, (action_logits, value) = batched_policy(net, data.state)

    # value loss (mse)
    mse_loss = optax.l2_loss(value, data.value)
    mse_loss = jnp.mean(mse_loss)

    # policy loss (KL)
    target_pr = data.action_weights
    target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    kl_loss = jnp.sum(target_pr * (jnp.log(target_pr) - action_logits), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    return mse_loss + kl_loss, (net, (mse_loss, kl_loss))


@partial(jax.pmap, axis_name="i")
def train_step(net, optim, data: TrainingExample):
    (_, (net, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


def initialize_agent_and_optim(
    game_class: str,
    agent_class: str,
    weight_decay: float,
    learning_rate: float,
    lr_decay_steps: int,
) -> Tuple[pax.Module, opax.optimizer]:
    """Initialize the agent and optimizer."""
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.sgd(lr_schedule, momentum=0.9),
    ).init(agent.parameters())

    return agent, optim


def load_checkpoint_if_exists(agent, optim, ckpt_filename: str):
    """Load the agent and optimizer states from a checkpoint if it exists."""
    if os.path.isfile(ckpt_filename):
        print("Loading weights at", ckpt_filename)
        with open(ckpt_filename, "rb") as f:
            dic = pickle.load(f)
            agent = agent.load_state_dict(dic["agent"])
            optim = optim.load_state_dict(dic["optim"])
            start_iter = dic["iter"] + 1
    else:
        start_iter = 0
    return agent, optim, start_iter


def train_one_epoch(
    agent: pax.Module,
    optim: opax.optimizer,
    data: List[TrainingExample],
    training_batch_size: int,
    devices,
) -> Tuple[pax.Module, opax.optimizer, float, float]:
    """Train the agent for one epoch on the provided data."""
    num_devices = len(devices)
    shuffler = random.Random(42)

    # Shuffle data before training
    shuffler.shuffle(data)

    def _stack_and_reshape(*xs):
        x = np.stack(xs)
        x = np.reshape(x, (num_devices, -1) + x.shape[1:])
        return x

    agent, optim = jax.device_put_replicated((agent.train(), optim), devices)
    ids = range(0, len(data) - training_batch_size, training_batch_size)
    losses = []
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
    return agent, optim, value_loss, policy_loss


def evaluate_agents_after_training(
    agent: pax.Module,
    old_agent: pax.Module,
    env: Enviroment,
    rng_key_1: chex.Array,
    rng_key_2: chex.Array,
    num_eval_games: int,
    num_simulations_per_move: int,
):
    """Evaluate the new agent against the old agent."""
    result_1: PlayResults = agent_vs_agent_multiple_games(
        agent.eval(),
        old_agent,
        env,
        rng_key_1,
        num_simulations_per_move=num_simulations_per_move,
        num_games=num_eval_games,
    )
    result_2: PlayResults = agent_vs_agent_multiple_games(
        old_agent,
        agent.eval(),
        env,
        rng_key_2,
        num_simulations_per_move=num_simulations_per_move,
        num_games=num_eval_games,
    )
    wins = result_1.win_count + result_2.loss_count
    draws = result_1.draw_count + result_2.draw_count
    losses = result_1.loss_count + result_2.win_count
    return wins, draws, losses


def save_training_state(agent, optim, iteration: int, filename: str):
    """Save the current training state to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as writer:
        dic = {
            "agent": jax.device_get(agent.state_dict()),
            "optim": jax.device_get(optim.state_dict()),
            "iter": iteration,
        }
        pickle.dump(dic, writer)


def train_normal(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
):
    """Normal training loop using self-play data generated by the current agent."""
    env = import_class(game_class)()
    devices = jax.local_devices()
    rng_key = jax.random.PRNGKey(random_seed)

    agent, optim = initialize_agent_and_optim(game_class, agent_class, weight_decay, learning_rate, lr_decay_steps)
    agent, optim, start_iter = load_checkpoint_if_exists(agent, optim, ckpt_filename)

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        agent = agent.eval()
        data = collect_self_play_data(
            agent,
            env,
            rng_key_1,
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )

        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, optim, value_loss, policy_loss = train_one_epoch(agent, optim, data, training_batch_size, devices)
        wins, draws, losses = evaluate_agents_after_training(agent, old_agent, env, rng_key_2, rng_key_3, num_eval_games, num_simulations_per_move)

        print(f"  evaluation      {wins} win - {draws} draw - {losses} loss")
        print(f"  value loss {value_loss:.3f}  policy loss {policy_loss:.3f}")

        save_training_state(agent, optim, iteration, ckpt_filename)
    print("Done!")


def train_frozen(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    freeze_iteration: int = 5,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./agent_frozen.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
    base_path: str = "./models",
):
    """
    Frozen Self-Play:
    - Generate data with the current agent until `freeze_iteration`.
    - At `freeze_iteration`, save and freeze that agent.
    - Beyond `freeze_iteration`, generate data using the frozen agent, but still train the main agent.
    """
    env = import_class(game_class)()
    devices = jax.local_devices()
    rng_key = jax.random.PRNGKey(random_seed)

    agent, optim = initialize_agent_and_optim(game_class, agent_class, weight_decay, learning_rate, lr_decay_steps)
    agent, optim, start_iter = load_checkpoint_if_exists(agent, optim, ckpt_filename)

    model_dir = os.path.join(base_path, "frozen_self_play")
    os.makedirs(model_dir, exist_ok=True)
    frozen_agent = None

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")

        # Freeze agent if we hit the freeze iteration
        if iteration == freeze_iteration:
            print(f"Freezing agent at iteration {freeze_iteration}")
            save_model(agent, model_dir, freeze_iteration)
            frozen_agent = load_model(game_class, agent_class, model_dir, freeze_iteration)
            frozen_agent = frozen_agent.eval()

        opponent_agent = frozen_agent if frozen_agent else agent.eval()

        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        data = collect_self_play_data(
            opponent_agent,
            env,
            rng_key_1,
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )

        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, optim, value_loss, policy_loss = train_one_epoch(agent, optim, data, training_batch_size, devices)

        agent = agent.eval()
        old_agent = old_agent.eval()

        wins, draws, losses = evaluate_agents_after_training(
            agent, old_agent, env, rng_key_2, rng_key_3, num_eval_games, num_simulations_per_move
        )

        print(f"  evaluation      {wins} win - {draws} draw - {losses} loss")
        print(f"  value loss {value_loss:.3f}  policy loss {policy_loss:.3f}")

        # save agent's weights to disk
        save_training_state(agent, optim, iteration, ckpt_filename)
    print("Done!")


def train_vs_all(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./models/train_vs_all/agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
    base_path: str = "./models",
):
    """
    Train vs all previous models:
    - On iteration i, 1/ith of the self-play data generated by each of the previous agents.
    - Combine all data and train the current agent.
    """
    env = import_class(game_class)()
    devices = jax.local_devices()
    rng_key = jax.random.PRNGKey(random_seed)

    model_dir = os.path.join(base_path, "self_play_all")
    os.makedirs(model_dir, exist_ok=True)

    agent, optim = initialize_agent_and_optim(game_class, agent_class, weight_decay, learning_rate, lr_decay_steps)
    agent, optim, start_iter = load_checkpoint_if_exists(agent, optim, ckpt_filename)

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")

        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        
        if iteration == 0:
            # Just generate data from the current agent
            data = collect_self_play_data(
                agent.eval(),
                env,
                rng_key_1,
                selfplay_batch_size,
                num_self_plays_per_iteration,
                num_simulations_per_move,
            )
        else:
            # Generate equal fraction of data from each previous agent
            data = []
            agents_count = iteration  # number of previously saved agents is `iteration`
            data_per_agent = num_self_plays_per_iteration // (agents_count + 1)
            batch_size_per_agent = selfplay_batch_size // (agents_count + 1)

            # Load previous agents and generate data
            rng_keys_for_agents = jax.random.split(rng_key_1, agents_count)
            for prev_iter in range(iteration):
                prev_agent = load_model(game_class, agent_class, model_dir, prev_iter)
                prev_agent = prev_agent.eval()
                data_prev = collect_self_play_data(
                    prev_agent,
                    env,
                    rng_keys_for_agents[prev_iter],
                    batch_size_per_agent,
                    data_per_agent,
                    num_simulations_per_move,
                )
                data.extend(data_prev)

            # Also generate data from the current agent
            rng_key_data, rng_key = jax.random.split(rng_key)
            data_current = collect_self_play_data(
                agent.eval(),
                env,
                rng_key_data,
                batch_size_per_agent,
                data_per_agent,
                num_simulations_per_move,
            )
            data.extend(data_current)

        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, optim, value_loss, policy_loss = train_one_epoch(agent, optim, data, training_batch_size, devices)

        # Set both agents to eval mode for evaluation
        agent = agent.eval()
        old_agent = old_agent.eval()

        wins, draws, losses = evaluate_agents_after_training(agent, old_agent, env, rng_key_2, rng_key_3, num_eval_games, num_simulations_per_move)

        print(f"  evaluation      {wins} win - {draws} draw - {losses} loss")
        print(f"  value loss {value_loss:.3f}  policy loss {policy_loss:.3f}")

        # Save current agent for future usage
        save_training_state(agent, optim, iteration, ckpt_filename)
        # Also save a copy of it for later loading
        save_model(agent, model_dir, iteration)

    print("Done!")


if __name__ == "__main__":
    fire.Fire()