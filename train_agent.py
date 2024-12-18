import os
import pickle
import csv
import random
from functools import partial
from typing import Optional, List, Tuple, Callable

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
    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class MoveOutput:
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
    def single_move(prev, _):
        env, rng_key, step = prev
        rng_key, rng_key_next = jax.random.split(rng_key, 2)
        state = jax.vmap(lambda e: e.canonical_observation())(env)
        terminated = env.is_terminated()
        policy_output = improve_policy_with_mcts(agent, env, rng_key, recurrent_fn, num_simulations_per_move)
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
        single_move, (env, rng_key, step), None, length=env.max_num_steps(), time_major=False
    )
    return self_play_data


def prepare_training_data(data: MoveOutput, env: Enviroment) -> List[TrainingExample]:
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
    num_iters = data_size // batch_size
    devices = jax.local_devices()
    num_devices = len(devices)
    rng_key_list = jax.random.split(rng_key, num_iters * num_devices)
    rng_keys = jnp.stack(rng_key_list).reshape((num_iters, num_devices, -1))
    data = []

    with click.progressbar(range(num_iters), label="  self play     ") as bar:
        for i in bar:
            batch = collect_batched_self_play_data(
                agent, env, rng_keys[i], batch_size // num_devices, num_simulations_per_move
            )
            batch = jax.device_get(batch)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), batch)
            data.extend(prepare_training_data(batch, env=env))
    return data


def loss_fn(net, data: TrainingExample):
    net, (action_logits, value) = batched_policy(net, data.state)
    mse_loss = jnp.mean(optax.l2_loss(value, data.value))

    target_pr = jnp.where(data.action_weights == 0, EPSILON, data.action_weights)
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
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape, num_actions=env.num_actions()
    )

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    optim = opax.chain(opax.add_decayed_weights(weight_decay), opax.sgd(lr_schedule, momentum=0.9)).init(agent.parameters())
    return agent, optim


def load_checkpoint_if_exists(agent, optim, ckpt_filename: str):
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
    num_devices = len(devices)
    shuffler = random.Random(42)
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as writer:
        dic = {
            "agent": jax.device_get(agent.state_dict()),
            "optim": jax.device_get(optim.state_dict()),
            "iter": iteration,
        }
        pickle.dump(dic, writer)


#############################################
# DATA GENERATION FUNCTIONS
#############################################

def generate_self_play_data_normal(
    iteration: int,
    agent: pax.Module,
    env: Enviroment,
    rng_key: chex.Array,
    selfplay_batch_size: int,
    num_self_plays_per_iteration: int,
    num_simulations_per_move: int,
    **kwargs
) -> List[TrainingExample]:
    # Just generate data from the current agent
    return collect_self_play_data(
        agent.eval(),
        env,
        rng_key,
        selfplay_batch_size,
        num_self_plays_per_iteration,
        num_simulations_per_move,
    )


def generate_self_play_data_frozen(
    iteration: int,
    agent: pax.Module,
    env: Enviroment,
    rng_key: chex.Array,
    selfplay_batch_size: int,
    num_self_plays_per_iteration: int,
    num_simulations_per_move: int,
    freeze_iteration: int,
    aux_dir: str,
    game_class: str,
    agent_class: str,
    frozen_agent_cache: dict,
    **kwargs
) -> List[TrainingExample]:
    # If not frozen yet, just use the current agent
    if iteration < freeze_iteration:
        opponent_agent = agent.eval()
    else:
        if iteration == freeze_iteration:
            print(f"Freezing agent at iteration {freeze_iteration}")
            save_model(agent, aux_dir, freeze_iteration)
            frozen_agent = load_model(game_class, agent_class, aux_dir, freeze_iteration)
            frozen_agent = frozen_agent.eval()
        
        # Load frozen agent if not in cache
        if 'frozen_agent' not in frozen_agent_cache:
            frozen_agent = load_model(game_class, agent_class, aux_dir, freeze_iteration)
            frozen_agent = frozen_agent.eval()
            frozen_agent_cache['frozen_agent'] = frozen_agent
        opponent_agent = frozen_agent_cache['frozen_agent']

    return collect_self_play_data(
        opponent_agent,
        env,
        rng_key,
        selfplay_batch_size,
        num_self_plays_per_iteration,
        num_simulations_per_move,
    )


def generate_self_play_data_vs_all(
    iteration: int,
    agent: pax.Module,
    env: Enviroment,
    rng_key: chex.Array,
    selfplay_batch_size: int,
    num_self_plays_per_iteration: int,
    num_simulations_per_move: int,
    aux_dir: str,
    game_class: str,
    agent_class: str,
    **kwargs
) -> List[TrainingExample]:
    rng_key_1, rng_key = jax.random.split(rng_key)

    if iteration == 0:
        return collect_self_play_data(
            agent.eval(),
            env,
            rng_key_1,
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )
    else:
        # Generate equal fraction of data from each previous agent plus current agent
        data = []
        agents_count = iteration
        data_per_agent = num_self_plays_per_iteration // (agents_count + 1)
        batch_size_per_agent = selfplay_batch_size // (agents_count + 1)

        rng_keys_for_agents = jax.random.split(rng_key_1, agents_count)
        for prev_iter in range(iteration):
            prev_agent = load_model(game_class, agent_class, aux_dir, prev_iter)
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

        # Current agent data
        rng_key_data, _ = jax.random.split(rng_key)
        data_current = collect_self_play_data(
            agent.eval(),
            env,
            rng_key_data,
            batch_size_per_agent,
            data_per_agent,
            num_simulations_per_move,
        )
        data.extend(data_current)
        return data


def generate_self_play_data_vs_other(
    iteration: int,
    agent: pax.Module,
    env: Enviroment,
    rng_key: chex.Array,
    selfplay_batch_size: int,
    num_self_plays_per_iteration: int,
    num_simulations_per_move: int,
    other_agent_ckpt: str,
    game_class: str,
    agent_class: str,
    other_agent_cache: dict,
    **kwargs
) -> List[TrainingExample]:
    # Load other_agent only once and store in cache
    if 'other_agent' not in other_agent_cache:
        if not os.path.isfile(other_agent_ckpt):
            raise FileNotFoundError(f"Other checkpoint not found: {other_agent_ckpt}")
        env_tmp = import_class(game_class)()
        other_agent = import_class(agent_class)(
            input_dims=env_tmp.observation().shape,
            num_actions=env_tmp.num_actions()
        )
        with open(other_agent_ckpt, "rb") as f:
            dic = pickle.load(f)
            other_agent = other_agent.load_state_dict(dic["agent"])
        other_agent = other_agent.eval()
        other_agent_cache['other_agent'] = other_agent
    other_agent = other_agent_cache['other_agent']

    return collect_self_play_data(
        other_agent,
        env,
        rng_key,
        selfplay_batch_size,
        num_self_plays_per_iteration,
        num_simulations_per_move,
    )


#############################################
# GENERIC TRAIN FUNCTION
#############################################

def train_agent_generic(
    game_class: str,
    agent_class: str,
    selfplay_batch_size: int,
    training_batch_size: int,
    num_iterations: int,
    num_simulations_per_move: int,
    num_self_plays_per_iteration: int,
    learning_rate: float,
    ckpt_filename: str,
    random_seed: int,
    weight_decay: float,
    lr_decay_steps: int,
    num_eval_games: int,
    data_generation_fn: Callable,
    output_dir: str,
    aux_dir: str = "",
    **kwargs
):
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    if aux_dir:
        os.makedirs(aux_dir, exist_ok=True)

    env = import_class(game_class)()
    devices = jax.local_devices()
    rng_key = jax.random.PRNGKey(random_seed)
    agent, optim = initialize_agent_and_optim(game_class, agent_class, weight_decay, learning_rate, lr_decay_steps)
    agent, optim, start_iter = load_checkpoint_if_exists(agent, optim, ckpt_filename)

    # Prepare log file
    log_filename = os.path.join(output_dir, "training_log.csv")
    log_exists = os.path.isfile(log_filename)
    with open(log_filename, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if not log_exists:
            writer.writerow(["iteration", "value_loss", "policy_loss", "wins", "draws", "losses"])

    # Caches for loading frozen/other agents only once
    frozen_agent_cache = {}
    other_agent_cache = {}

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)

        data = data_generation_fn(
            iteration=iteration,
            agent=agent,
            env=env,
            rng_key=rng_key_1,
            selfplay_batch_size=selfplay_batch_size,
            num_self_plays_per_iteration=num_self_plays_per_iteration,
            num_simulations_per_move=num_simulations_per_move,
            game_class=game_class,
            agent_class=agent_class,
            aux_dir=aux_dir,
            frozen_agent_cache=frozen_agent_cache,
            other_agent_cache=other_agent_cache,
            **kwargs
        )

        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, optim, value_loss, policy_loss = train_one_epoch(agent, optim, data, training_batch_size, devices)

        agent = agent.eval()
        old_agent = old_agent.eval()

        wins, draws, losses = evaluate_agents_after_training(agent, old_agent, env, rng_key_2, rng_key_3, num_eval_games, num_simulations_per_move)
        print(f"  evaluation      {wins} win - {draws} draw - {losses} loss")
        print(f"  value loss {value_loss:.3f}  policy loss {policy_loss:.3f}")

        # Append results to log
        with open(log_filename, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([iteration, value_loss, policy_loss, wins, draws, losses])

        save_training_state(agent, optim, iteration, ckpt_filename)

        # If we want to save models each iteration for vs_all scenario
        if data_generation_fn == generate_self_play_data_vs_all:
            save_model(agent, aux_dir, iteration)

    print("Done!")


#############################################
# SPECIFIC TRAIN FUNCTIONS
#############################################

def train_normal(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./all_models/normal/agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
):
    output_dir = os.path.dirname(ckpt_filename)
    return train_agent_generic(
        game_class,
        agent_class,
        selfplay_batch_size,
        training_batch_size,
        num_iterations,
        num_simulations_per_move,
        num_self_plays_per_iteration,
        learning_rate,
        ckpt_filename,
        random_seed,
        weight_decay,
        lr_decay_steps,
        num_eval_games,
        data_generation_fn=generate_self_play_data_normal,
        output_dir=output_dir,
    )


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
    ckpt_filename: str = "./all_models/frozen/agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
):
    output_dir = os.path.dirname(ckpt_filename)
    aux_dir = os.path.join(output_dir, "frozen_self_play")
    os.makedirs(aux_dir, exist_ok=True)
    return train_agent_generic(
        game_class,
        agent_class,
        selfplay_batch_size,
        training_batch_size,
        num_iterations,
        num_simulations_per_move,
        num_self_plays_per_iteration,
        learning_rate,
        ckpt_filename,
        random_seed,
        weight_decay,
        lr_decay_steps,
        num_eval_games,
        data_generation_fn=generate_self_play_data_frozen,
        output_dir=output_dir,
        aux_dir=aux_dir,
        freeze_iteration=freeze_iteration,
    )


def train_vs_all(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./all_models/vs_all/agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
):
    output_dir = os.path.dirname(ckpt_filename)
    aux_dir = os.path.join(output_dir, "all_previous_agents")
    os.makedirs(aux_dir, exist_ok=True)
    return train_agent_generic(
        game_class,
        agent_class,
        selfplay_batch_size,
        training_batch_size,
        num_iterations,
        num_simulations_per_move,
        num_self_plays_per_iteration,
        learning_rate,
        ckpt_filename,
        random_seed,
        weight_decay,
        lr_decay_steps,
        num_eval_games,
        data_generation_fn=generate_self_play_data_vs_all,
        output_dir=output_dir,
        aux_dir=aux_dir,
    )


def train_play_vs_other(
    game_class="games.connect_four_game.Connect4Game",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 10,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./all_models/play_vs_other/agent.ckpt",
    other_ckpt_filename: str = "./agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
    num_eval_games: int = 128,
):
    output_dir = os.path.dirname(ckpt_filename)
    return train_agent_generic(
        game_class,
        agent_class,
        selfplay_batch_size,
        training_batch_size,
        num_iterations,
        num_simulations_per_move,
        num_self_plays_per_iteration,
        learning_rate,
        ckpt_filename,
        random_seed,
        weight_decay,
        lr_decay_steps,
        num_eval_games,
        data_generation_fn=generate_self_play_data_vs_other,
        output_dir=output_dir,
        other_agent_ckpt=other_ckpt_filename,
    )


if __name__ == "__main__":
    fire.Fire()