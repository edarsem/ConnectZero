"""Self-Play Variants for AlphaZero Training."""

from utils import save_model, load_model
from train_agent import collect_self_play_data

import jax


def frozen_self_play(agent, env, save_path, freeze_iteration, total_iterations, batch_size, data_size, simulations):
    """
    Frozen self-play variant. Starts with normal self-play and freezes the agent that will generate the self-play data until the end of training.
    """
    frozen_agent = None
    for iteration in range(total_iterations):
        print(f"Iteration {iteration}: Generating data")
        
        if iteration == freeze_iteration:
            print(f"Freezing agent at iteration {freeze_iteration}")
            save_model(agent, save_path, iteration)
            frozen_agent = load_model(save_path, iteration)

        # Generate data using the frozen agent if applicable
        if frozen_agent:
            collect_self_play_data(
                frozen_agent, 
                env, 
                rng_key=jax.random.PRNGKey(iteration), 
                batch_size=batch_size, 
                data_size=data_size, 
                num_simulations_per_move=simulations
                )
        else:
            collect_self_play_data(
                agent, 
                env, 
                rng_key=jax.random.PRNGKey(iteration), 
                batch_size=batch_size, 
                data_size=data_size, 
                num_simulations_per_move=simulations
                )

        save_model(agent, save_path, iteration)


def self_play_with_all(agent, env, save_path, total_iterations, batch_size, data_size, simulations):
    """
    Self-play with all previous models. Instead of using the current model to generate self-play data, we use agglomerated data from all previous versions of the agent.
    """
    for iteration in range(total_iterations):
        print(f"Iteration {iteration}: Generating data with all previous models")
        save_model(agent, save_path, iteration)
        for prev_iteration in range(iteration + 1):
            print(f"Loading model from iteration {prev_iteration}")
            prev_agent = load_model(save_path, prev_iteration)
            collect_self_play_data(
                prev_agent, 
                env, 
                rng_key=jax.random.PRNGKey(iteration),
                batch_size=batch_size // (iteration + 1),
                data_size=data_size // (iteration + 1),
                num_simulations_per_move=simulations
                )


def self_play_with_fixed_agents(agent, env, fixed_agent_path, fixed_agent_type, batch_size, data_size, simulations):
    """
    Self-play with a fixed weak/strong agent.

    Args:
        agent: The current agent model.
        env: The game environment.
        fixed_agent_path: Path to the fixed agent.
        fixed_agent_type: Type of fixed agent ('weak_agent' or 'strong_agent').
        batch_size: Batch size for self-play.
        data_size: Total size of self-play data.
        simulations: Number of MCTS simulations per move.
    """
    print(f"Loading fixed agent: {fixed_agent_type}")
    fixed_agent = load_model(fixed_agent_path, fixed_agent_type)
    collect_self_play_data(
        fixed_agent, 
        env, 
        rng_key=jax.random.PRNGKey(0),
        batch_size=batch_size, 
        data_size=data_size, 
        num_simulations_per_move=simulations
        )