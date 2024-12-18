# run_train_variants.py
import jax
from train_agent import train_normal, train_frozen, train_vs_all, train_play_vs_other

if __name__ == "__main__":
    GAME_CLASS = "games.connect_four_game.Connect4Game"
    AGENT_CLASS = "policies.resnet_policy.ResnetPolicyValueNet"
    RANDOM_SEED = 42

    SELFPLAY_BATCH_SIZE = 128
    TRAINING_BATCH_SIZE = 128
    NUM_ITERATIONS = 5
    NUM_SELF_PLAYS_PER_ITERATION = 128 * 10
    NUM_SIMULATIONS_PER_MOVE = 32
    NUM_EVAL_GAMES = 8
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    LR_DECAY_STEPS = 100_000

    print("=== Testing Normal Training ===")
    train_normal(
        game_class=GAME_CLASS,
        agent_class=AGENT_CLASS,
        selfplay_batch_size=SELFPLAY_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_simulations_per_move=NUM_SIMULATIONS_PER_MOVE,
        num_self_plays_per_iteration=NUM_SELF_PLAYS_PER_ITERATION,
        learning_rate=LEARNING_RATE,
        ckpt_filename="./all_models/normal/agent.ckpt",
        random_seed=RANDOM_SEED,
        weight_decay=WEIGHT_DECAY,
        lr_decay_steps=LR_DECAY_STEPS,
        num_eval_games=NUM_EVAL_GAMES,
    )

    print("=== Testing Frozen Self-Play Training ===")
    train_frozen(
        game_class=GAME_CLASS,
        agent_class=AGENT_CLASS,
        selfplay_batch_size=SELFPLAY_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        freeze_iteration=NUM_ITERATIONS//2,
        num_simulations_per_move=NUM_SIMULATIONS_PER_MOVE,
        num_self_plays_per_iteration=NUM_SELF_PLAYS_PER_ITERATION,
        learning_rate=LEARNING_RATE,
        ckpt_filename="./all_models/frozen/agent.ckpt",
        random_seed=RANDOM_SEED,
        weight_decay=WEIGHT_DECAY,
        lr_decay_steps=LR_DECAY_STEPS,
        num_eval_games=NUM_EVAL_GAMES,
    )

    print("=== Testing Training vs All Previous Models ===")
    train_vs_all(
        game_class=GAME_CLASS,
        agent_class=AGENT_CLASS,
        selfplay_batch_size=SELFPLAY_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_simulations_per_move=NUM_SIMULATIONS_PER_MOVE,
        num_self_plays_per_iteration=NUM_SELF_PLAYS_PER_ITERATION,
        learning_rate=LEARNING_RATE,
        ckpt_filename="./all_models/vs_all/agent.ckpt",
        random_seed=RANDOM_SEED,
        weight_decay=WEIGHT_DECAY,
        lr_decay_steps=LR_DECAY_STEPS,
        num_eval_games=NUM_EVAL_GAMES,
    )

    print("=== Testing Training vs Other Model ===")
    train_play_vs_other(
        game_class=GAME_CLASS,
        agent_class=AGENT_CLASS,
        selfplay_batch_size=SELFPLAY_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        num_simulations_per_move=NUM_SIMULATIONS_PER_MOVE,
        num_self_plays_per_iteration=NUM_SELF_PLAYS_PER_ITERATION,
        learning_rate=LEARNING_RATE,
        ckpt_filename="./all_models/play_vs_other/agent.ckpt",
        other_ckpt_filename="./agent.ckpt",
        random_seed=RANDOM_SEED,
        weight_decay=WEIGHT_DECAY,
        lr_decay_steps=LR_DECAY_STEPS,
        num_eval_games=NUM_EVAL_GAMES,
    )

    print("Tests completed!")