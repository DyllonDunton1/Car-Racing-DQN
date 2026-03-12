from dqn import DQNLearner

lr = 3e-4
replay_buffer_max_size = 200_000
train_start_buffer_length = 20_000
steps_between_fit = 4
target_update_step_interval = 8_000
batch_size = 64
total_steps = 12_000_000
exploration_portion = 0.05
epsilon_final = 0.05
gamma = 0.99
stack_size = 4
render_mode = None
do_flips = True



DQN = DQNLearner(
    "CNN",
    lr,
    replay_buffer_max_size,
    train_start_buffer_length,
    steps_between_fit,
    target_update_step_interval,
    batch_size,
    total_steps,
    exploration_portion,
    epsilon_final,
    gamma,
    stack_size=stack_size,
    render_mode=render_mode,
    do_flips=do_flips
    
)

DQN.episodic_train()