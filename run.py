import argparse
import random
import time

import numpy as np
import psutil
import pyglet
import tensorflow.contrib.keras as keras

try:
    from gym.utils.play import play
except pyglet.canvas.xlib.NoSuchDisplayException:
    print("No X server running on $DISPLAY, so interactive --play won't work.")

from atari_wrappers import wrap_deepmind, make_atari
from replay_buffer import ReplayBuffer
from tensor_board_logger import TensorBoardLogger

DISCOUNT_FACTOR_GAMMA = 0.99
UPDATE_FREQUENCY = 4
BATCH_SIZE = 32
REPLAY_START_SIZE = 50000
REPLAY_BUFFER_SIZE = 1000000
MAX_TIME_STEPS = 10000000
SNAPSHOT_EVERY = 1000000
LOG_EVERY = 10000


def one_hot_encode(env, action):
    one_hot = np.zeros(env.action_space.n)
    one_hot[action] = 1
    return one_hot


def fit_batch(env, model, batch):
    observations, actions, rewards, next_observations, dones = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = model.predict([next_observations, np.ones((BATCH_SIZE, env.action_space.n))])
    # The Q values of terminal states is 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    # Passing the actions as the mask and multiplying the targets by the actions masks.
    one_hot_actions = np.array([one_hot_encode(env, action) for action in actions])
    model.fit(
        x=[observations, one_hot_actions],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )


def create_atari_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    print('n_actions {}'.format(n_actions))
    print('obs_shape {}'.format(obs_shape))
    frames_input = keras.layers.Input(obs_shape, name='frames_input')
    actions_input = keras.layers.Input((n_actions,), name='actions_input')
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')(normalized)
    conv_2 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(conv_1)
    conv_flattened = keras.layers.Flatten()(conv_2)
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])
    model = keras.models.Model([frames_input, actions_input], filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model


def epsilon_for_step(step):
    # epsilon annealed linearly from 1 to 0.1 over first million of steps and fixed at 0.1 thereafter
    return max(-9e-7 * step + 1, 0.1)


def greedy_action(env, model, observation):
    next_q_values = model.predict([np.array([observation]), np.ones((1, env.action_space.n))])
    return np.argmax(next_q_values)


def epsilon_greedy(env, model, observation, step):
    epsilon = epsilon_for_step(step)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(env, model, observation)
    return action


def save_model(env, model, step):
    filename = '{}-{}-{}.h5'.format(env.spec.id, time.strftime("%m-%d-%H-%M"), step)
    model.save(filename)
    print('Saved {}'.format(filename))


def train(env, model, max_time_steps):
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    logdir = '{}-{}-log'.format(env.spec.id, time.strftime("%m-%d-%H-%M"))
    board = TensorBoardLogger(logdir)
    print('Created {}'.format(logdir))
    steps_after_logging = 0
    for step in range(1, max_time_steps):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(env, model, step)
            if done:
                if episode > 0 and steps_after_logging >= LOG_EVERY:
                    steps_after_logging = 0
                    episode_end = time.time()
                    episode_seconds = episode_end - episode_start
                    episode_steps = step - episode_start_step
                    steps_per_second = episode_steps / episode_seconds
                    memory = psutil.virtual_memory()
                    to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
                    print("episode {} steps {}/{} return {} in {:.2f}s {:.1f} steps/s {:.1f}/{:.1f} GB RAM".format(
                        episode,
                        episode_steps,
                        step,
                        episode_return,
                        episode_seconds,
                        steps_per_second,
                        to_gb(memory.used),
                        to_gb(memory.total),
                    ))
                    board.log_scalar('episode_return', episode_return, step)
                    board.log_scalar('episode_steps', episode_steps, step)
                    board.log_scalar('episode_seconds', episode_seconds, step)
                    board.log_scalar('steps_per_second', steps_per_second, step)
                    board.log_scalar('epsilon', epsilon_for_step(step), step)
                    board.log_scalar('memory_used', to_gb(memory.used), step)
                episode_start = time.time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0
            else:
                obs = next_obs
            action = epsilon_greedy(env, model, obs, step)
            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            replay.add(obs, action, reward, next_obs, done)
            if step >= REPLAY_START_SIZE and step % UPDATE_FREQUENCY == 0:
                batch = replay.sample(BATCH_SIZE)
                fit_batch(env, model, batch)
            steps_after_logging += 1
        except KeyboardInterrupt:
            save_model(env, model, step)
            break


def main(args):
    assert BATCH_SIZE <= REPLAY_START_SIZE <= REPLAY_BUFFER_SIZE
    random.seed(args.seed)
    env = make_atari('{}NoFrameskip-v4'.format(args.env))
    env.seed(args.seed)
    if args.play:
        env = wrap_deepmind(env, frame_stack=False)
        play(env)
    else:
        env = wrap_deepmind(env, frame_stack=True)
        if args.model:
            model = keras.models.load_model(args.model)
            print('Loaded {}'.format(args.model))
        else:
            model = create_atari_model(env)
        if args.test:
            train(env, model, max_time_steps=100)
        else:
            train(env, model, max_time_steps=MAX_TIME_STEPS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', action='store', default='Breakout', help='Atari game name')
    parser.add_argument('--model', action='store', default=None, help='h5 model filename to load')
    parser.add_argument('--play', action='store_true', default=False, help='play with WSAD + Space')
    parser.add_argument('--seed', action='store', type=int, help='pseudo random number generator seed')
    parser.add_argument('--test', action='store_true', default=False, help='run tests')
    args = parser.parse_args()
    main(args)
