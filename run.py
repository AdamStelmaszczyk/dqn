import time

import argparse
import numpy as np
import psutil
import pyglet
import random
import tensorflow as tf
import tensorflow.contrib.keras as keras

try:
    from gym.utils.play import play
except pyglet.canvas.xlib.NoSuchDisplayException:
    print("No X server running on $DISPLAY, so interactive --play won't work.")

from atari_wrappers import wrap_deepmind, make_atari
from replay_buffer import ReplayBuffer
from tensor_board_logger import TensorBoardLogger

DISCOUNT_FACTOR_GAMMA = 0.99
LEARNING_RATE = 0.0001
UPDATE_EVERY = 4
BATCH_SIZE = 128
TARGET_UPDATE_EVERY = 1000
TRAIN_START = 10000
REPLAY_BUFFER_SIZE = 100000
MAX_STEPS = 10000000
SNAPSHOT_EVERY = 500000
EVAL_EVERY = 100000
EVAL_STEPS = 10000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_STEPS = 100000
LOG_EVERY = 10000
VALIDATION_SIZE = 500


def one_hot_encode(env, action):
    one_hot = np.zeros(env.action_space.n)
    one_hot[action] = 1
    return one_hot


def predict(env, model, observations):
    frames_input = np.array(observations)
    actions_input = np.ones((len(observations), env.action_space.n))
    return model.predict([frames_input, actions_input])


def fit_batch(env, model, target_model, batch):
    observations, actions, rewards, next_observations, dones = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = predict(env, target_model, next_observations)
    # The Q values of terminal states is 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    # Passing the actions as the mask and multiplying the targets by the actions masks.
    one_hot_actions = np.array([one_hot_encode(env, action) for action in actions])
    history = model.fit(
        x=[observations, one_hot_actions],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return history.history['loss'][0]


def create_atari_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    print('n_actions {}'.format(n_actions))
    print(' '.join(env.unwrapped.get_action_meanings()))
    print('obs_shape {}'.format(obs_shape))
    frames_input = keras.layers.Input(obs_shape, name='frames_input')
    actions_input = keras.layers.Input((n_actions,), name='actions_input')
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(normalized)
    conv_2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(conv_1)
    conv_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(conv_2)
    conv_flattened = keras.layers.Flatten()(conv_3)
    hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])
    model = keras.models.Model([frames_input, actions_input], filtered_output)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer, loss='logcosh')
    return model


def epsilon_for_step(step):
    return max(EPSILON_FINAL, (EPSILON_FINAL - EPSILON_START) / EPSILON_STEPS * step + EPSILON_START)


def greedy_action(env, model, observation):
    next_q_values = predict(env, model, observations=[observation])
    return np.argmax(next_q_values)


def epsilon_greedy_action(env, model, observation, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(env, model, observation)
    return action


def save_model(model, step, name):
    filename = '{}-{}.h5'.format(name, step)
    model.save(filename)
    print('Saved {}'.format(filename))


def evaluate(env, model, view=False, eval_steps=EVAL_STEPS):
    done = True
    episode = 0
    episode_return = 0.0
    total_episode_reward = 0.0
    for i in range(1, eval_steps):
        if done:
            if episode > 0:
                print("episode {} steps {} return {}".format(
                    episode,
                    episode_steps,
                    episode_return,
                ))
            obs = env.reset()
            total_episode_reward += episode_return
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            if view:
                env.render()
        else:
            obs = next_obs
        action = epsilon_greedy_action(env, model, obs, EPSILON_FINAL)
        next_obs, reward, done, _ = env.step(action)
        episode_return += reward
        episode_steps += 1
        if view:
            env.render()
    return total_episode_reward / episode


def train(env, env_eval, model, max_steps, name):
    target_model = create_atari_model(env)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    logdir = '{}-log'.format(name)
    board = TensorBoardLogger(logdir)
    print('Created {}'.format(logdir))
    steps_after_logging = 0
    loss = 0.0
    for step in range(1, max_steps + 1):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(model, step, name)
            if done:
                if episode > 0 and steps_after_logging >= LOG_EVERY:
                    steps_after_logging = 0
                    episode_end = time.time()
                    episode_seconds = episode_end - episode_start
                    episode_steps = step - episode_start_step
                    steps_per_second = episode_steps / episode_seconds
                    memory = psutil.virtual_memory()
                    to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
                    print(
                        "episode {} steps {}/{} loss {:.7f} return {} in {:.2f}s {:.1f} steps/s {:.1f}/{:.1f} GB RAM".format(
                            episode,
                            episode_steps,
                            step,
                            loss,
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
                    board.log_scalar('loss', loss, step)
                episode_start = time.time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0
            else:
                obs = next_obs
            epsilon = epsilon_for_step(step)
            action = epsilon_greedy_action(env, model, obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            replay.add(obs, action, reward, next_obs, done)
            if step >= TRAIN_START and step % UPDATE_EVERY == 0:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.set_weights(model.get_weights())
                batch = replay.sample(BATCH_SIZE)
                loss = fit_batch(env, model, target_model, batch)
            if step == TRAIN_START:
                validation_obs, _, _, _, _ = replay.sample(VALIDATION_SIZE)
            if step >= TRAIN_START and step % EVAL_EVERY == 0:
                avg_episode_reward = evaluate(env_eval, model)
                q_values = predict(env, model, validation_obs)
                max_q_values = np.max(q_values, axis=1)
                avg_max_q_value = np.mean(max_q_values)
                print('episode {} step {} avg_episode_reward {:.1f} avg_max_q_value {:.1f}'.format(
                    episode,
                    step,
                    avg_episode_reward,
                    avg_max_q_value,
                ))
                board.log_scalar('avg_episode_reward', avg_episode_reward, step)
                board.log_scalar('avg_max_q_value', avg_max_q_value, step)
            steps_after_logging += 1
        except KeyboardInterrupt:
            save_model(model, step, name)
            break


def load_or_create_model(env, model_filename):
    if model_filename:
        model = keras.models.load_model(model_filename)
        print('Loaded {}'.format(model_filename))
    else:
        model = create_atari_model(env)
    return model


def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)


def main(args):
    assert BATCH_SIZE <= TRAIN_START <= REPLAY_BUFFER_SIZE
    assert TARGET_UPDATE_EVERY % UPDATE_EVERY == 0
    print(args)
    env = make_atari('{}NoFrameskip-v4'.format(args.env))
    set_seed(env, args.seed)
    if args.play:
        env = wrap_deepmind(env)
        play(env)
    else:
        env_train = wrap_deepmind(env, frame_stack=True, episode_life=True, clip_rewards=args.clip_rewards)
        env_eval = wrap_deepmind(env, frame_stack=True)
        model_filename = args.model or args.view
        model = load_or_create_model(env_train, model_filename)
        if args.view:
            evaluate(env_eval, model, view=True)
        else:
            max_steps = 100 if args.test else MAX_STEPS
            train(env_train, env_eval, model, max_steps, args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', action='store', default='Breakout', help='Atari game name')
    parser.add_argument('--clip_rewards', action='store_true', default=False, help='clip rewards to -1/0/1')
    parser.add_argument('--model', action='store', default=None, help='model filename to load')
    parser.add_argument('--name', action='store', default=time.strftime("%m-%d-%H-%M"), help='name for saved files')
    parser.add_argument('--play', action='store_true', default=False, help='play with WSAD + Space')
    parser.add_argument('--seed', action='store', type=int, help='pseudo random number generator seed')
    parser.add_argument('--test', action='store_true', default=False, help='run tests')
    parser.add_argument('--view', action='store', metavar='MODEL', default=None, help='view the model playing the game')
    main(parser.parse_args())
