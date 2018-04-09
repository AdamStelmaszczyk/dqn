import sys
import time
from math import isnan

import cv2
import deepsense.neptune as neptune
import numpy as np
import psutil
import random
import tensorflow as tf
import tensorflow.contrib.keras as keras
import traceback

from loggers import TensorBoardLogger, NeptuneLogger, AggregatedLogger

try:
    from gym.utils.play import play
except Exception as e:
    print("The following exception is typical for servers because they don't have display stuff installed. "
          "It only means that interactive --play won't work because `from gym.utils.play import play` failed with:")
    traceback.print_exc()
    print("You probably don't need --play on server, so let's continue.")

from atari_wrappers import wrap_deepmind, make_atari
from replay_buffer import ReplayBuffer


def create_goal(position):
    goal = np.zeros(shape=(84, 84, 1))
    box_start = lambda x: (x // BOX_PIXELS) * BOX_PIXELS
    start_x, start_y = map(box_start, position)
    goal[start_x:start_x + BOX_PIXELS, start_y:start_y + BOX_PIXELS, 0] = 255
    return goal


DISCOUNT_FACTOR_GAMMA = 0.99
LEARNING_RATE = 0.0001
UPDATE_EVERY = 4
BATCH_SIZE = 128
TARGET_UPDATE_EVERY = 10000
TRAIN_START = 10000
REPLAY_BUFFER_SIZE = 100000
MAX_STEPS = 10000000
SNAPSHOT_EVERY = 500000
EVAL_EVERY = 100000
EVAL_STEPS = 20000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_STEPS = 100000
LOG_EVERY = 10000
VALIDATION_SIZE = 500
SIDE_BOXES = 4
BOX_PIXELS = 84 // SIDE_BOXES


def one_hot_encode(env, action):
    one_hot = np.zeros(env.action_space.n)
    one_hot[action] = 1
    return one_hot


def predict(env, model, goals, observations):
    frames_input = np.array(observations)
    actions_input = np.ones((len(observations), env.action_space.n))
    goals_input = np.array(goals)
    return model.predict([frames_input, actions_input, goals_input])


def fit_batch(env, model, target_model, batch):
    goals, observations, actions, rewards, next_observations, dones = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = predict(env, target_model, goals, next_observations)
    # The Q values of terminal states is 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    # Passing the actions as the mask and multiplying the targets by the actions masks.
    one_hot_actions = np.array([one_hot_encode(env, action) for action in actions])
    history = model.fit(
        x=[observations, one_hot_actions, goals],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    loss = history.history['loss'][0]
    if isnan(loss):
        print('predicted q_values {}'.format(np.array2string(one_hot_actions * q_values[:, None], threshold=10000000)))
        sys.exit(1)
    return loss


def create_atari_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    print('n_actions {}'.format(n_actions))
    print(' '.join(env.unwrapped.get_action_meanings()))
    print('obs_shape {}'.format(obs_shape))
    frames_input = keras.layers.Input(obs_shape, name='frames_input')
    actions_input = keras.layers.Input((n_actions,), name='actions_input')
    goals_input = keras.layers.Input((84, 84, 1), name='goals_input')
    concatenated = keras.layers.concatenate([frames_input, goals_input])
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(concatenated)
    conv_1 = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(normalized)
    conv_2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(conv_1)
    conv_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(conv_2)
    conv_flattened = keras.layers.Flatten()(conv_3)
    hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])
    model = keras.models.Model([frames_input, actions_input, goals_input], filtered_output)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=0.1)
    model.compile(optimizer, loss='logcosh')
    return model


def epsilon_for_step(step):
    return max(EPSILON_FINAL, (EPSILON_FINAL - EPSILON_START) / EPSILON_STEPS * step + EPSILON_START)


def greedy_action(env, model, goal, observation):
    next_q_values = predict(env, model, goals=[goal], observations=[observation])
    return np.argmax(next_q_values)


def epsilon_greedy_action(env, model, goal, observation, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(env, model, goal, observation)
    return action


def save_model(model, step, logdir, name):
    filename = '{}/{}-{}.h5'.format(logdir, name, step)
    model.save(filename)
    print('Saved {}'.format(filename))
    return filename


def save_image(env, episode, step):
    frame = env.render(mode='rgb_array')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # following cv2.imwrite assumes BGR
    filename = "{}_{:06d}.png".format(episode, step)
    cv2.imwrite(filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])


def evaluate(env, model, view=False, images=False, eval_steps=EVAL_STEPS):
    done = True
    episode = 0
    episode_return_sum = 0.0
    episode_return_min = float('inf')
    episode_return_max = float('-inf')
    for step in range(1, eval_steps):
        if done:
            if episode > 0:
                print("eval episode {} steps {} return {}".format(
                    episode,
                    episode_steps,
                    episode_return,
                ))
                episode_return_sum += episode_return
                episode_return_min = min(episode_return_min, episode_return)
                episode_return_max = max(episode_return_max, episode_return)
            obs = env.reset()
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            goal = sample_goal()
            if view:
                env.render()
            if images:
                save_image(env, episode, step)
        else:
            obs = next_obs
        action = epsilon_greedy_action(env, model, goal, obs, EPSILON_FINAL)
        next_obs, _, done, _ = env.step(action)
        episode_return += goal_reward(next_obs, goal)
        episode_steps += 1
        if view:
            env.render()
        if images:
            save_image(env, episode, step)
    assert episode > 0
    episode_return_avg = episode_return_sum / episode
    return episode_return_avg, episode_return_min, episode_return_max


def find_agent(obs):
    image = obs[:, :, -1]
    indices = np.flatnonzero(image == 110)
    if len(indices) == 0:
        return None
    index = indices[0]
    x = index % 84
    y = index // 84
    return x, y


def goal_reward(obs, goal):
    agent_position = find_agent(obs)
    if agent_position is None:
        return 0
    return int(goal[agent_position] > 0)


def find_last_agent_position(trajectory):
    for experience in reversed(trajectory):
        _, _, _, _, next_obs, _ = experience
        agent = find_agent(next_obs)
        if agent:
            return agent
    return None


def sample_goal():
    position = np.random.randint(0, 84, 2)
    return create_goal(position)


def train(env, env_eval, model, max_steps, name, logdir, logger):
    target_model = create_atari_model(env)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    steps_after_logging = 0
    loss = 0.0
    for step in range(1, max_steps + 1):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(model, step, logdir, name)
            if done:
                if episode > 0:
                    agent = find_last_agent_position(trajectory)
                    if agent:
                        extra_goal = create_goal(agent)
                        for experience in trajectory:
                            goal, obs, action, reward, next_obs, done = experience
                            replay.add(goal, obs, action, reward, next_obs, done)
                            # Hindsight Experience Replay - add experience with an extra goal, that was reached
                            replay.add(extra_goal, obs, action, goal_reward(next_obs, extra_goal), next_obs, done)
                    else:
                        print("Not found the agent in the trajectory - not adding it to the replay")
                    if steps_after_logging >= LOG_EVERY:
                        steps_after_logging = 0
                        episode_end = time.time()
                        episode_seconds = episode_end - episode_start
                        episode_steps = step - episode_start_step
                        steps_per_second = episode_steps / episode_seconds
                        memory = psutil.virtual_memory()
                        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
                        print(
                            "episode {} "
                            "steps {}/{} "
                            "loss {:.7f} "
                            "return {} "
                            "in {:.2f}s "
                            "{:.1f} steps/s "
                            "{:.1f}/{:.1f} GB RAM".format(
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
                        logger.log_scalar('episode_return', episode_return, step)
                        logger.log_scalar('episode_steps', episode_steps, step)
                        logger.log_scalar('episode_seconds', episode_seconds, step)
                        logger.log_scalar('steps_per_second', steps_per_second, step)
                        logger.log_scalar('epsilon', epsilon_for_step(step), step)
                        logger.log_scalar('memory_used', to_gb(memory.used), step)
                        logger.log_scalar('loss', loss, step)
                trajectory = []
                goal = sample_goal()
                episode_start = time.time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0
                epsilon = epsilon_for_step(step)
            else:
                obs = next_obs
            action = epsilon_greedy_action(env, model, goal, obs, epsilon)
            next_obs, _, done, _ = env.step(action)
            reward = goal_reward(next_obs, goal)
            episode_return += reward
            trajectory.append((goal, obs, action, reward, next_obs, done))

            if step >= TRAIN_START and step % UPDATE_EVERY == 0:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.set_weights(model.get_weights())
                batch = replay.sample(BATCH_SIZE)
                loss = fit_batch(env, model, target_model, batch)
            if step == TRAIN_START:
                validation_goals, validation_observations, _, _, _, _ = replay.sample(VALIDATION_SIZE)
            if step >= TRAIN_START and step % EVAL_EVERY == 0:
                episode_return_avg, episode_return_min, episode_return_max = evaluate(env_eval, model)
                q_values = predict(env, model, validation_goals, validation_observations)
                max_q_values = np.max(q_values, axis=1)
                avg_max_q_value = np.mean(max_q_values)
                print(
                    "episode {} "
                    "step {} "
                    "episode_return_avg {:.1f} "
                    "episode_return_min {:.1f} "
                    "episode_return_max {:.1f} "
                    "avg_max_q_value {:.1f}".format(
                        episode,
                        step,
                        episode_return_avg,
                        episode_return_min,
                        episode_return_max,
                        avg_max_q_value,
                    ))
                logger.log_scalar('episode_return_avg', episode_return_avg, step)
                logger.log_scalar('episode_return_min', episode_return_min, step)
                logger.log_scalar('episode_return_max', episode_return_max, step)
                logger.log_scalar('avg_max_q_value', avg_max_q_value, step)
            steps_after_logging += 1
        except KeyboardInterrupt:
            save_model(model, step, logdir, name)
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


def fix_neptune_args(args):
    '''
    neptune run --offline gives 'None' as a default value for string parameters instead of None.
    '''
    for arg in args:
        if args[arg] == 'None':
            args[arg] = None
    return args


def main(context):
    assert BATCH_SIZE <= TRAIN_START <= REPLAY_BUFFER_SIZE
    assert TARGET_UPDATE_EVERY % UPDATE_EVERY == 0
    assert 84 % SIDE_BOXES == 0
    args = fix_neptune_args(context.params)
    print('args: {}'.format({arg: args[arg] for arg in args}))
    env = make_atari('{}NoFrameskip-v4'.format(args.env), max_episode_steps=4000)
    set_seed(env, args.seed)
    if args.play:
        env = wrap_deepmind(env)
        play(env)
    else:
        env_train = wrap_deepmind(env, frame_stack=True, episode_life=True, clip_rewards=True)
        env_eval = wrap_deepmind(env, frame_stack=True)
        model = load_or_create_model(env_train, args.model)
        if args.view or args.images or args.eval:
            evaluate(env_eval, model, args.view, args.images)
        else:
            max_steps = 100 if args.test else MAX_STEPS
            logdir = '{}-log'.format(context.params.name)
            board = TensorBoardLogger(logdir)
            print('Created {}'.format(logdir))
            neptune = NeptuneLogger(context)
            logger = AggregatedLogger([board, neptune])
            train(env_train, env_eval, model, max_steps, args.name, logdir, logger)
            if args.test:
                filename = save_model(model, EVAL_STEPS, logdir='.', name='test')
                load_or_create_model(env_train, filename)


if __name__ == '__main__':
    context = neptune.Context()
    main(context)
