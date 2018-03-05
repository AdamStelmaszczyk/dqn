import argparse

from gym.utils.play import play

from atari_wrappers import wrap_deepmind, make_atari


def main(args):
    env = make_atari('{}NoFrameskip-v4'.format(args.env))
    env = wrap_deepmind(env)
    if args.play:
        play(env)
    else:
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-play', dest='play', action='store_true', default=False)
    parser.add_argument('-env', dest='env', action='store', default='Breakout')
    args = parser.parse_args()
    main(args)
