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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', action='store', default='MontezumaRevenge', help='Atari game name')
    parser.add_argument('--play', action='store_true', default=False, help='Play with WSAD + Space')
    args = parser.parse_args()
    main(args)
