[![Build Status](https://travis-ci.org/AdamStelmaszczyk/dqn.svg?branch=master)](https://travis-ci.org/AdamStelmaszczyk/dqn)

TensorFlow + Keras implementation of deep Q-learning.

## Install

1. Clone this repo: `git clone https://github.com/AdamStelmaszczyk/dqn.git`.
2. [Install `conda`](https://conda.io/docs/user-guide/install/index.html) for dependency management.
3. Create `dqn` conda environment: `conda create -n dqn python=3.5.2 -y`.
4. Activate `dqn` conda environment: `source activate dqn`. All the following commands should be run in the activated `dqn` environment.
5. Install dependencies: `pip install -r requirements.txt`.

There is an automatic build on Travis which [does the same](https://github.com/AdamStelmaszczyk/dqn/blob/master/.travis.yml).

## Uninstall

1. Deactivate conda environment: `source deactivate`.
2. Remove `dqn` conda environment: `conda env remove -n dqn -y`.

## Usage

Basic file is `run.py`.

```
usage: run.py [-h] [--env ENV] [--clip_rewards] [--model MODEL] [--play]
              [--seed SEED] [--test] [--view MODEL]

optional arguments:
  -h, --help      show this help message and exit
  --env ENV       Atari game name (default: Breakout)
  --clip_rewards  clip rewards to -1/0/1 (default: False)
  --model MODEL   model filename to load (default: None)
  --play          play with WSAD + Space (default: False)
  --seed SEED     pseudo random number generator seed (default: None)
  --test          run tests (default: False)
  --view MODEL    view the model playing the game (default: None)
```

### Train

`python run.py --env Pong`

There are 60 games you can choose from:

`AirRaid, Alien, Amidar, Assault, Asterix, Asteroids, Atlantis, BankHeist, BattleZone, BeamRider, Berzerk, Bowling, Boxing, Breakout, Carnival, Centipede, ChopperCommand, CrazyClimber, DemonAttack, DoubleDunk, ElevatorAction, Enduro, FishingDerby, Freeway, Frostbite, Gopher, Gravitar, Hero, IceHockey, Jamesbond, JourneyEscape, Kangaroo, Krull, KungFuMaster, MontezumaRevenge, MsPacman, NameThisGame, Phoenix, Pitfall, Pong, Pooyan, PrivateEye, Qbert, Riverraid, RoadRunner, Robotank, Seaquest, Skiing, Solaris, SpaceInvaders, StarGunner, Tennis, TimePilot, Tutankham, UpNDown, Venture, VideoPinball, WizardOfWor, YarsRevenge, Zaxxon`

Note that REPLAY_BUFFER_SIZE = 1000000 and stacking 4 frames in observation requires at least 84 \* 84 \* 4 \* 1000000 = 26.3 GB RAM for the replay buffer.

### Play using AI observations

`python run.py --play`

Keys:

- <kbd>W</kbd> - up
- <kbd>S</kbd> - down
- <kbd>A</kbd> - left
- <kbd>D</kbd> - right
- <kbd>SPACE</kbd> - fire button (concrete action depends on a game)

## Links

- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
- https://blog.openai.com/openai-baselines-dqn
- https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
- https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
- https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L483
- https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner