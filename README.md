TensorFlow implementation of deep Q-learning.

## Install

1. Clone this repo: `git clone https://github.com/AdamStelmaszczyk/dqn.git`.
2. [Install `conda`](https://conda.io/docs/user-guide/install/index.html) for dependency management.
3. Create `dqn` conda environment: `conda create -n dqn python=3.5.2`.
4. Activate `dqn` conda environment: `source activate dqn`. All the following commands should be run in the activated `dqn` environment.
5. Install dependencies: `pip install -r requirements.txt`.

## Uninstall

1. Deactivate conda environment: `source deactivate`.

2. Remove `dqn` conda environment: `conda env remove -n dqn`.

## Train

`python run.py -env Breakout`

## Play

`python run.py -play`

## Priorities

1. Correctness.
2. Ease of running on TPU.
3. Simplicity.

## Links

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
https://blog.openai.com/openai-baselines-dqn
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L483
