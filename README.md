# EnvPool + CleanRL

This repo contains the source code of the CleanRL experiments presented in [**EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine**](https://arxiv.org/abs/2206.10558)

* Paper url: https://arxiv.org/abs/2206.10558
* Tracked Weights and Biases experiments: https://wandb.ai/openrlbenchmark/envpool-cleanrl

If you like this repo, consider checking out CleanRL (https://github.com/vwxyzjn/cleanrl), the RL library that we used to build this repo.


## Get started

Prerequisites:
* Python 3.9+
* [Poetry 1.2+](https://python-poetry.org)

Install dependencies:
```
poetry install
```

By default, the `torch` wheel is built with CUDA 10.2. If you are using newer NVIDIA GPU (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the `torch` dependency with `pip`:

```
poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
```


### Atari
Train agents:
```
poetry run python ppo_atari.py
```
Train agents with experiment tracking:
```
poetry run python ppo_atari.py --track
```


### MuJoCo
Train agents:
```
poetry run python ppo_continuous_action.py
```
Train agents with experiment tracking:
```
poetry run python ppo_continuous_action.py --track
```

## Replicating results from the paper

See `benchmark.sh`

## Citation

```bibtex
@article{envpool,
  title={EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine},
  author={Weng, Jiayi and Lin, Min and Huang, Shengyi and Liu, Bo and Makoviichuk, Denys and Makoviychuk, Viktor and Liu, Zichen and Song, Yufan and Luo, Ting and Jiang, Yukun and Xu, Zhongwen and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2206.10558},
  year={2022}
}
```
