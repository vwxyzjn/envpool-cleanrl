from turtle import width
import numpy as np
import expt
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot
from tueplots import bundles, figsizes

api = wandb.Api()

def create_hypothesis(name, wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()

        # data cleaning
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        wandb_run["_runtime"] /= 60  # convert to minutes

        if "rewards/step" in wandb_run:
            wandb_run["return"] = wandb_run["rewards/step"]
        if "charts/episodic_return" in wandb_run:
            wandb_run["return"] = wandb_run["charts/episodic_return"]

        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


env_ids = ["Walker2d-v4", "HalfCheetah-v4"]
nrows = 1
ncols = 4
plt.rcParams.update(bundles.neurips2022())
 
plt.rcParams.update(figsizes.neurips2022(rel_width=1.2, nrows=nrows, ncols=ncols, height_to_width_ratio=0.9))
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    # sharex=True,
    # sharey=True,
)

env_ids
env_id = "Ant-v4"
ex = expt.Experiment("Comparison of PPO")
wandb_runs = api.runs(
    path="openrlbenchmark/rl_games",
    filters={"$and": [{"config.params.value.config.env_config.env_name": env_id}, {"config.params.value.config.num_actors": 64}]},
)
h = create_hypothesis("rl_games' PPO + EnvPool (N=64)", wandb_runs)
ex.add_hypothesis(h)
wandb_runs = api.runs(
    path="openrlbenchmark/envpool-cleanrl",
    filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_continuous_action_envpool"}]},
)
h = create_hypothesis("CleanRL's PPO + EnvPool (N=64)", wandb_runs)
ex.add_hypothesis(h)
wandb_runs = api.runs(
    path="openrlbenchmark/baselines",
    filters={"$and": [{"config.env.value": env_id.replace("v4", "v2")}, {"config.exp_name.value": "baselines-ppo2-mlp"}]},
)
h = create_hypothesis("openai/baselines' PPO (N=1)", wandb_runs)
ex.add_hypothesis(h)


ax = axes.flatten()[0]
ex.plot(
    ax=ax,
    title=env_id,
    x="global_step",
    y="return",
    err_style="band",
    std_alpha=0.1,
    rolling=50,
    n_samples=400,
    legend=False,
    tight_layout=False,
    suptitle=False,
    # colors=["#1f77b4", "#ff7f0e"],
)
ax.xaxis.set_label_text("Frames")
ax.yaxis.set_label_text("Episodic Return")
h, l = ax.get_legend_handles_labels()
ax = axes.flatten()[1]
ex.plot(
    ax=ax,
    # title=env_id,
    x="_runtime",
    y="return",
    err_style="band",
    std_alpha=0.1,
    rolling=50,
    n_samples=400,
    legend=False,
    tight_layout=False,
    suptitle=False,
    # colors=["#1f77b4", "#ff7f0e"],
)
ax.set_title("")
ax.xaxis.set_label_text("Runtime (minutes)")
ax.yaxis.set_label_text("")

env_id = "Pong-v5"
ex = expt.Experiment("Comparison of PPO")
wandb_runs = api.runs(
    path="openrlbenchmark/rl_games",
    filters={"$and": [{"config.params.value.config.env_config.env_name": env_id}, {"config.params.value.config.num_actors": 64}]},
)
h = create_hypothesis("rl_games' PPO + EnvPool (N=64)", wandb_runs)
ex.add_hypothesis(h)
wandb_runs = api.runs(
    path="openrlbenchmark/envpool-cleanrl",
    filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool"}, {"config.num_envs.value": 64}]},
)
h = create_hypothesis("CleanRL's PPO + EnvPool (N=64)", wandb_runs)
ex.add_hypothesis(h)
wandb_runs = api.runs(
    path="openrlbenchmark/baselines",
    filters={"$and": [{"config.env.value": env_id.replace("-v5", "NoFrameskip-v4")}, {"config.exp_name.value": "baselines-ppo2-cnn"}]},
)
h = create_hypothesis("openai/baselines' PPO (default setting)", wandb_runs)
ex.add_hypothesis(h)


ax = axes.flatten()[2]
ex.plot(
    ax=ax,
    title=env_id,
    x="global_step",
    y="return",
    err_style="band",
    std_alpha=0.1,
    rolling=50,
    n_samples=400,
    legend=False,
    tight_layout=False,
    suptitle=False,
    # colors=["#1f77b4", "#ff7f0e"],
)
ax.xaxis.set_label_text("Frames")
ax.yaxis.set_label_text("Episodic Return")
h, l = ax.get_legend_handles_labels()
ax = axes.flatten()[3]
ex.plot(
    ax=ax,
    # title=env_id,
    x="_runtime",
    y="return",
    err_style="band",
    std_alpha=0.1,
    rolling=50,
    n_samples=400,
    legend=False,
    tight_layout=False,
    suptitle=False,
    # colors=["#1f77b4", "#ff7f0e"],
)
ax.set_title("")
ax.xaxis.set_label_text("Runtime (minutes)")
ax.yaxis.set_label_text("")


fig.legend(h, l, loc='lower right', ncol=2, bbox_to_anchor=(0.75, -0.3))
plt.savefig("baselines_plot.png",  bbox_inches='tight')
plt.savefig("baselines_plot.pdf",  bbox_inches='tight')
