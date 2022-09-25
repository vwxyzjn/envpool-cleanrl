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

        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


env_ids = ["HalfCheetah-v3"]
nrows = 1
ncols = 2
plt.rcParams.update(bundles.neurips2022())
 
plt.rcParams.update(figsizes.neurips2022(rel_width=0.6, nrows=nrows, ncols=ncols, height_to_width_ratio=0.9))
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    # sharex=True,
    # sharey=True,
)
for env_idx, env_id in enumerate(env_ids):
    ex = expt.Experiment("Comparison of PPO")
    wandb_runs = api.runs(
        path="openrlbenchmark/acme",
        filters={"$and": [{"config.env_name.value": env_id}, {"config.num_envs.value": 32}, {"config.use_envpool.value": True}]},
    )
    h = create_hypothesis("ACME's PPO + EnvPool", wandb_runs)
    ex.add_hypothesis(h)
    wandb_runs = api.runs(
        path="openrlbenchmark/acme",
        filters={"$and": [{"config.env_name.value": env_id}, {"config.num_envs.value": 32}, {"config.use_vec_env.value": True}]},
    )
    h = create_hypothesis("ACME's PPO + vecenv", wandb_runs)
    ex.add_hypothesis(h)


    ax = axes.flatten()[env_idx*2]
    ex.plot(
        ax=ax,
        title=env_id,
        x="global_step",
        y="episode_return",
        err_style="band",
        std_alpha=0.1,
        rolling=50,
        n_samples=400,
        legend=False,
        tight_layout=False,
        suptitle=False,
        colors=["#1f77b4", "#ff7f0e"],
    )
    # ax.set_title("")
    ax.xaxis.set_label_text("Steps")
    if env_idx == 0:
        ax.yaxis.set_label_text("Episodic Return")
    else:
        ax.yaxis.set_label_text("")

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
    #       ncol=1)
    h, l = ax.get_legend_handles_labels()
    # ax.legend().set_visible(False)

    ax = axes.flatten()[env_idx*2+1]
    ex.plot(
        ax=ax,
        # title=env_id,
        x="_runtime",
        y="episode_return",
        err_style="band",
        std_alpha=0.1,
        rolling=50,
        n_samples=400,
        legend=False,
        tight_layout=False,
        suptitle=False,
        colors=["#1f77b4", "#ff7f0e"],
    )
    ax.set_title("")
    ax.xaxis.set_label_text("Runtime (minutes)")
    ax.yaxis.set_label_text("")


fig.legend(h, l, loc='lower right', ncol=2, bbox_to_anchor=(0.95, -0.16))
plt.savefig("acme_plot.png",  bbox_inches='tight')
plt.savefig("acme_plot.pdf",  bbox_inches='tight')
