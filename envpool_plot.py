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

NUM_STACKED_FRAMES = 4


def create_hypothesis(name, wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()

        # data cleaning
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        wandb_run["_runtime"] /= 60  # convert to minutes
        wandb_run["global_step"] *= NUM_STACKED_FRAMES # convert to frames

        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


env_ids = [
    "Breakout-v5",
]
nrows = 1
ncols = 3
plt.rcParams.update(bundles.neurips2022())
 
plt.rcParams.update(figsizes.neurips2022(nrows=nrows, ncols=ncols, height_to_width_ratio=0.7))
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    # sharex=True,
    # sharey=True,
)

for env_id in env_ids:


    ex = expt.Experiment("Comparison of PPO")
    wandb_runs = api.runs(
        path="openrlbenchmark/envpool-cleanrl",
        filters={"$and": [{"config.env_id.value": env_id.replace("-v5", "NoFrameskip-v4")}, {"config.exp_name.value": "ppo_atari"}]},
    )
    h = create_hypothesis("CleanRL's PPO + For-loop", wandb_runs)
    ex.add_hypothesis(h)
    wandb_runs = api.runs(
        path="costa-huang/cleanrl-ppo-subprocess",
        filters={"$and": [{"config.env_id.value": env_id.replace("-v5", "NoFrameskip-v4")}, {"config.exp_name.value": "ppo_atari"}]},
    )
    h = create_hypothesis("CleanRL's PPO + Subprocess", wandb_runs)
    ex.add_hypothesis(h)
    wandb_runs = api.runs(
        path="openrlbenchmark/envpool-cleanrl",
        filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool"}]},
    )
    h = create_hypothesis("CleanRL's PPO + EnvPool (Sync)", wandb_runs)
    ex.add_hypothesis(h)


    # ex.plot(ax=g[env_id], title=env_id, y=, x="_runtime", n_samples=300, rolling=50, err_style="fill", tight_layout=False, legend=False)

    ax = axes.flatten()[1]
    ex.plot(
        ax=ax,
        # title=env_id,
        x="global_step",
        y="charts/episodic_return",
        err_style="band",
        std_alpha=0.1,
        rolling=50,
        n_samples=400,
        legend=False,
        tight_layout=False,
        suptitle=False,
        colors=["#1f77b4", "#ff7f0e", "#d62728"],
    )
    ax.set_title("")
    ax.xaxis.set_label_text("Frames")
    ax.yaxis.set_label_text("Episodic Return")
    
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
    #       ncol=1)
    h, l = ax.get_legend_handles_labels()
    # ax.legend().set_visible(False)
    
    ax = axes.flatten()[2]
    ex.plot(
        ax=ax,
        # title=env_id,
        x="_runtime",
        y="charts/episodic_return",
        err_style="band",
        std_alpha=0.1,
        rolling=50,
        n_samples=400,
        legend=False,
        tight_layout=False,
        suptitle=False,
        colors=["#1f77b4", "#ff7f0e", "#d62728"],
    )
    ax.set_title("")
    ax.xaxis.set_label_text("Runtime (minutes)")
    ax.yaxis.set_label_text("")
    

labels = ['For-loop', 'Subprocess', 'EnvPool (Sync)']
environment_time_means = np.array([1.0135670900344849, 0.3469454348087311, 0.19825412333011627])
inference_time_means = np.array([0.15798313915729523, 0.1825909912586212, 0.1951054036617279])
training_time_means = np.array([0.11033231765031816, 0.11766397207975388, 0.11970197409391405])
iteration_time_means = np.array([1.348467469215393, 0.7159662842750549, 0.585216760635376])
other_time_means = iteration_time_means - (environment_time_means + training_time_means + inference_time_means)
print(other_time_means)


ax = axes.flatten()[0]
ax.xaxis.set_label_text("Seconds per iteration")
# ax.bar(x=labels, height=environment_time_means, width=0.35, label='Environment Step Time')
# ax.bar(x=labels, height=inference_time_means, width=0.35, bottom=environment_time_means, label='Inference Time')
# ax.bar(x=labels, height=training_time_means, width=0.35, bottom=environment_time_means + inference_time_means, label='Training Time')
# ax.bar(x=labels, height=iteration_time_means, width=0.35, bottom=environment_time_means + inference_time_means + training_time_means, label='Other Time')

ax.barh(y=labels, width=environment_time_means, height=0.45, label='Environment Step Time', color="#2E4052")
ax.barh(y=labels, width=inference_time_means, height=0.45, left=environment_time_means, label='Inference Time', color="#A997DF")
ax.barh(y=labels, width=training_time_means, height=0.45, left=environment_time_means + inference_time_means, label='Training Time', color="#FFC857")
ax.barh(y=labels, width=other_time_means, height=0.45, left=environment_time_means + inference_time_means + training_time_means, label='Other Time', color="#BDD9BF")

fig.legend(h, l, loc='lower right', ncol=1, bbox_to_anchor=(0.9, -0.4))
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc='lower left', ncol=2, bbox_to_anchor=(0.07, -0.3))
plt.savefig("test.png",  bbox_inches='tight')
plt.savefig("test.pdf",  bbox_inches='tight')
