xvfb-run -a python benchmark.py \
    --env-ids Breakout-v5 Pong-v5 \
    --command "poetry run python ppo_atari_envpool.py --track --wandb-project-name envpool-cleanrl" \
    --num-seeds 5 \
    --workers 1

xvfb-run -a python benchmark.py \
    --env-ids BreakoutNoFrameskip-v4 PongNoFrameskip-v4 \
    --command "poetry run python ppo_atari.py --track --wandb-project-name envpool-cleanrl" \
    --num-seeds 5 \
    --workers 1

xvfb-run -a python benchmark.py \
    --env-ids Ant-v4 HalfCheetah-v4 Walker2d-v4 Humanoid-v4 \
    --command "poetry run python ppo_continuous_action_envpool.py --total-timesteps 10000000 --num-envs 64 --anneal-lr=True --clip-vloss=False --ent-coef=0 --learning-rate=0.00295 --max-grad-norm=3.5 --num-minibatches=4 --num-steps=64 --update-epochs=2 --vf-coef=1.3 --track --wandb-project-name envpool-cleanrl" \
    --num-seeds 5 \
    --workers 1

OMP_NUM_THREADS=2 python benchmark.py \
    --env-ids Ant-v2 HalfCheetah-v2 Walker2d-v2 Humanoid-v2 \
    --command "poetry run python ppo_continuous_action.py --total-timesteps 10000000 --num-envs 64 --anneal-lr=True --clip-vloss=False --ent-coef=0 --learning-rate=0.00295 --max-grad-norm=3.5 --num-minibatches=4 --num-steps=64 --update-epochs=2 --vf-coef=1.3 --track --wandb-project-name envpool-cleanrl" \
    --num-seeds 5 \
    --workers 1
