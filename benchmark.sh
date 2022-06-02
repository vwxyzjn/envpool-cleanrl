xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids pong_v3 surround_v2 tennis_v3  \
    --command "poetry run python cleanrl/ppo_pettingzoo_ma_atari.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3