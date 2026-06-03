PROJECT=curiosity-critic-vizdoom

# Curiosity-Critic and RND: seeds 1-5
for SEED in 1 2 3 4 5; do
  for METHOD in cc rnd; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

# Curiosity V1 (special case, zero baseline): seeds 1-3
for SEED in 1 2 3; do
  python cleanrl/ppo_curiosity_critic_vizdoom.py \
    --method c_v1 --scenario sparse --noisy-tv \
    --total-timesteps 30000000 --seed $SEED \
    --capture-video --save-model \
    --track --wandb-project-name $PROJECT
done

