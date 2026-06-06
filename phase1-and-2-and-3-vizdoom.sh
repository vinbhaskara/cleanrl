PROJECT=curiosity-critic-vizdoom-JUN2026

# Headline noisy-TV (full static, alpha=1 default): CC, RND, C_V2 -- seeds 1-3
for SEED in 1 2 3; do
  for METHOD in cc rnd c_v2; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

PROJECT=curiosity-critic-vizdoom-JUN2026

for SEED in 1 2 3; do
  for METHOD in c_v1 ppo random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse --noisy-tv \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done


PROJECT=curiosity-critic-vizdoom-JUN2026

for SEED in 1 2 3; do
  for METHOD in cc rnd; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

for SEED in 1 2 3; do
  for METHOD in c_v1 c_v2 ppo random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario sparse \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done




