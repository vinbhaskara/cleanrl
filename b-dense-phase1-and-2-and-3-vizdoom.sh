PROJECT=curiosity-critic-vizdoom-JUN2026-Final2

# core dense column: ours + the two contested baselines (~2 days)
for SEED in 1 2 3; do
  for METHOD in cc rnd ppo; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario dense \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done

# optional fill-in for the complete 6-method grid (run only if time permits)
for SEED in 1 2 3; do
  for METHOD in c_v1 c_v2 random; do
    python cleanrl/ppo_curiosity_critic_vizdoom.py \
      --method $METHOD --scenario dense \
      --total-timesteps 30000000 --seed $SEED \
      --capture-video --save-model \
      --track --wandb-project-name $PROJECT
  done
done


