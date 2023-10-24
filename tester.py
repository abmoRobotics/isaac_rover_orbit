import gym
import custom_envs
# List all registered environments
env_ids = [spec.id for spec in gym.envs.registry.all()]
for env_id in sorted(env_ids):
    print(env_id)