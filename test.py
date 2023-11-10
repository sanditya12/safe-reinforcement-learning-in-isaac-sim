from env import RacingEnv
from stable_baselines3 import PPO

identifier = "8-11-23"
policy_path = "./cnn_policies/" + identifier + "/jetbot_policy"

policy_path = "./cnn_policies/" + identifier + "/jetbot_policy_checkpoint_800000_steps"

#isaac_python -m tensorboard.main --logdir=./cnn_policies/4-11-23/PPO_1


env = RacingEnv(headless=False, safety_filter=False, identifier = identifier)
model = PPO.load(policy_path)

for _ in range(1):
    obs = env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation = obs, deterministic = True)
        obs, reward, done, info = env.step(actions)
        env.render()

env.close()