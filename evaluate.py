from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from arm_reach_env import ArmReachEnv
import numpy as np

# Create environment
def make_env():
    return ArmReachEnv(render=True)

env = DummyVecEnv([make_env])

# Load normalization statistics
try:
    env = VecNormalize.load("models/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    print("Loaded normalization statistics")
except FileNotFoundError:
    print("Warning: No normalization stats found, proceeding without normalization")

# Load model
try:
    model = PPO.load("models/ppo_arm_reach_final", env=env)
    print("Loaded final model")
except FileNotFoundError:
    try:
        model = PPO.load("models/best_model/best_model", env=env)
        print("Loaded best model")
    except FileNotFoundError:
        model = PPO.load("ppo_arm_reach", env=env)
        print("Loaded basic model")

# Evaluation statistics
num_episodes = 50
episode_rewards = []
episode_lengths = []
success_count = 0

print(f"\nEvaluating for {num_episodes} episodes...")

obs = env.reset()

for episode in range(num_episodes):
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        # Check if episode ended
        if done[0]:
            # Check for success (distance < 0.05)
            if "distance" in info[0] and info[0]["distance"] < 0.05:
                success_count += 1
                print(f"Episode {episode + 1}: SUCCESS! Reward: {episode_reward:.2f}, Length: {episode_length}")
            else:
                print(f"Episode {episode + 1}: Failed. Reward: {episode_reward:.2f}, Length: {episode_length}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            obs = env.reset()
            break

# Print statistics
print("\n" + "="*50)
print("EVALUATION STATISTICS")
print("="*50)
print(f"Episodes: {num_episodes}")
print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
print(f"Best Reward: {np.max(episode_rewards):.2f}")
print(f"Worst Reward: {np.min(episode_rewards):.2f}")
print("="*50)

env.close()