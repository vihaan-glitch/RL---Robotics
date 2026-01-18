from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from arm_reach_env import ArmReachEnv
import os

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create training environment
def make_env():
    return ArmReachEnv(render=False)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Create evaluation environment
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_arm_reach"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Check if tensorboard is available
try:
    import tensorboard
    tensorboard_log = "./logs/tensorboard/"
    print("TensorBoard logging enabled")
except ImportError:
    tensorboard_log = None
    print("TensorBoard not installed - proceeding without TensorBoard logging")

# Create PPO model with improved hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Encourage exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=tensorboard_log
)

print("\nStarting training...")
print(f"Total timesteps: 500,000")
print(f"Environment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")

model.learn(
    total_timesteps=500_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=False  # Set to False if tqdm/rich not installed
)

# Save final model
model.save("models/ppo_arm_reach_final")
env.save("models/vec_normalize.pkl")

print("\nTraining completed!")
print("Model saved to: models/ppo_arm_reach_final.zip")
print("Normalization stats saved to: models/vec_normalize.pkl")

env.close()
eval_env.close()