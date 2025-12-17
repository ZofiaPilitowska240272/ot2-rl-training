import numpy as np
from ot2_gym_wrapper import OT2GymEnv  # Replace with your actual filename

env = OT2GymEnv(success_threshold=0.002, max_steps=1000, record_trajectory=True)

# Reset environment
obs, info = env.reset()
print("Initial observation:", obs)
print("Target position:", info['target_position'])
print("Initial distance to target (mm):", info['initial_distance']*1000)

done = False
step = 0

# Proportional gain for simple control
Kp = 5.0

while not done:
    # Simple proportional controller: move towards target
    current_pos = obs[:3]
    target_pos = obs[3:6]
    
    # Compute direction
    direction = target_pos - current_pos
    norm = np.linalg.norm(direction)
    if norm > 0:
        action = (direction / norm).astype(np.float32)  # unit vector
    else:
        action = np.zeros_like(direction)
    action = np.clip(action, -1.0, 1.0)

    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1

    if step % 20 == 0:
        print(f"Step {step}: Distance = {info['distance']*1000:.3f} mm, Reward = {reward:.2f}")

print("Episode finished.")
print("Steps:", step)
print("Success:", info['success'])
print("Final pipette position:", info['position'])
print("Final distance to target (mm):", info['distance']*1000)
print("Cumulative reward:", info['cumulative_reward'])
print("Recorded trajectory length:", len(info['trajectory']))
print("First trajectory entry:", info['trajectory'][0])
print("Last trajectory entry:", info['trajectory'][-1])

env.close()