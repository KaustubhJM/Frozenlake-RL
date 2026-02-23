import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Training Function
def run(episodes):

  env = gym.make('FrozenLake-v1', map_name ='4x4', is_slippery =True, render_mode =None)

# For 4x4 map: 16 states, 4 actions
  q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 16x4 array

# Hyperparameters
  learning_rate_a = 0.1
  discount_factor_g = 0.99

  epsilon = 1.0
  epsilon_decay_rate = 0.995
  epsilon_min = 0.05
  rng = np.random.default_rng()

  rewards_per_episode = np.zeros(episodes)
  success_count = 0

# Episode loop
  for i in range(episodes):

    total_reward = 0
    state = env.reset()[0] # states: 0 to 15, 0=top left corner, 63=bottom right corner
    terminated = False # true when falls in hole or reached goal
    truncated = False # true when actions > 1000

    # greedy action selection
    while(not terminated and not truncated):
        if rng.random() < epsilon:
            action = env.action_space.sample() # action: 0=left, 1=down, 2=right, 3=up
        else:
            action = np.argmax(q[state,:])  

        new_state,reward,terminated,truncated,_= env.step(action)
        total_reward += reward

        # Updating
        q[state,action]= q[state,action] + learning_rate_a*(reward + discount_factor_g*np.max(q[new_state,:]) - q[state,action])
        state = new_state

    # Decay Exploration Rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay_rate) 

    
    rewards_per_episode[i] = total_reward

    if total_reward == 1:
        success_count += 1  

  print("Total successes during training:", success_count)
  env.close()  
  return q, rewards_per_episode

def evaluate_policy(q, episodes=100):
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)

    total_reward = 0

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q[state, :])   # Greedy policy
            state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

    env.close()
    return total_reward / episodes

def evaluate_random(episodes=100):
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)

    total_reward = 0

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample() # Random action
            state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

    env.close()
    return total_reward / episodes


def visualize(q):
    import time

    env = gym.make(
        'FrozenLake-v1',
        map_name='4x4',
        is_slippery=True,
        render_mode='human'
    )

    state, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = np.argmax(q[state,:])
        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.5)

    time.sleep(2)
    env.close()

if __name__ == "__main__":
    q, rewards = run(1000)

    # Evaluate trained agent
    agent_avg = evaluate_policy(q, episodes=100)
    random_avg = evaluate_random(episodes=100)

    print("Agent Average Reward (100 episodes):", agent_avg)
    print("Random Policy Average Reward (100 episodes):", random_avg)

    # Rolling Average for smoother learning curve
    window_size = 100
    rolling_avg = np.convolve(
        rewards,
        np.ones(window_size) / window_size,
        mode='valid'
    )

    visualize(q)

    # Learning Curve
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3, label="Raw Reward")
    plt.plot(rolling_avg, linewidth=2, label="Smoothed (100 ep avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.show()


print("Max Q value:", np.max(q))
print(np.argmax(q, axis=1).reshape(4,4))