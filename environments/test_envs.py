import gymnasium as gym

def test_environment(env_name):
    print(f"\nTesting: {env_name}")
    env = gym.make(env_name)
    obs, info = env.reset()

    print(f"  Observation shape : {env.observation_space.shape}")
    print(f"  Number of actions : {env.action_space.n}")

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"  OK - {env_name} works!")

test_environment("CartPole-v1")
test_environment("LunarLander-v3")
 