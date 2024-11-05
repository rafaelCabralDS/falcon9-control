import pygame
from environment.env import BoosterEnv
import numpy as np
from common.yaml_helper import load_config
import os
import time

simulation_env_configs = dict(
    drag=10_000,
    turbulence=0.0,
    wind=0.0,
    max_steps=None,
    render=True,
    initial_condition=dict( # mean value +- variance
        Vx="0.000001 +- 0.0",
        Vy="0.000001 +- 0.0", # do not set linear velocity to zero!!
        alpha="0.0 +- 0.0",
        fuel_ratio="0.5 +- 0.0",
        w="0.0 +- 0.0",
        x="2500.0 +- 0.0",
        y="1200.0 +- 0.0",
    )
)




# Initialize Pygame
#pygame.init()




# Action mapping
action_mapping = {
    pygame.K_UP: (0.5, 0, 0),  # Throttle increase
    pygame.K_DOWN: (-0.5, 0, 0),  # Throttle decrease
    pygame.K_LEFT: (0, -0.5, 0),  # Left side engine
    pygame.K_RIGHT: (0, 0.5, 0),  # Right side engine
    pygame.K_a: (0, 0, -1),  # Gimbal angle left
    pygame.K_d: (0, 0, 1),  # Gimbal angle right
    pygame.K_ESCAPE: 'quit'  # Quit
}

if __name__ == "__main__":

    # Load friendly playground conditions
    env_config = load_config('env.yaml')
    simulation_env_configs['reward'] = env_config['reward']
    simulation_env_configs['reward_version'] = env_config['reward_version']

    env = BoosterEnv(config=simulation_env_configs)
    env.reset()
    done = False

    while not done:
        action = [0,0,0]
        s, r, done, truncated, obs = env.step(action)
        #time.sleep(0.3)

    print(obs['termination_cause'], f'after {obs["t"]} flight seconds')


    """
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            print('aa')
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in action_mapping:
                    action = action_mapping[event.key]

                    # Handle quit action
                    if action == 'quit':
                        running = False
                        continue

                    # Convert action to proper format
                    main_engine_power = max(-1, min(1, action[0]))  # Clamping
                    side_engine_power = max(-1, min(1, action[1]))  # Clamping
                    gimbal_angle = max(-1, min(1, action[2]))  # Clamping

                    # Create the action
                    env_action = np.array([main_engine_power, side_engine_power, gimbal_angle])

                    # Step the environment with the action
                    state, reward, done, info, obs = env.step(env_action)
                    print(state)


                    # Print the current state and reward
                    print("State:", state)
                    print("Reward:", reward)

                    if done:
                        print("Episode finished!")
                        env.reset()

    # Cleanup
    env.close()
    pygame.quit()
    """
