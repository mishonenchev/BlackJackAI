from __future__ import annotations

from collections import defaultdict
import dill
import os.path
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

import gym

class BlackjackAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        q_values = None
    ):
        self.env = env

        if(q_values == None):
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        else:
            self.q_values = q_values
        
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2) 
final_epsilon = 0.1
aiAgent = None

def create_grids(agent, usable_ace=False):
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)
    
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stand"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

def play_games(num: int):
    if os.path.isfile("q_values.pkl"):
        with open("q_values.pkl", "rb") as f:
            q_values = dill.load(f)
        
        envP = gym.make("Blackjack-v1", sab=True, render_mode='human')
        envP = gym.wrappers.RecordEpisodeStatistics(envP, deque_size=n_episodes)

        agentP = BlackjackAgent(
            env=envP,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            q_values=q_values
        )
        
        for i in range(num):
            obs, info = envP.reset()
            envP.render()
            time.sleep(5)
            done = False
            while not done:
                action = agentP.get_action(obs)
                next_obs, reward, terminated, truncated, info = envP.step(action)

                done = terminated or truncated
                obs = next_obs
                
                envP.render()
                time.sleep(5) # Add a delay of 5 second   
        envP.close() 
    else:
        print("File 'q_values.pkl' does not exist. Train AI first!")

def play_test_games(num: int):
    q_values = None
    if os.path.isfile("q_values.pkl"):
        with open("q_values.pkl", "rb") as f:
            q_values = dill.load(f)
    envPB = gym.make("Blackjack-v1", sab=True)
    envPB = gym.wrappers.RecordEpisodeStatistics(envPB, deque_size=n_episodes)

    agentPB = BlackjackAgent(
            env=envPB,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            q_values=q_values
    )
    timesWon = 0
    timesLost = 0
    timesDraw = 0
    for i in range(num):
        obs, info = envPB.reset()
        done = False
        while not done:
            action = agentPB.get_action(obs)
            next_obs, reward, terminated, truncated, info = envPB.step(action)

            done = terminated or truncated
            obs = next_obs
        
        if reward == 0:
            timesDraw = timesDraw+1
        elif reward == 1:
            timesWon = timesWon +1
        else:
            timesLost =timesLost+1
    print(f"Wins: {timesWon}, Lost: {timesLost}, Draw: {timesDraw}")


while True:
    print("Choose: Train[T], Statistics[S], Play game simulator[P], Bulk Test[B]")
    user_input = input()
    if user_input == "S":
        if(aiAgent != None):
            # state values & policy with usable ace (ace counts as 11)
            value_grid, policy_grid = create_grids(aiAgent, usable_ace=True)
            fig1 = create_plots(value_grid, policy_grid, title="With usable ace - Ace can be 11")
            # state values & policy without usable ace (ace counts as 1)
            value_grid, policy_grid = create_grids(aiAgent, usable_ace=False)
            fig2 = create_plots(value_grid, policy_grid, title="Without usable ace - Ace can be only 1")
            # Display figures
            plt.show()
        else:
            print("Can't display statistics. Train AI first!")
    elif user_input == "P":
        play_games(15)
    elif user_input == "B":
        play_test_games(1000)
    elif user_input == "T":
        env = gym.make("Blackjack-v1", sab=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

        agent = BlackjackAgent(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )

        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()

        with open("q_values.pkl", "wb") as f:
            dill.dump(agent.q_values, f)
        aiAgent = agent
        print("Training has finished")
        
    elif input == "exit":
        env.close()