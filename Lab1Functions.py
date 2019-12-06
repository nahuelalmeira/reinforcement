import numpy as np

def choose_action(q, state, actions, epsilon, random_state):
    """
    Chooses an action according to the learning previously performed 
    using an epsilon-greedy exploration policy
    """

    q_values = [q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)

    if random_state.uniform() < epsilon:
        return random_state.choice(actions)  # a random action is selected

    count = q_values.count(max_q)

    # In case there're several state-action max values
    # we select a random one among them
    if count > 1:
        best = [i for i in range(len(actions)) if q_values[i] == max_q]
        i = random_state.choice(best)
    else:
        i = q_values.index(max_q)

    return actions[i]


def SARSA_learn(q, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Performs a SARSA update for a given state transition
    """
    
    current_q = q.get((state, action), 0.0)
    next_q = q.get((next_state, next_action), 0.0)
    
    q[(state, action)] = current_q  + alpha * (reward + gamma*next_q - current_q)

    return q


def SARSA_run(q, env, episodes_to_run, alpha, gamma, epsilon, random_state=None):
    """
    Runs the reinforcement learning agent with a given configuration.
    """

    if not random_state:
        random_state = np.random.RandomState(42)

    actions = range(env.action_space.n)

    # list that contains the amount of time-steps of the episode. It is used as a way to score the performance of
    # the agent.
    timesteps_of_episode = []
    # list that contains the amount of reward given to the agent in each episode
    reward_of_episode = []

    for _ in range(episodes_to_run):
        # an instance of an episode is run until it fails or until it reaches 200 time-steps

        # resets the environment, obtaining the first state observation
        state = env.reset()

        episode_reward = 0
        done = False
        t = 0

        # Pick an action based on the current state
        action = choose_action(q, state, actions, epsilon, random_state)
        
        while not done:        
            
            # Execute the action and get feedback
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            next_action = choose_action(q, state, actions, epsilon, random_state)

            if not done and t < 2000:  # if the algorithm does not converge, it stops after 2000 timesteps
                q = SARSA_learn(q, state, action, reward, next_state, next_action, alpha, gamma)
                state = next_state
                action = next_action
            else:
                done = True  # done is ensured to be True (in the case the algorithm did not reach convergence)
                q = SARSA_learn(q, state, action, reward, next_state, next_action, alpha, gamma)
                timesteps_of_episode = np.append(timesteps_of_episode, [int(t + 1)])
                reward_of_episode = np.append(reward_of_episode, max(episode_reward, -100))
            t += 1

    return reward_of_episode.mean(), timesteps_of_episode, reward_of_episode, q


