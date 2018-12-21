import random
import gym
import numpy as np


def show_stats(func):
    best_current_policy = []
    best_current_func = []
    second_best_policy = []
    second_best_func = []
    for state in func:
        action = np.argmax(state)
        best_current_policy.append(action)
        best_current_func.append(str("%.3f" % state[action]))

        temp = np.copy(state)
        temp = np.delete(temp, action)

        action = np.argmax(temp)
        second_best_policy.append(action)

        for i in range(len(state)):
            if state[i] == temp[action]:
                second_best_func.append(str("%.3f" % state[i]))
                break

    return \
        "Best Current Policy:\t\t\t" + str(best_current_policy) + "\n" + \
        "Second Best Current Policy:\t\t" + str(second_best_policy) + "\n" + \
        "Best Current Function:\t\t\t" + str(best_current_func) + "\n" + \
        "Second Best Current Function:\t" + str(second_best_func)


def get_dictionary_key(state, action):
    return str(state) + ", " + str(action)


def store_state_action(state, action, _memory):
    key = get_dictionary_key(state, action)

    if key in _memory:
        _memory[key] += 1
    else:
        _memory[key] = 1

    return _memory


def playthrough(env, _value_function, _memory, epsilon):
    state = env.reset()
    is_done = False
    num_moves = 0
    max_moves = 100
    local_memory = {}
    reward = 0

    # play the game!
    while is_done is False and num_moves < max_moves:
        # If random in epsilon, take random action
        # TODO: be certain random number is not max arg
        if np.random.rand() < epsilon:
            action = random.randint(0, env.nA - 1)
        else:
            # action should be selected randomly between indices of largest values
            maxi = np.max(_value_function[state])
            action_array = []
            for a in range(_value_function[state].size):
                if _value_function[state][a] == maxi:
                    action_array.append(a)
            action = action_array[random.randint(0, len(action_array) - 1)]

        # store value for control method
        _memory = store_state_action(state, action, _memory)
        local_memory = store_state_action(state, action, local_memory)

        # take a move and get its variables
        next_state, reward, is_done, prob = env.step(action)

        # Update state and proceed to next move
        state = next_state

    # update the value function based on the new value at the end of the playthrough
    for state in range(env.nS):
        for action in range(env.nA):
            key = get_dictionary_key(state, action)
            if key in local_memory:
                _value_function[state][action] += (reward - _value_function[state][action]) / _memory[key]

    # return the value function, and play again!
    goal = 0
    if reward > 0:
        goal = 1

    return _value_function, _memory, epsilon, goal


def run_mc_glie(env, func, playthroughs):
    global_memory = {}
    thousand_plays = 0
    win_array = []
    epsilon = 0.1

    for i in range(playthroughs):
        func, global_memory, epsilon, goal = playthrough(env, func, global_memory, epsilon)
        win_array.append(goal)
        if len(win_array) > 100:
            win_array.pop(0)
        if (i + 1)//1000 > thousand_plays:
            thousand_plays += 1
            print("Playthrough: " + str(i + 1))
            print(show_stats(func) + "\n")

    optimal_policy = []

    for s in func:
        action = np.argmax(s)
        optimal_policy.append(action)

    return optimal_policy


if __name__ == "__main__":
    environment = gym.make("FrozenLake-v0").env
    desired_playthroughs = 100000

    # initialize null value function
    value_function = np.zeros((environment.nS, environment.nA))

    # Solve the game!
    policy = run_mc_glie(environment, value_function, desired_playthroughs)

    plays = 1000
    wins = 0

    for i in range(plays):
        state = environment.reset()
        is_done = False
        while is_done is False:
            state, r, is_done, p = environment.step(policy[state])

        if r > 0:
            wins += 1

    print("Winrate: " + str(wins / plays))

"""
A good decision cannot guarantee a good outcome. All real decisions are made under uncertainty. A decision
is, therefore, a bet. Evaluating it as good or not must depend on the states and the odds, not on the outcome.
        - Ward Edwards -
"""
