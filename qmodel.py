import numpy as np
from scipy.spatial import distance_matrix


def epsilon_greedy_policy(epsilon, state, q_values, options):
    valid = np.array(options)

    if np.random.random() > epsilon:
        max_value = np.max(q_values[state][valid])
        max_actions = np.intersect1d(valid, np.where(q_values[state] == max_value))
        action = np.random.choice(max_actions)
    else:
        action = np.random.choice(valid)

    return action


def update_qvalues(q_values, distances, state, action, alpha, gamma):
    next_value = q_values[action].max()
    reward = -distances[state, action]

    q_values[state, action] *= 1 - alpha
    q_values[state, action] += alpha * (reward + gamma * next_value)

    if state != 0:
        next_value = q_values[0].max()
        zreward = -distances[state, 0]
        q_values[state, 0] *= 1 - (alpha / 100)
        q_values[state, 0] += (alpha / 100) * (zreward + gamma * next_value)

    return q_values, reward


class QModel:
    def __init__(self, points):
        np.random.seed(42)
        self.points = points
        self.n = len(points)
        self.distances = distance_matrix(self.points, self.points)

    def learn(self, distances, epochs=100, epsilon0=1.0, alpha0=0.1, gamma=0.97, decay=0.0):
        rewards = []
        q_values = np.zeros([self.n, self.n])
        q_values[range(self.n), range(self.n)] = -np.inf

        for i in range(epochs):
            total_reward = 0

            state = 0
            path = [state]
            options = list(range(self.n))

            alpha = alpha0 / (1 + i * decay)
            epsilon = epsilon0 / (1 + i * decay)

            while len(options) > 1:
                options.remove(state)

                action = epsilon_greedy_policy(epsilon, state, q_values, options)
                q_values, reward = update_qvalues(
                    q_values, distances, state, action, alpha, gamma
                )
                total_reward += reward
                path.append(action)
                state = action

            # back to start
            action = 0
            q_values, reward = update_qvalues(
                q_values, distances, state, action, alpha, gamma
            )
            total_reward += reward
            path.append(action)
            rewards.append(total_reward)

            if i % 200 == 0:
                print("reward", reward)

        return q_values

    def solve(self):
        q_values = self.learn(self.distances, epochs=400, epsilon0=1, gamma=-1, alpha0=1, decay=0.05)

        state = 0
        path = [state]
        options = list(range(self.n))
        distance = 0

        while len(options) > 1:
            options.remove(state)
            action = epsilon_greedy_policy(0, state, q_values, options)
            distance += self.distances[state, action]
            path.append(action)

            state = action

        path.append(0)
        distance += self.distances[state, 0]

        return {
            "ordered_points": np.array(self.points)[path].tolist(),
            "distance": distance,
        }
