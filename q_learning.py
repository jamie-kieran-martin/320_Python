Q_table = [
    [0, 1,   2,   3],
    [1, 0.1, 0.5, 0.9],
    [2, 0.2, 0.6, 1.0],
    [3, 0.3, 0.7, 1.1],
    [4, 0.4, 0.8, 1.2]
]

# Q_table is flipped so that we can use array slices [1:]

# set external values here
learning_rate = 0.2
discount = 0.9
reward = 1
sn = 2
# sn = new state

# s = old state, a = action
Q = lambda s, a: (1 - learning_rate) * Q_table[s][a] + learning_rate * (
            reward + discount * max(Q_table[sn][1:]))

if __name__ == "__main__":
    print(Q(3, 1))
