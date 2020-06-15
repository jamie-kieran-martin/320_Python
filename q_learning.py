Q = lambda new, old, r, gamma, actions = []: (1 - new) * old + new * (r + gamma * max(actions))

if __name__ == "__main__":
    print(Q(0.2,0.6, 0, 0.9, [0.4, 0.8, 1.2]))