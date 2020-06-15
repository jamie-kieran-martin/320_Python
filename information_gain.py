from math import log2

H = lambda p, n: (-p / (p + n)) * log2(p / (p + n)) + (-n / (p + n)) * log2(n / (p + n))

if __name__ == "__main__":
    print(((3 / 6) * H(2, 1) + (3 / 6) * H(1, 2)))
