from random import random

n = 0
m = 0

while True:
    for k in range(10**6):
        x = random()
        y = random()
        d2 = x**2 + y**2
        if d2 < 1.0:
            m += 1
        n += 1
    print(f"n={n} m={m} pi={4*m/n}")
