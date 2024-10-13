from simulator.math.lti_systems import *

num = [1, 5]
den = [2, -3, 1]
print(normalize_tf(num, den))

# A, B, C, D = tf2ccf(num, den)
# print(A)
# print(B)
# print(C)
# print(D)

zeros, poles, gain = tf2zpk(num, den)
print(zeros)
print(poles)
print(gain)

zpk2tf(zeros, poles, gain)