from simulator.math.rotation import quat2euler, euler2quat, rot_matrix_zyx, rot_matrix_quat

import numpy as np

e1 = np.deg2rad([76, 23, 270])
q1 = euler2quat(e1)
nq1 = np.linalg.norm(q1)
e2 = quat2euler(q1)

rot1 = rot_matrix_zyx(e1)
rot2 = rot_matrix_quat(q1)

print(rot1)
print(rot2)
print((rot1-rot2)**2)

print("end")