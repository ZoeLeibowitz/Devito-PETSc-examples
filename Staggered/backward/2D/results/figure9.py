import numpy as np
import matplotlib.pyplot as plt
import os, re

# load data
results_dir = os.path.dirname(os.path.abspath(__file__))
user_re, user_xr = [], []

for folder in sorted(os.listdir(results_dir)):
    path = os.path.join(results_dir, folder)
    if not os.path.isdir(path):
        continue
    for fname in os.listdir(path):
        if fname.startswith('reattachment') and fname.endswith('.txt'):
            with open(os.path.join(path, fname)) as f:
                lines = f.readlines()
            # Parse Re_code from first line e.g. "Re=150, grid=601x41"
            m_re = re.search(r'Re=(\d+)', lines[0])
            # Parse x_r/h from second line
            m_xr = re.search(r'x_r/h\s*=\s*([\d.]+)', lines[1])
            if m_re and m_xr:
                re_code = int(m_re.group(1))
                xr = float(m_xr.group(1))
                re_armaly = 2 * re_code   # Re_armaly = 2 * Re_code
                user_re.append(re_armaly)
                user_xr.append(xr)

user_re, user_xr = zip(*sorted(zip(user_re, user_xr)))
print('User data (Re_armaly, x_r/h):')
for r, x in zip(user_re, user_xr):
    print(f'  Re={r}, x_r/h={x:.4f}')

# Reference data
# Armaly et al. 2D computation (dashed)
armaly_2d_data = np.loadtxt(os.path.join(results_dir, 'armaly_computation.csv'), delimiter=',')
armaly_2d_data = armaly_2d_data[armaly_2d_data[:, 0].argsort()]
armaly_2d_re, armaly_2d_xr = armaly_2d_data[:, 0], armaly_2d_data[:, 1]

# Kim & Moin computed results
km_data = np.loadtxt(os.path.join(results_dir, 'kim_moin_results.csv'), delimiter=',')
km_re, km_xr = km_data[:, 0], km_data[:, 1]

# Armaly experimental data (circles)
armaly_num_data = np.loadtxt(os.path.join(results_dir, 'armaly_experimental.csv'), delimiter=',')
armaly_num_re, armaly_num_xr = armaly_num_data[:, 0], armaly_num_data[:, 1]


fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(armaly_num_re, armaly_num_xr, marker='o', s=40, facecolors='none',
           edgecolors='k', linewidths=1.0, label='Armaly et al. (experimental)')
ax.plot(armaly_2d_re, armaly_2d_xr, 'k--', linewidth=1.2, label='Armaly et al. (computational)')
ax.plot(km_re, km_xr, 'k-', linewidth=1.5, label='Kim & Moin')
ax.plot(user_re, user_xr, 'b-o', linewidth=1.5, markersize=5, label='Present results')


ax.set_xlabel('Re')
ax.set_ylabel(r'$x_r/h$')
ax.set_xlim(0, 850)
ax.set_ylim(0, 15)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Reattachment length as a function of Reynolds number')

plt.tight_layout()
plt.savefig('figure9.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved figure9.png')
