"""
Agent-based swarm coverage simulation for Project Sycamore.

Models N Sycamore UAVs deployed over an active wildfire perimeter.
Each agent carries a thermal/optical sensor with a circular footprint.
Agents maintain communication via a range-constrained mesh network.

Metrics reported:
  - Perimeter coverage fraction (% of perimeter within any UAV's sensor cone)
  - Network connectivity (is the swarm graph connected?)
  - Coverage density map

Run with: ~/ds/bin/python swarm_coverage.py
Outputs : output/swarm_snapshot.png, output/swarm_coverage_vs_n.png
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('output', exist_ok=True)

RNG = np.random.default_rng(42)

# ── UAV sensor + comms parameters ─────────────────────────────────────────
R_SENSE  = 80.0    # sensor footprint radius [m] (thermal/optical at AGL)
R_COMM   = 500.0   # max inter-UAV communication range [m]
AGL      = 150.0   # nominal operating altitude AGL [m]

# ── Fire perimeter model (ellipse) ────────────────────────────────────────
FIRE_A = 800.0   # semi-major axis along wind [m]
FIRE_B = 400.0   # semi-minor axis cross-wind [m]
N_PERIM = 400    # points on perimeter for coverage check

phi_p = np.linspace(0, 2*np.pi, N_PERIM, endpoint=False)
perim_x = FIRE_A * np.cos(phi_p)
perim_y = FIRE_B * np.sin(phi_p)
perim   = np.column_stack([perim_x, perim_y])

def deploy_uniform(n):
    """Deploy n UAVs uniformly along the fire perimeter with jitter."""
    idx = np.linspace(0, N_PERIM, n, endpoint=False).astype(int)
    pts = perim[idx].copy()
    jitter = RNG.uniform(-R_SENSE * 0.5, R_SENSE * 0.5, pts.shape)
    return pts + jitter

def coverage_fraction(agents):
    """Fraction of perimeter points within R_SENSE of any agent."""
    dists = np.min(
        np.linalg.norm(perim[:, None, :] - agents[None, :, :], axis=2),
        axis=1
    )
    return np.mean(dists < R_SENSE)

def build_comm_graph(agents):
    """Return networkx Graph of agents within R_COMM of each other."""
    G = nx.Graph()
    G.add_nodes_from(range(len(agents)))
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            d = np.linalg.norm(agents[i] - agents[j])
            if d <= R_COMM:
                G.add_edge(i, j, weight=d)
    return G

# ── Snapshot with N=20 UAVs ───────────────────────────────────────────────
N_UAV  = 20
agents = deploy_uniform(N_UAV)
G      = build_comm_graph(agents)
cov    = coverage_fraction(agents)
is_connected = nx.is_connected(G)

print(f"\n{'─'*50}")
print(f"  Swarm simulation  |  N={N_UAV} UAVs")
print(f"  Fire ellipse      |  {FIRE_A*2:.0f}m × {FIRE_B*2:.0f}m")
print(f"  Sensor radius     |  {R_SENSE:.0f} m")
print(f"  Comm range        |  {R_COMM:.0f} m")
print(f"  Coverage          |  {cov*100:.1f}%")
print(f"  Network connected |  {is_connected}")
print(f"  Components        |  {nx.number_connected_components(G)}")
print(f"{'─'*50}\n")

fig, ax = plt.subplots(figsize=(9, 6))
# Fire perimeter
ax.plot(perim_x, perim_y, 'r-', lw=1.5, label='Fire perimeter')
ax.fill(perim_x, perim_y, color='lightsalmon', alpha=0.25)
# Comm edges
for u, v in G.edges():
    ax.plot([agents[u,0], agents[v,0]], [agents[u,1], agents[v,1]],
            'k-', lw=0.5, alpha=0.4)
# Sensor footprints
for i, (ax_, ay_) in enumerate(agents):
    circle = plt.Circle((ax_, ay_), R_SENSE, color='steelblue', alpha=0.20)
    ax.add_patch(circle)
# Agent positions
sc = ax.scatter(agents[:,0], agents[:,1], s=60, zorder=5,
                c=['steelblue' if G.degree(i) > 0 else 'crimson'
                   for i in range(N_UAV)],
                edgecolors='white', linewidths=0.8)
ax.set_aspect('equal')
ax.set(xlabel='Easting [m]', ylabel='Northing [m]',
       title=f'Sycamore Swarm  —  N={N_UAV} UAVs, coverage={cov*100:.1f}%, '
             f'{"connected" if is_connected else "DISCONNECTED"}')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/swarm_snapshot.png', dpi=150, bbox_inches='tight')
print('Saved: output/swarm_snapshot.png')

# ── Coverage vs fleet size ────────────────────────────────────────────────
fleet_sizes = np.arange(5, 55, 5)
cov_mean, cov_std, connected_frac = [], [], []

N_TRIALS = 8   # Monte Carlo trials per fleet size
for n in fleet_sizes:
    c_trials, conn_trials = [], []
    for _ in range(N_TRIALS):
        ag = deploy_uniform(n)
        c_trials.append(coverage_fraction(ag))
        conn_trials.append(float(nx.is_connected(build_comm_graph(ag))))
    cov_mean.append(np.mean(c_trials))
    cov_std.append(np.std(c_trials))
    connected_frac.append(np.mean(conn_trials))

cov_mean = np.array(cov_mean)
cov_std  = np.array(cov_std)
conn_frac = np.array(connected_frac)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Sycamore Swarm  —  Coverage vs Fleet Size', fontsize=11)

axes2[0].plot(fleet_sizes, cov_mean*100, 'steelblue', lw=2)
axes2[0].fill_between(fleet_sizes,
                      (cov_mean - cov_std)*100,
                      (cov_mean + cov_std)*100,
                      alpha=0.25, color='steelblue')
axes2[0].axhline(80, ls='--', color='crimson', lw=1, label='80% target')
axes2[0].set(xlabel='Fleet size N', ylabel='Perimeter coverage [%]',
             title=f'Coverage  (R_sense={R_SENSE:.0f} m)')
axes2[0].legend(); axes2[0].grid(alpha=0.4); axes2[0].set_ylim(0, 105)

axes2[1].plot(fleet_sizes, conn_frac*100, 'darkorange', lw=2)
axes2[1].axhline(95, ls='--', color='crimson', lw=1, label='95% target')
axes2[1].set(xlabel='Fleet size N', ylabel='Connectivity probability [%]',
             title=f'Mesh connectivity  (R_comm={R_COMM:.0f} m)')
axes2[1].legend(); axes2[1].grid(alpha=0.4); axes2[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig('output/swarm_coverage_vs_n.png', dpi=150, bbox_inches='tight')
print('Saved: output/swarm_coverage_vs_n.png')
