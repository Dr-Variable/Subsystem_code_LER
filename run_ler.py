"""Edit the inputs below, then run: python run_ler.py

Keep this file and subsystem_ler.py in the same directory.
Install: python -m pip install 'stim>=1.16' numpy 'matplotlib>=3.6'
Outputs: results/ler_results.json and results/ler_vs_p.png.
Re-running replaces these outputs; change output_dir to keep separate runs.

All functions are in subsystem_ler.py. This file only configures and runs them.
"""

from pathlib import Path

from subsystem_ler import (
    honeycomb_open_gauge_group, honeycomb_torus_gauge_group,
    plot_ler_results, run_ler_sweep,
    stacked_honeycomb_open_gauge_group, stacked_honeycomb_torus_gauge_group,
)


# ===================== EDIT YOUR EXPERIMENT HERE =====================

# Example: four-qubit Bacon-Shor code. Replace n and G with your own code.
#n = 4
#G = ["X1X2", "X3X4", "Z1Z3", "Z2Z4"]

# For the periodic honeycomb, replace the two lines above with:
# Lx, Ly = 4, 3
# n, G = honeycomb_torus_gauge_group(Lx, Ly)
# Convention: 2*Lx qubits per staggered zigzag row, Ly rows; see the function docstring.

# For open boundaries with disjoint top/bottom XX pairs, instead use:
# Lx, Ly = 2, 1                       # Lx must be even; Ly >= 1.
# n, G = honeycomb_open_gauge_group(Lx, Ly)
# This example has 10 qubits, including additional X2X4 and X7X9.

# For a 3D stack with all boundaries open:
# Lx, Ly, Lz = 2, 2, 2                 # Lx even; Ly,Lz >= 1.
# n, G = stacked_honeycomb_open_gauge_group(Lx, Ly, Lz)
# This example has 32 qubits, 58 generators, and 2 protected logical qubits.

# For a 3D stack periodic within each layer and OPEN along z:
Lx, Ly, Lz = 4, 2, 4
n, G = stacked_honeycomb_torus_gauge_group(Lx, Ly, Lz)
# This example has 16 qubits, 32 generators, and 1 protected logical qubit.
# Both stacked functions connect EVERY corresponding vertex in adjacent layers.

# These are total physical error probabilities p = px + py + pz.
# Use any explicit list; p=0 is also supported with plot_scale="auto".
p_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
shots = 100_000                       # Number of trials AT EACH p value.

# Relative probabilities X:Y:Z; the function normalizes these weights.
noise_weights = (1, 1, 1)              # Depolarizing: px=py=pz=p/3.
# noise_weights = (1, 0, 0)            # X-only: px=p, py=pz=0.
# noise_weights = (1, 1, 10)           # Z-biased noise.

seed = 12345                          # Reproducible for the same environment.
batch_size = 4096
max_cosets = 1 << 20                  # Exact decoder size limit.

output_dir = Path(__file__).resolve().parent / "results"
json_file = output_dir / "ler_results.json"
plot_file = output_dir / "ler_vs_p.png"
plot_scale = "auto"                   # "auto", "linear", or "log".
show_exact_curve = True
show_plot_window = False              # True to open a plot window locally.

# ====================================================================

if __name__ == "__main__":
    sweep = run_ler_sweep(
        n=n, G=G, p_values=p_values, shots=shots,
        noise_weights=noise_weights, seed=seed,
        batch_size=batch_size, max_cosets=max_cosets,
        json_path=json_file,
    )
    # Read the saved JSON for plotting, so the figure uses the recorded data.
    plot_ler_results(
        json_file, plot_file, scale=plot_scale,
        show_exact=show_exact_curve, show=show_plot_window,
    )
    print(f"\nJSON saved to: {json_file}")
    print(f"Plot saved to: {plot_file}")
