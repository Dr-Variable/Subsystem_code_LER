"""Run stacked honeycomb codes with open or periodic layers; z is always OPEN.

Keep this file beside the UPDATED subsystem_ler.py.
Install: python -m pip install "numpy>=1.24" "stim>=1.16" "matplotlib>=3.6"
Run:     python -u full_run.py --boundary periodic
         python -u full_run.py --boundary open

The defaults perform 10 million Stim trials (100 code/p points).
Progress prints during decoder construction and after every 5,000 trials.
Completed batches are saved automatically; Ctrl+C and rerun to resume.

Outputs in stacked_results/ (periodic) or stacked_results_open/ (open):
    full_results.json        All inputs, seeds, counts, CIs and exact LERs.
    threshold_summary.json   Refined finite-size crossings and their meaning.
    ler_vs_p.png             Linear and log plots of all four LER curves.
    crossing_drift.png       Crossings between adjacent lattice sizes.

For the PERIODIC Ly=2, Lx=Lz family under IID depolarizing code-capacity noise,
the gauge group is equivalent to square Bacon-Shor. Its asymptotic threshold
is zero. The nonzero finite-size curve crossings are reported separately.
See FULL_RUN.md for the derivation; this is not a noisy-measurement threshold.
Open defaults vary only Lz at fixed Lx=Ly=2 and fit the exact-decoder limit.
Open layers require even Lx; choose custom sizes with repeated --size LX LY LZ.

Quick end-to-end check, using a separate directory:
    python -u full_run.py --shots 256 --batch-shots 128 \
        --p-values 0.02 0.04 0.06 0.08 --output-dir smoke_results

All reusable functions remain in subsystem_ler.py.
"""

from pathlib import Path
import argparse

from subsystem_ler import run_stacked_ler_study


# ==================== EDIT YOUR EXPERIMENT HERE ====================

BOUNDARY = "open"        # Override at runtime with --boundary open|periodic.
CODE_SIZES = [(2, 2, 2), (3, 2, 3), (4, 2, 4), (5, 2, 5)]  # Periodic defaults.
# Runnable open examples: fixed layer size, increasing stack height.
# These are NOT an isotropic size-scaling family for threshold estimation.
OPEN_CODE_SIZES = [(2,1,2),(8,1,5)]
    #(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]

# Total error probability: p = px + py + pz.
# A broad grid covers the finite-size crossings; roots are refined afterward.
P_VALUES = (
    [0.001, 0.002, 0.005]
    + [round(0.010 + 0.005*j, 3) for j in range(19)]
    + [0.12, 0.15, 0.20]
)
SHOTS_PER_POINT = 10_000
NOISE_WEIGHTS = (1, 1, 1)      # px=py=pz=p/3; normalized automatically.
SEED = 12345
BATCH_SHOTS = 5_000           # A progress line and checkpoint after each batch.
MAX_COSETS = 1 << 28          # Enough for both default lists; larger codes may fail.
OUTPUT_DIR = None            # Auto: stacked_results or stacked_results_open.
RESUME = True                # Requires exactly matching run settings.
REFINE_CROSSINGS = True       # Bisection using exact ML rates, without extra MC.
CROSSING_TOLERANCE = 1e-5     # Absolute bracket width in total physical p.

# Change OUTPUT_DIR when changing the noise, sizes, p grid, shots, or batch size.
# Decoder construction can take longer on the largest lattice.
# Refinement is a separate stage after all sampling reaches 100%.
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boundary", choices=("open", "periodic"), default=BOUNDARY,
                        help="Boundary of each 2D layer; z stays open. Default: %(default)s.")
    parser.add_argument("--size", nargs=3, type=int, action="append", metavar=("LX", "LY", "LZ"),
                        help="Custom lattice size; repeat for at least two sizes, in comparison order.")
    parser.add_argument("--shots", type=int, default=SHOTS_PER_POINT,
                        help="Number of Stim shots for EACH code and p.")
    parser.add_argument("--batch-shots", type=int, default=BATCH_SHOTS,
                        help="Shots per checkpoint and progress update.")
    parser.add_argument("--p-values", nargs="+", type=float, default=P_VALUES,
                        help="Explicit total physical error probabilities.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Override the mode-specific output directory; compatible runs resume.")
    parser.add_argument("--crossing-tolerance", type=float, default=CROSSING_TOLERANCE,
                        help="Final bisection bracket width in p.")
    args = parser.parse_args()
    code_sizes = args.size or (OPEN_CODE_SIZES if args.boundary == "open" else CODE_SIZES)
    output_dir = args.output_dir
    if output_dir is None:
        folder = "stacked_results_open" if args.boundary == "open" else "stacked_results"
        output_dir = Path(__file__).resolve().parent / folder
    print(f"Using {'custom' if args.size else 'default'} {args.boundary} sizes: "
          f"{[tuple(size) for size in code_sizes]}", flush=True)
    try:
        run_stacked_ler_study(
            code_sizes=code_sizes, p_values=args.p_values, boundary=args.boundary,
            shots=args.shots,
            noise_weights=NOISE_WEIGHTS, seed=SEED, batch_shots=args.batch_shots,
            max_cosets=MAX_COSETS, output_dir=output_dir, resume=RESUME,
            crossing_tolerance=args.crossing_tolerance,
            refine_crossings=REFINE_CROSSINGS, progress=True
            )
    except KeyboardInterrupt:
        raise SystemExit(130)
