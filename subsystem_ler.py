"""Logical block-error rates of Pauli subsystem codes using Stim.

Model: one independent PAULI_CHANNEL_1 on every data qubit, followed by a
perfect stabilizer syndrome and a perfect Pauli correction. All operators and
group membership tests are modulo global phase. This is a code-capacity model;
it does not simulate a noisy gauge-measurement circuit.

Install: python -m pip install 'numpy>=1.24' 'stim>=1.16' 'matplotlib>=3.6'
Run:     python run_ler.py

The three main functions are analyze_code, simulate_noise, and
estimate_logical_error_rate. The sweep, JSON, and plotting functions are at the
end of this file. All editable experiment settings are in run_ler.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import operator
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Callable, Iterable, Iterator, Protocol

import numpy as np
import stim


# Binary Paulis are Python integers: bits 0..n-1 are x, bits n..2*n-1 are z.
# On one qubit: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1).
# Pauli multiplication, ignoring global phase, is therefore integer XOR (^).


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _check_pauli(pauli: int, n: int) -> int:
    pauli = _integer(pauli, "Pauli mask")
    if pauli.bit_length() > 2 * n:
        raise ValueError(f"Pauli mask contains bits outside {n} qubits.")
    return pauli


def parse_pauli(text: str, n: int) -> int:
    """Parse one-based strings, e.g. 'X1X2', 'Y1*Z3', or 'I'.

    Spaces and '*' separators are allowed. Each qubit may occur only once.
    An optional leading '+' or '-' is ignored because phases are irrelevant.
    """
    n = _integer(n, "n", 1)
    if not isinstance(text, str):
        raise ValueError("Every gauge generator must be a string.")
    compact = re.sub(r"[\s*]", "", text).upper()
    if compact.startswith(("+", "-")):
        compact = compact[1:]
    if compact == "I":
        return 0
    terms = list(re.finditer(r"([IXYZ])([1-9][0-9]*)", compact))
    if not terms or "".join(term.group() for term in terms) != compact:
        raise ValueError(f"Invalid Pauli string {text!r}; use e.g. 'X1Y2Z3'.")
    result = 0
    seen = set()
    for term in terms:
        letter, label = term.groups()
        qubit = int(label) - 1
        if qubit >= n:
            raise ValueError(f"Qubit {qubit + 1} is outside 1..{n}.")
        if qubit in seen:
            raise ValueError(f"Qubit {qubit + 1} occurs twice in {text!r}.")
        seen.add(qubit)
        if letter in "XY":
            result |= 1 << qubit
        if letter in "YZ":
            result |= 1 << (n + qubit)
    return result


def format_pauli(pauli: int, n: int) -> str:
    """Convert an integer Pauli mask back to one-based sparse notation."""
    n = _integer(n, "n", 1)
    pauli = _check_pauli(pauli, n)
    terms = []
    for qubit in range(n):
        value = ((pauli >> qubit) & 1) | (((pauli >> (n + qubit)) & 1) << 1)
        if value:
            terms.append(f"{'IXZY'[value]}{qubit + 1}")
    return "".join(terms) or "I"


def symplectic_product(a: int, b: int, n: int) -> int:
    """Return 1 for anticommuting Paulis and 0 for commuting Paulis."""
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count()
            + ((a >> n) & (b & mask)).bit_count()) & 1


@dataclass(frozen=True)
class BinarySpan:
    """GF(2) row space with a descending-pivot echelon basis."""

    rows: tuple[int, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[int]) -> BinarySpan:
        pivots: dict[int, int] = {}
        for row in rows:
            while row:
                pivot = row.bit_length() - 1
                if pivot in pivots:
                    row ^= pivots[pivot]
                else:
                    pivots[pivot] = row
                    break
        return cls(tuple(pivots[p] for p in sorted(pivots, reverse=True)))

    def reduce(self, value: int) -> int:
        """Canonical remainder modulo this row space; zero means membership."""
        for row in self.rows:
            if value & (1 << (row.bit_length() - 1)):
                value ^= row
        return value


def _nullspace(rows: Iterable[int], columns: int) -> tuple[int, ...]:
    """Basis for the right kernel of a GF(2) matrix stored as integer rows."""
    matrix = list(rows)
    pivots = []
    rank = 0
    for column in range(columns):
        chosen = next((j for j in range(rank, len(matrix))
                       if (matrix[j] >> column) & 1), None)
        if chosen is None:
            continue
        matrix[rank], matrix[chosen] = matrix[chosen], matrix[rank]
        for j in range(len(matrix)):
            if j != rank and (matrix[j] >> column) & 1:
                matrix[j] ^= matrix[rank]
        pivots.append(column)
        rank += 1
    free = sorted(set(range(columns)) - set(pivots))
    vectors = []
    for column in free:
        vector = 1 << column
        for row, pivot in zip(matrix, pivots):
            if (row >> column) & 1:
                vector |= 1 << pivot
        vectors.append(vector)
    return tuple(vectors)


# ---------------------------------------------------------------------------
# PART 1: Gauge group -> independent stabilizers, gauge qubits, logical qubits.
# ---------------------------------------------------------------------------


def honeycomb_torus_gauge_group(Lx: int, Ly: int) -> tuple[int, list[str]]:
    """Return (n, G) for a periodic honeycomb with vertical XX and slanted ZZ.

    Convention: Lx two-site cells per zigzag row and Ly staggered rows.
    There are Lx*Ly hexagonal faces and n=2*Lx*Ly qubits. Indices are one-based:

        A(x,y) = 2*(y*Lx + x) + 1
        B(x,y) = A(x,y) + 1

    The three edge families, with x modulo Lx and y modulo Ly, are:

        ZZ: A(x,y) -- B(x,y)
        ZZ: B(x,y) -- A(x+1,y)
        XX: B(x,y) -- A(x,y+1)

    To see the geometry, before wrapping embed
        A(x,y) = (sqrt(3)*(x+y/2), 3*y/2),
        B(x,y) = A(x,y) + (sqrt(3)/2, 1/2).
    The XX bonds are vertical and both ZZ bonds are slanted unit-length edges.
    Periods are (sqrt(3)*Lx, 0) and (sqrt(3)*Ly/2, 3*Ly/2): this is a
    parallelogram torus, periodic along the chosen honeycomb cell directions.
    It does not silently impose a different rectangular seam convention.

    For Lx,Ly>=2 every hexagonal face has six distinct vertices. Each qubit
    touches one XX and two ZZ bonds, and G contains 3*Lx*Ly edge generators
    (these need not be algebraically independent). Each face has two XX and
    four ZZ edges. ZZ alone has Ly connected components.

    Lx=1 or Ly=1 gives a collapsed small periodic quotient. Equal Pauli
    generators are removed; different XX and ZZ on the same pair are retained.
    In particular Ly=1 has no protected logical qubits. G is a generating
    set for the gauge group, not an enumeration of every group element.
    """
    Lx = _integer(Lx, "Lx", 1)
    Ly = _integer(Ly, "Ly", 1)

    def a(x: int, y: int) -> int:
        return 2 * ((y % Ly) * Lx + (x % Lx)) + 1

    G = []
    seen = set()
    for y in range(Ly):
        for x in range(Lx):
            A = a(x, y)
            B = A + 1
            edges = (("X", B, a(x, y + 1)),
                     ("Z", A, B),
                     ("Z", B, a(x + 1, y)))
            for pauli, u, v in edges:
                u, v = sorted((u, v))
                generator = f"{pauli}{u}{pauli}{v}"
                if generator not in seen:
                    seen.add(generator)
                    G.append(generator)
    return 2 * Lx * Ly, G


def honeycomb_open_gauge_group(Lx: int, Ly: int) -> tuple[int, list[str]]:
    """Return (n, G) for an open honeycomb with disjoint top/bottom XX pairs.

    Lx is the number of hexagons per row and Ly the number of staggered face
    rows. Lx must be positive and even; Ly may be any positive integer.
    Build complete faces rather than cutting the qubit set of the torus:
        n = 2*Lx*Ly + 2*Lx + 2*Ly.
    There are no periodic identifications or wraparound bonds.

    Use the same infinite-lattice geometry as honeycomb_torus_gauge_group:
        A(x,y) = (sqrt(3)*(x+y/2), 3*y/2),
        B(x,y) = A(x,y) + (sqrt(3)/2, 1/2).
    Take the union of the faces indexed 0<=x<Lx, 0<=y<Ly. One face is
        B(x,y), A(x,y+1), B(x,y+1),
        A(x+1,y+1), B(x+1,y), A(x+1,y).
    Its two vertical edges are XX and its four slanted edges are ZZ.

    The vertices with no vertical XX bond are exactly:
        bottom: A(1,0), ..., A(Lx,0),
        top:    B(0,Ly), ..., B(Lx-1,Ly).
    Pair consecutive vertices in EACH of these left-to-right lists:
        bottom: A(1,0)--A(2,0), A(3,0)--A(4,0), ...,
        top:    B(0,Ly)--B(1,Ly), B(2,Ly)--B(3,Ly), ... .
    Thus each qubit belongs to exactly one XX generator. These extra bonds
    connect alternate sites of the full zigzag boundary, without joining
    the two boundaries or closing a boundary into a cycle. Odd Lx cannot
    complete this disjoint pairing within each boundary and is rejected.

    Labels are one-based, ordered by zigzag row from bottom to top, then
    left to right within each row. G lists XX first, then ZZ, with sorted
    endpoints and no duplicate face edges. Shared face edges appear once.

    Example Lx=2,Ly=1: n=10, with additional X2X4 and X7X9.
    For this construction, the ZZ graph has Ly+1 components and k=Ly.
    """
    Lx = _integer(Lx, "Lx", 1)
    Ly = _integer(Ly, "Ly", 1)
    if Lx % 2:
        raise ValueError("Lx must be even to pair every free top/bottom vertex "
                         "once without wrapping or joining the boundaries.")

    # A vertex is (x, y, sublattice), with sublattice 0=A and 1=B.
    vertices: set[tuple[int, int, int]] = set()
    x_edges = set()
    z_edges = set()
    for y in range(Ly):
        for x in range(Lx):
            face = [(x, y, 1), (x, y+1, 0), (x, y+1, 1),
                    (x+1, y+1, 0), (x+1, y, 1), (x+1, y, 0)]
            vertices.update(face)
            for j in range(6):
                edge = tuple(sorted((face[j], face[(j+1) % 6])))
                (x_edges if j in (0, 3) else z_edges).add(edge)

    # Determine eligibility from the actual vertical edges, before adding pairs.
    has_vertical_x = {vertex for edge in x_edges for vertex in edge}
    for boundary_y in (0, Ly):
        free = sorted((v for v in vertices
                       if v[1] == boundary_y and v not in has_vertical_x),
                      key=lambda v: (v[0], v[2]))
        for u, v in zip(free[::2], free[1::2]):
            x_edges.add(tuple(sorted((u, v))))

    ordered_vertices = sorted(vertices, key=lambda v: (v[1], v[0], v[2]))
    labels = {vertex: j+1 for j, vertex in enumerate(ordered_vertices)}
    G = []
    for pauli, edges in (("X", x_edges), ("Z", z_edges)):
        numbered_edges = sorted(tuple(sorted((labels[u], labels[v])))
                                for u, v in edges)
        G.extend(f"{pauli}{u}{pauli}{v}" for u, v in numbered_edges)
    return len(ordered_vertices), G


def _stack_with_xx(n_layer: int, layer_generators: list[str],
                   Lz: int) -> tuple[int, list[str]]:
    """Copy validated layers and join equal local labels; z is always open."""
    G = []
    for layer in range(Lz):
        offset = layer * n_layer
        for generator in layer_generators:
            G.append(re.sub(r"[0-9]+",
                            lambda match: str(int(match.group()) + offset),
                            generator))
    for layer in range(Lz - 1):
        offset = layer * n_layer
        G.extend(f"X{offset + q}X{offset + n_layer + q}"
                 for q in range(1, n_layer + 1))
    return n_layer * Lz, G


def stacked_honeycomb_open_gauge_group(
        Lx: int, Ly: int, Lz: int) -> tuple[int, list[str]]:
    """Return (n, G) for Lz open honeycomb layers joined by weight-two XX.

    Each layer is exactly honeycomb_open_gauge_group(Lx, Ly), including its
    disjoint top/bottom boundary XX pairs. Lx must be positive and even;
    Ly and Lz must be positive integers. All three directions are open.

    If n0 = 2*Lx*Ly + 2*Lx + 2*Ly, local qubit q (1<=q<=n0) in layer
    ell (0<=ell<Lz) has the one-based label q + ell*n0. Keep every 2D
    generator in every layer, and add

        X[q + ell*n0] X[q + (ell+1)*n0]

    for every q and 0<=ell<Lz-1. There is no last-to-first layer bond.
    This links corresponding vertices, as in the stacked reference drawing;
    local numbering remains the numbering of the existing 2D function.

    The output contains n=n0*Lz qubits and
        Lz*len(G_2d) + n0*(Lz-1)
    distinct weight-two generators. Some are algebraically dependent.
    Lz=1 returns exactly the original 2D result. The protected logical
    qubit count is k=Ly, independent of Lz.
    """
    Lz = _integer(Lz, "Lz", 1)
    n_layer, layer_generators = honeycomb_open_gauge_group(Lx, Ly)
    return _stack_with_xx(n_layer, layer_generators, Lz)


def stacked_honeycomb_torus_gauge_group(
        Lx: int, Ly: int, Lz: int) -> tuple[int, list[str]]:
    """Return (n, G) for Lz periodic honeycomb layers, with open stacking.

    Each layer is exactly honeycomb_torus_gauge_group(Lx, Ly), with its
    existing two in-plane period vectors. Periodicity applies ONLY within
    each layer: z is open, so the topology is a torus times an interval.
    Lx, Ly, and Lz must be positive integers. Dimensions Lx=1 or Ly=1
    retain the collapsed-quotient convention of the 2D helper; for ordinary
    six-vertex hexagons use Lx,Ly>=2.

    With n0=2*Lx*Ly, local qubit q in layer ell has label q+ell*n0.
    Copy all intralayer XX and ZZ generators, then add one XX generator
    between q+ell*n0 and q+(ell+1)*n0 for every q=1,...,n0 and
    ell=0,...,Lz-2. No XX or ZZ bonds wrap from the last layer to the first.

    The output has n=n0*Lz qubits and
        Lz*len(G_2d) + n0*(Lz-1)
    distinct weight-two generators, including algebraic redundancies.
    Lz=1 returns exactly the original 2D result. The protected logical
    qubit count is k=Ly-1, independent of Lz.
    """
    Lz = _integer(Lz, "Lz", 1)
    n_layer, layer_generators = honeycomb_torus_gauge_group(Lx, Ly)
    return _stack_with_xx(n_layer, layer_generators, Lz)


@dataclass(frozen=True)
class SubsystemCode:
    n: int
    gauge_span: BinarySpan
    stabilizers: tuple[int, ...]
    s: int  # Number of independent stabilizer generators.
    r: int  # Number of gauge qubits.
    k: int  # Number of protected logical qubits.

    @property
    def gauge_rank(self) -> int:
        return len(self.gauge_span.rows)

    @property
    def stabilizer_generators(self) -> list[str]:
        return [format_pauli(p, self.n) for p in self.stabilizers]

    @property
    def coset_bits(self) -> int:
        return 2 * self.n - self.gauge_rank  # Also equal to s + 2*k.

    def syndrome(self, error: int) -> int:
        """Syndrome bit j is 1 iff error anticommutes with stabilizer j."""
        return sum(symplectic_product(error, stab, self.n) << j
                   for j, stab in enumerate(self.stabilizers))

    def syndrome_bits(self, syndrome: int) -> np.ndarray:
        """Unpack a syndrome for an external decoder, in stabilizer order."""
        syndrome = _integer(syndrome, "syndrome")
        if syndrome >= (1 << self.s):
            raise ValueError("Syndrome has too many bits.")
        return np.array([(syndrome >> j) & 1 for j in range(self.s)], dtype=np.uint8)

    def in_gauge(self, pauli: int) -> bool:
        return self.gauge_span.reduce(_check_pauli(pauli, self.n)) == 0

    def summary(self) -> dict:
        return {"n": self.n, "gauge_rank": self.gauge_rank,
                "stabilizers": self.s, "gauge_qubits": self.r,
                "logical_qubits": self.k,
                "stabilizer_generators": self.stabilizer_generators}


def analyze_code(n: int, G: Iterable[str]) -> SubsystemCode:
    """Analyze arbitrary Pauli gauge generators, including non-CSS codes.

    First remove dependent generators. For independent gauge matrix B, let
    Omega = B J B^T over GF(2). Then
        r = rank(Omega)/2, s = rank(B)-2*r, k = n-s-r.
    Kernel vectors of Omega give products of gauge generators in the center.
    The returned stabilizers are independent, with global phases omitted.
    """
    n = _integer(n, "n", 1)
    if isinstance(G, str):
        raise ValueError("G must be a sequence of strings, not one string.")
    span = BinarySpan.from_rows(parse_pauli(text, n) for text in G)
    basis = span.rows
    gram = [sum(symplectic_product(a, b, n) << j for j, b in enumerate(basis))
            for a in basis]
    kernel = _nullspace(gram, len(basis))
    stabilizers = []
    for coefficients in kernel:
        pauli = 0
        for j, generator in enumerate(basis):
            if (coefficients >> j) & 1:
                pauli ^= generator
        stabilizers.append(pauli)
    s = len(stabilizers)
    r = (len(basis) - s) // 2
    k = n - s - r
    return SubsystemCode(n, span, tuple(stabilizers), s, r, k)


# ---------------------------------------------------------------------------
# PART 2: Stim samples the physical noise, retaining the actual Pauli errors.
# ---------------------------------------------------------------------------


def _probabilities(px: float, py: float, pz: float) -> tuple[float, ...]:
    values = tuple(float(p) for p in (px, py, pz))
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("px, py, pz must be finite probabilities in [0, 1].")
    total = math.fsum(values)
    if total > 1:
        raise ValueError("px + py + pz must be <= 1.")
    return (1 - total, *values)  # I, X, Y, Z.


def simulate_noise(n: int, px: float, py: float, pz: float, shots: int,
                   *, seed: int | None = None, batch_size: int = 4096) -> Iterator[int]:
    """Yield Stim-sampled Pauli masks, one per shot; all qubits have the same noise.

    X/Y/Z are disjoint events with probabilities px/py/pz. In particular, a
    Y error has both x and z bits set. Sampling uses PAULI_CHANNEL_1, not three
    independent X_ERROR/Y_ERROR/Z_ERROR operations.

    Only error frames are needed: disabling stabilizer randomization prevents
    fictitious initial Z flips. There is no state preparation or measurement in
    this noise-only circuit, so no encoded-state amplitudes are needed.
    """
    n = _integer(n, "n", 1)
    shots = _integer(shots, "shots", 1)
    batch_size = min(_integer(batch_size, "batch_size", 1), shots)
    _, px, py, pz = _probabilities(px, py, pz)
    if seed is not None:
        seed = _integer(seed, "seed")
        if seed >= (1 << 64):
            raise ValueError("seed must be smaller than 2**64.")
    circuit = stim.Circuit()
    circuit.append("PAULI_CHANNEL_1", range(n), [px, py, pz])
    simulator = stim.FlipSimulator(batch_size=batch_size, num_qubits=n,
                                   disable_stabilizer_randomization=True, seed=seed)
    remaining = shots
    while remaining:
        simulator.clear()  # Reset frames between batches, retaining the RNG.
        simulator.do(circuit)
        xs, zs, *_ = simulator.to_numpy(transpose=True, bit_packed=True,
                                        output_xs=True, output_zs=True)
        count = min(remaining, batch_size)
        for xrow, zrow in zip(xs[:count], zs[:count]):
            yield (int.from_bytes(xrow.tobytes(), "little")
                   | (int.from_bytes(zrow.tobytes(), "little") << n))
        remaining -= count


# ---------------------------------------------------------------------------
# PART 3: Decode the syndrome, apply the correction, and test membership in G.
# ---------------------------------------------------------------------------


class Decoder(Protocol):
    def decode(self, syndrome: int) -> int:
        """Return a Pauli correction mask using only the measured syndrome."""
        ...


class ExactGaugeMLDecoder:
    """Exact degenerate maximum-likelihood decoder for a manageable quotient.

    Errors differing by a gauge operator are equivalent. The quotient has
    2**(s+2*k) classes. Convolving independent single-qubit distributions on
    this quotient gives the TOTAL probability of every gauge class. For each
    syndrome, choose the class with greatest probability (ties: lowest label).

    Time: O(n * 2**(s+2*k)); memory: O(2**(s+2*k)). No low-weight truncation.
    The default cap is 2**20 classes. This is suitable when s+2*k is modest,
    even if n is larger. Supply another Decoder for larger quotients.
    """

    def __init__(self, code: SubsystemCode, px: float, py: float, pz: float,
                 *, max_cosets: int = 1 << 20,
                 progress_callback: Callable[[str, int, int], None] | None = None):
        self.code = code
        probabilities = _probabilities(px, py, pz)
        self.probabilities = probabilities
        max_cosets = _integer(max_cosets, "max_cosets", 1)
        count = 1 << code.coset_bits
        if count > max_cosets:
            raise ValueError(
                f"Exact decoding needs 2**{code.coset_bits} = {count:,} gauge "
                f"cosets; max_cosets={max_cosets:,}. Supply another decoder "
                "or explicitly increase max_cosets if memory permits.")
        if count > np.iinfo(np.intp).max:
            raise ValueError("The quotient is too large for NumPy array indices.")
        pivot_columns = {row.bit_length() - 1 for row in code.gauge_span.rows}
        self._free_columns = tuple(j for j in range(2 * code.n)
                                   if j not in pivot_columns)
        labels = np.arange(count, dtype=np.intp)
        distribution = np.zeros(count, dtype=np.float64)
        distribution[0] = 1.0
        p_i, p_x, p_y, p_z = probabilities
        if progress_callback is not None:
            progress_callback("noise convolution", 0, code.n)
        for qubit in range(code.n):
            x = self._label(1 << qubit)
            z = self._label(1 << (code.n + qubit))
            updated = p_i * distribution
            for probability, shift in ((p_x, x), (p_y, x ^ z), (p_z, z)):
                if probability:
                    updated += probability * distribution[labels ^ shift]
            distribution = updated
            if progress_callback is not None:
                progress_callback("noise convolution", qubit + 1, code.n)
        distribution /= distribution.sum()  # Remove floating-point drift.

        # Syndrome is constant on each gauge coset and linear in its label.
        if progress_callback is not None:
            progress_callback("syndrome and recovery tables", 0, 1)
        syndromes = np.zeros(count, dtype=np.intp)
        for j, column in enumerate(self._free_columns):
            half = 1 << j
            syndromes[half:2 * half] = syndromes[:half] ^ code.syndrome(1 << column)
        best_probability = np.full(1 << code.s, -1.0)
        np.maximum.at(best_probability, syndromes, distribution)
        candidates = np.where(distribution == best_probability[syndromes], labels, count)
        self._best_cosets = np.full(1 << code.s, count, dtype=np.intp)
        np.minimum.at(self._best_cosets, syndromes, candidates)

        # Sum failed classes directly to avoid 1 - success cancellation at low p.
        self.exact_logical_error_rate = float(
            distribution[labels != self._best_cosets[syndromes]].sum())
        if progress_callback is not None:
            progress_callback("syndrome and recovery tables", 1, 1)

    def _label(self, pauli: int) -> int:
        remainder = self.code.gauge_span.reduce(pauli)
        return sum(((remainder >> column) & 1) << j
                   for j, column in enumerate(self._free_columns))

    def decode(self, syndrome: int) -> int:
        syndrome = _integer(syndrome, "syndrome")
        if syndrome >= len(self._best_cosets):
            raise ValueError("Syndrome has too many bits.")
        label = int(self._best_cosets[syndrome])
        return sum(((label >> j) & 1) << column
                   for j, column in enumerate(self._free_columns))


@dataclass(frozen=True)
class TrialResult:
    error: int
    syndrome: int
    correction: int
    residual: int
    success: bool


def decode_and_check(code: SubsystemCode, error: int, decoder: Decoder) -> TrialResult:
    """Correct one error. Success means C*E is in G, up to global phase.

    A decoder returning a correction with the wrong syndrome raises an error;
    such a recovery is not silently reported as a logical Pauli failure.
    """
    error = _check_pauli(error, code.n)
    syndrome = code.syndrome(error)
    correction = _check_pauli(decoder.decode(syndrome), code.n)
    if code.syndrome(correction) != syndrome:
        raise ValueError("Decoder returned a correction with the wrong syndrome.")
    residual = correction ^ error
    return TrialResult(error, syndrome, correction, residual, code.in_gauge(residual))


@dataclass(frozen=True)
class LogicalErrorResult:
    shots: int
    failures: int
    logical_error_rate: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    exact_logical_error_rate: float | None


def _wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    """95% Wilson interval; remains informative with zero observed failures."""
    z = 1.959963984540054
    p = failures / shots
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    lower = 0.0 if failures == 0 else max(0.0, center - half)
    upper = 1.0 if failures == shots else min(1.0, center + half)
    return lower, upper


def estimate_logical_error_rate(
    code: SubsystemCode, px: float, py: float, pz: float, shots: int = 100_000,
    *, decoder: Decoder | None = None, seed: int | None = None,
    batch_size: int = 4096, max_cosets: int = 1 << 20,
) -> LogicalErrorResult:
    """Estimate Pr(C*E not in G), a block failure on ANY protected logical qubit.

    The default decoder is built for the supplied noise probabilities. To
    replace it, pass an object implementing decode(syndrome_int)->pauli_int.
    A supplied ExactGaugeMLDecoder must match both the code and the noise.
    No actual error information is passed to the decoder.
    """
    shots = _integer(shots, "shots", 1)
    probabilities = _probabilities(px, py, pz)
    if decoder is None:
        decoder = ExactGaugeMLDecoder(code, px, py, pz, max_cosets=max_cosets)
    exact = None
    if isinstance(decoder, ExactGaugeMLDecoder):
        if decoder.code != code or decoder.probabilities != probabilities:
            raise ValueError("The supplied exact decoder must match the code and noise.")
        exact = decoder.exact_logical_error_rate
    failures = 0
    for error in simulate_noise(code.n, px, py, pz, shots, seed=seed, batch_size=batch_size):
        failures += not decode_and_check(code, error, decoder).success
    rate = failures / shots
    return LogicalErrorResult(shots, failures, rate,
                              math.sqrt(rate * (1 - rate) / shots),
                              _wilson_interval(failures, shots), exact)


# ---------------------------------------------------------------------------
# SWEEPS AND OUTPUT: called by run_ler.py; no simulation runs on import.
# ---------------------------------------------------------------------------


def pauli_probabilities(p: float, noise_weights: Iterable[float] = (1, 1, 1)
                        ) -> tuple[float, float, float]:
    """Convert total error probability p to (px, py, pz) at a fixed bias.

    noise_weights gives the ratio px:py:pz. Examples:
      (1, 1, 1): depolarizing, each Pauli has probability p/3.
      (1, 0, 0): X-only, px=p.
      (1, 1, 10): Z-biased, px=py=p/12 and pz=10*p/12.
    """
    p = float(p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Every p must be finite and in [0, 1].")
    weights = tuple(float(w) for w in noise_weights)
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("noise_weights must contain three finite nonnegative numbers.")
    largest = max(weights)
    if largest == 0:
        raise ValueError("At least one noise weight must be positive.")
    # Scale first to avoid overflow when summing large input weights.
    scaled = tuple(w / largest for w in weights)
    total = math.fsum(scaled)
    probabilities = [p * w / total for w in scaled]
    # Absorb floating-point rounding into the largest component; preserve sum=p.
    j = max(range(3), key=probabilities.__getitem__)
    probabilities[j] += p - math.fsum(probabilities)
    _probabilities(*probabilities)
    return tuple(probabilities)


def save_results_json(results: dict, path: str | Path) -> Path:
    """Write strict JSON atomically, preserving the previous file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# MULTI-SIZE STUDY: resumable runs and finite-size crossing diagnostics.
# ---------------------------------------------------------------------------


def _study_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


def _bacon_shor_component_grid(n: int, gauges: list[str]) -> dict:
    """Check gauge-group equivalence to a rectangular Bacon-Shor grid.

    A connected graph of XX edges spans all even X subsets of its vertices;
    likewise for ZZ. If every X component meets every Z component at exactly
    one qubit, these are precisely the row/column Bacon-Shor gauge spaces,
    after a qubit permutation. This tests the spaces, not the drawn geometry.
    """
    component_sets = []
    for pauli in ("X", "Z"):
        neighbors = [set() for _ in range(n)]
        for text in gauges:
            if not text.startswith(pauli):
                continue
            u, v = (int(q)-1 for q in text[1:].split(pauli))
            neighbors[u].add(v)
            neighbors[v].add(u)
        unseen = set(range(n))
        groups = []
        while unseen:
            root = unseen.pop()
            group, pending = {root}, [root]
            while pending:
                for other in neighbors[pending.pop()] & unseen:
                    unseen.remove(other)
                    group.add(other)
                    pending.append(other)
            groups.append(group)
        component_sets.append(groups)
    xs, zs = component_sets
    equivalent = len(xs)*len(zs) == n and all(len(x & z) == 1 for x in xs for z in zs)
    return {"equivalent": equivalent, "x_components": len(xs), "z_components": len(zs)}


def _study_batch_seed(seed: int, code_index: int, p_index: int, batch_index: int) -> int:
    return int(np.random.SeedSequence([seed, code_index, p_index, batch_index])
               .generate_state(1, dtype=np.uint64)[0])


def _crossing_brackets(ps: list[float], differences: list[float]) -> list[tuple[int, int]]:
    """Find sign changes; exclude the trivial p=0 equality and flat ties."""
    resolved = [j for j, (p, delta) in enumerate(zip(ps, differences))
                if p > 0 and abs(delta) > 1e-12]
    return [(a, b) for a, b in zip(resolved, resolved[1:])
            if (differences[a] > 0) != (differences[b] > 0)]


def analyze_stacked_crossings(
    study: dict, *,
    exact_rate: Callable[[int, float], float] | None = None,
    tolerance: float = 1e-5, progress: bool = True,
) -> dict:
    """Find ALL adjacent-size sign-changing crossings of exact LER curves.

    With exact_rate, refine each root by bisection. Its final bracket is a
    numerical root-location interval, NOT a statistical confidence interval.
    Without that callback, report linear interpolation on the sampled grid.
    Monte Carlo noise is not used to locate crossings because exact ML rates
    are available. No common threshold is fabricated by averaging crossings.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("crossing tolerance must be finite and between 0 and 1.")
    codes = study["codes"]
    ps = study["config"]["p_values"]
    if any(p["exact_logical_error_rate"] is None for c in codes for p in c["results"]):
        raise ValueError("Every requested exact rate is needed before crossing analysis.")
    pairs = []
    for index in range(len(codes)-1):
        left, right = codes[index:index+2]
        a_rates = [p["exact_logical_error_rate"] for p in left["results"]]
        b_rates = [p["exact_logical_error_rate"] for p in right["results"]]
        delta = [b-a for a, b in zip(a_rates, b_rates)]
        roots = []
        label = f"{tuple(left['size'])} vs {tuple(right['size'])}"
        _study_log(f"CROSSINGS [{index+1}/{len(codes)-1}] {label}", progress)
        for start, stop in _crossing_brackets(ps, delta):
            lo, hi = ps[start], ps[stop]
            flo, fhi = delta[start], delta[stop]
            original_bracket = [lo, hi]
            estimate = lo - flo*(hi-lo)/(fhi-flo)
            root_ler = float(np.interp(estimate, ps, a_rates))
            iterations = 0
            if exact_rate is not None:
                while hi-lo > tolerance and iterations < 64:
                    mid = (lo+hi)/2
                    ra, rb = exact_rate(index, mid), exact_rate(index+1, mid)
                    fm = rb-ra
                    iterations += 1
                    _study_log(f"  root step {iterations}: p={mid:.7f}, "
                               f"LER difference={fm:+.3e}, bracket width={hi-lo:.3g}", progress)
                    estimate, root_ler = mid, (ra+rb)/2
                    if fm == 0:
                        break  # Retain the honest bracket at a floating-point zero.
                    if (flo > 0) == (fm > 0):
                        lo, flo = mid, fm
                    else:
                        hi, fhi = mid, fm
                estimate = min(hi, max(lo, estimate))
            roots.append({
                "p_crossing": estimate,
                "ler_at_last_evaluation_or_interpolation": root_ler,
                "initial_bracket": original_bracket,
                "final_bracket": [lo, hi],
                "bracket_width": hi-lo,
                "iterations": iterations,
                "tolerance_met": hi-lo <= tolerance if exact_rate is not None else False,
                "direction": "larger_code_becomes_worse" if delta[start] < 0 else
                             "larger_code_becomes_better",
                "method": "exact_ML_bisection" if exact_rate is not None else
                          "exact_ML_grid_linear_interpolation",
            })
            _study_log(f"  finite-size crossing p={estimate:.7f}; "
                       f"bracket [{lo:.7f}, {hi:.7f}]", progress)
        if not roots:
            _study_log("  No resolved sign-changing crossing in the supplied p range.", progress)
        pairs.append({"size_a": left["size"], "size_b": right["size"],
                      "status": "crossings_found" if roots else "no_crossing_in_sampled_range",
                      "crossings": roots})

    # Checkpoints predating boundary selection all used periodic layers.
    boundary = study["config"].get("boundary", "periodic")
    weights = study["config"]["noise_weights"]
    square_family = all(c["size"][1] == 2 and c["size"][0] == c["size"][2]
                        and c["bacon_shor_grid"]["equivalent"] for c in codes)
    depolarizing = weights[0] == weights[1] == weights[2] and weights[0] > 0
    analytic_zero = boundary == "periodic" and square_family and depolarizing
    theory = {
        "p_threshold": 0.0 if analytic_zero else None,
        "status": "analytic_zero_threshold" if analytic_zero else
                  "not_determined_from_finite_size_crossings",
        "inferred_from_four_curve_fit": False,
        "scope": "Periodic layers, open z; Lx=Lz=L, Ly=2, L->infinity; "
                 "IID depolarizing code-capacity noise",
        "explanation": (
            "The gauge group is that of a (2L)x(2L) Bacon-Shor code under a qubit "
            "permutation. At every fixed p>0 the depolarizing gauge-coset distribution "
            "approaches uniform as L grows, so optimal block LER approaches 3/4. "
            "Finite-size crossings are crossovers, not a positive asymptotic threshold."
            if analytic_zero else
            "The analytic square-family depolarizing conclusion is not applied to "
            "these settings. Pairwise finite-size crossings alone do not establish "
            "an asymptotic threshold."
        ),
        "background_reference": "https://arxiv.org/abs/1209.0794",
        "proof_location": "FULL_RUN.md, Threshold interpretation",
    }
    return {
        "boundary": boundary, "z_boundary": "open",
        "noise": study["noise"], "decoder": "exact_degenerate_maximum_likelihood",
        "asymptotic_threshold": theory, "adjacent_size_pairs": pairs,
        "numerical_interval_meaning": "Bisection brackets, not statistical confidence intervals.",
        "crossing_source": "Exact model rates; Monte Carlo estimates are plotted separately.",
        "crossing_limit": "Only crossings bracketed by the supplied p grid are detected.",
    }


def plot_stacked_ler_study(study: dict, report: dict, output_dir: str | Path) -> list[Path]:
    """Plot Monte Carlo Wilson intervals, exact curves, and crossing drift."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    figure = Figure(figsize=(11, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for index, code in enumerate(study["codes"]):
        points = [p for p in code["results"] if p["shots"] > 0]
        ps = np.array([p["p"] for p in points])
        rates = np.array([p["logical_error_rate"] for p in points])
        exact = np.array([p["exact_logical_error_rate"] for p in points])
        exact_curve = dict(zip(ps, exact))
        exact_curve.update({p["p"]: p["exact_logical_error_rate"]
                            for p in study.get("extra_exact_evaluations", [])
                            if p["code_index"] == index})
        exact_ps = np.array(sorted(exact_curve))
        exact_values = np.array([exact_curve[p] for p in exact_ps])
        low = np.array([p["confidence_interval_95"][0] for p in points])
        high = np.array([p["confidence_interval_95"][1] for p in points])
        label = f"{tuple(code['size'])}, n={code['n']}"
        color = colors[index % len(colors)]
        axes[0].plot(exact_ps, exact_values, color=color, lw=1.8, label=label)
        axes[0].errorbar(ps, rates, yerr=[np.maximum(0, rates-low), np.maximum(0, high-rates)],
                         fmt="o", ms=3, color=color, capsize=2, lw=0.8)
        mask = (exact_ps > 0) & (exact_values > 0)
        axes[1].plot(exact_ps[mask], exact_values[mask], color=color, lw=1.8)
        mask = (ps > 0) & (rates > 0)
        # Truncate a lower interval endpoint only for logarithmic display.
        lower_errors = rates[mask]-np.maximum(low[mask], rates[mask]*1e-4)
        axes[1].errorbar(ps[mask], rates[mask],
                         yerr=[lower_errors, high[mask]-rates[mask]],
                         fmt="o", ms=3, color=color, capsize=2, lw=0.8)
        zero = (ps > 0) & (rates == 0)
        if np.any(zero):
            axes[1].errorbar(ps[zero], high[zero], yerr=high[zero]*0.3,
                             uplims=True, fmt="v", ms=3, color=color)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Exact ML curves and Stim samples")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Log scale; zero counts shown as upper limits")
    for ax in axes:
        ax.set_xlabel("Total physical error probability p")
        ax.set_ylabel("Logical block error rate")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    boundary = study["config"].get("boundary", "periodic")
    figure.suptitle(f"{boundary.capitalize()} honeycomb layers, open z | "
                    "Perfect syndrome and correction")
    path = output_dir / "ler_vs_p.png"
    figure.savefig(path, dpi=240)
    figure.clear()
    paths.append(path)

    figure = Figure(figsize=(7, 4.4), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    labels = []
    any_root = False
    observed_roots = []
    for index, pair in enumerate(report["adjacent_size_pairs"]):
        labels.append(f"{tuple(pair['size_a'])}\nvs {tuple(pair['size_b'])}")
        for root in pair["crossings"]:
            p = root["p_crossing"]
            observed_roots.append(p)
            lo, hi = root["final_bracket"]
            ax.errorbar(index, p, yerr=[[max(0, p-lo)], [max(0, hi-p)]],
                         fmt="o", color="#0072B2", capsize=5)
            ax.annotate(f"{p:.5f}", (index, p), xytext=(0, 10),
                         textcoords="offset points", ha="center")
            any_root = True
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Finite-size crossing probability")
    ax.set_title(f"Crossing drift | {boundary} layers, open z")
    ax.set_xlim(-0.5, max(0.5, len(labels)-0.5))
    ax.set_ylim(0, 1.28*max(observed_roots, default=0.1))
    ax.grid(axis="y", alpha=0.22)
    if report["asymptotic_threshold"]["p_threshold"] == 0.0:
        ax.axhline(0, ls="--", color="#555555", label="Analytic asymptotic threshold: 0")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    if not any_root:
        ax.text(0.5, 0.5, "No resolved crossings in the supplied p range",
                 transform=ax.transAxes, ha="center")
    path = output_dir / "crossing_drift.png"
    figure.savefig(path, dpi=240)
    figure.clear()
    paths.append(path)
    return paths


def run_stacked_ler_study(
    code_sizes: Iterable[tuple[int, int, int]],
    p_values: Iterable[float], *,
    boundary: str = "periodic",
    shots: int = 100_000, noise_weights: Iterable[float] = (1, 1, 1),
    seed: int = 12345, batch_shots: int = 5000, max_cosets: int = 1 << 20,
    output_dir: str | Path | None = None, resume: bool = True,
    crossing_tolerance: float = 1e-5, refine_crossings: bool = True,
    progress: bool = True,
) -> dict:
    """Run open or periodic stacked codes with checkpoints and detailed progress.

    boundary selects the 2D layers; z is always open. Open layers require even
    Lx. With output_dir=None, periodic runs use stacked_results and open runs
    use stacked_results_open. Old checkpoints without a boundary are periodic.

    Uses Stim and the existing exact ML decoder and CE-in-G scoring. Default
    noise is px=py=pz=p/3. Every batch uses an independent, recorded seed.
    Resume never reuses a completed batch; changed settings or incompatible
    NumPy/Stim versions are rejected. An interrupted batch is rerun from its
    seed because it has not been committed. One process keeps memory bounded.

    Writes full_results.json after every batch and root-refinement evaluation,
    then threshold_summary.json, ler_vs_p.png, and crossing_drift.png.
    Finite-size crossings and analytic asymptotic conclusions are separate.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'; z is always open.")
    gauge_group = (stacked_honeycomb_open_gauge_group if boundary == "open"
                   else stacked_honeycomb_torus_gauge_group)
    sizes = [tuple(_integer(v, "lattice dimension", 1) for v in size) for size in code_sizes]
    if len(sizes) < 2 or any(len(size) != 3 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Provide at least two distinct (Lx,Ly,Lz) triples, in comparison order.")
    values = sorted(float(p) for p in p_values)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("Provide at least two distinct physical p values.")
    weights = tuple(float(w) for w in noise_weights)
    channels = [pauli_probabilities(p, weights) for p in values]
    shots = _integer(shots, "shots", 1)
    batch_shots = min(shots, _integer(batch_shots, "batch_shots", 1))
    max_cosets = _integer(max_cosets, "max_cosets", 1)
    seed = _integer(seed, "seed")
    crossing_tolerance = float(crossing_tolerance)
    if not math.isfinite(crossing_tolerance) or not 0 < crossing_tolerance < 1:
        raise ValueError("crossing_tolerance must be finite and between 0 and 1.")
    config = {
        "boundary": boundary,
        "code_sizes": [list(size) for size in sizes], "p_values": values,
        "shots_per_point": shots, "noise_weights": list(weights), "seed": seed,
        "batch_shots": batch_shots, "max_cosets": max_cosets,
        "crossing_tolerance": crossing_tolerance, "refine_crossings": bool(refine_crossings),
    }
    if output_dir is None:
        output_dir = "stacked_results_open" if boundary == "open" else "stacked_results"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "full_results.json"
    _study_log(f"Output directory: {output_dir}", progress)
    _study_log(f"Boundary: {boundary.upper()} honeycomb layers; z is OPEN.", progress)
    _study_log(f"{len(sizes)} codes x {len(values)} p values x {shots:,} shots = "
               f"{len(sizes)*len(values)*shots:,} total shots", progress)
    _study_log("Noise: p=px+py+pz; perfect stabilizer syndrome and correction.", progress)

    analyzed, code_records = [], []
    for index, size in enumerate(sizes):
        n, gauges = gauge_group(*size)
        code = analyze_code(n, gauges)
        if code.k == 0:
            raise ValueError(f"{size} with {boundary} layers has no protected logical qubits; "
                             "periodic layers require Ly>=2 for this study.")
        cosets = 1 << code.coset_bits
        if cosets > max_cosets:
            raise ValueError(f"{size}: exact decoder requires {cosets:,} cosets, "
                             f"above max_cosets={max_cosets:,}.")
        analyzed.append(code)
        code_records.append({
            "size": list(size), **code.summary(), "gauge_generators": gauges,
            "cosets": cosets, "bacon_shor_grid": _bacon_shor_component_grid(n, gauges),
            "results": [{
                "p": p, "px": ch[0], "py": ch[1], "pz": ch[2],
                "shots": 0, "failures": 0, "batches_completed": 0, "batch_seeds": [],
                "logical_error_rate": None, "confidence_interval_95": None,
                "exact_logical_error_rate": None, "complete": False,
                "decoder_build_seconds": 0.0, "sampling_seconds": 0.0,
            } for p, ch in zip(values, channels)],
        })
        _study_log(f"CODE [{index+1}/{len(sizes)}] {size}: n={n}, s={code.s}, "
                   f"r={code.r}, k={code.k}, ML cosets={cosets:,}", progress)

    versions = {"python": platform.python_version(), "numpy": np.__version__, "stim": stim.__version__}
    if json_path.exists():
        if not resume:
            raise FileExistsError(f"{json_path} exists; use resume=True or a new output directory.")
        study = json.loads(json_path.read_text(encoding="utf-8"))
        saved_config = dict(study.get("config", {}))
        saved_config.setdefault("boundary", "periodic")
        if saved_config["boundary"] != boundary:
            raise ValueError(f"Saved run uses boundary={saved_config['boundary']!r}; "
                             f"requested {boundary!r}. Choose another output_dir.")
        if study.get("schema") != "stacked_ler_study_v1" or saved_config != config:
            raise ValueError("Saved run settings differ. Use the original settings or a new output_dir.")
        if any(study["versions"].get(key) != versions[key] for key in ("numpy", "stim")):
            raise ValueError("Saved NumPy/Stim versions differ; use a new output_dir.")
        if len(study["codes"]) != len(code_records):
            raise ValueError("Saved code list is inconsistent.")
        for index, record in enumerate(study["codes"]):
            expected = code_records[index]
            for key in ("size", "n", "gauge_generators", "stabilizer_generators", "cosets"):
                if record.get(key) != expected[key]:
                    raise ValueError(f"Saved code metadata differs for code {index}.")
            if len(record["results"]) != len(values):
                raise ValueError("Saved p list is inconsistent.")
            for p_index, point in enumerate(record["results"]):
                for key in ("p", "px", "py", "pz"):
                    if point.get(key) != expected["results"][p_index][key]:
                        raise ValueError("Saved noise parameters are inconsistent.")
                count = _integer(point["shots"], "saved shots")
                failures = _integer(point["failures"], "saved failures")
                batches = _integer(point["batches_completed"], "saved batches")
                seeds = [_study_batch_seed(seed, index, p_index, b) for b in range(batches)]
                if (not 0 <= failures <= count <= shots or count != min(shots, batches*batch_shots)
                        or seeds != point["batch_seeds"] or bool(point["complete"]) != (count == shots)):
                    raise ValueError("Saved batch counts or seeds are inconsistent.")
                if count == shots and point["exact_logical_error_rate"] is None:
                    raise ValueError("Saved completed point has no exact rate.")
        study["config"] = config  # Upgrade validated legacy periodic checkpoints.
        _study_log("RESUME: validated checkpoint; completed batches will be skipped.", progress)
    else:
        study = {
            "schema": "stacked_ler_study_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False, "simulation_complete": False,
            "config": config, "versions": versions, "codes": code_records,
            "noise": {"p_definition": "p=px+py+pz",
                      "fractions_X_Y_Z": list(pauli_probabilities(1, weights)),
                      "model": "IID single-qubit Pauli, code capacity",
                      "syndrome_and_correction": "perfect",
                      "success": "C E belongs to the gauge group, modulo global phase"},
            "extra_exact_evaluations": [], "threshold_report": None,
        }
        save_results_json(study, json_path)

    total_shots = len(sizes)*len(values)*shots
    done_shots = sum(p["shots"] for c in study["codes"] for p in c["results"])
    initial_shots = done_shots
    start_time = time.perf_counter()
    _study_log(f"Already completed: {done_shots:,}/{total_shots:,} shots "
               f"({100*done_shots/total_shots:.1f}%).", progress)
    _study_log("Threshold report distinguishes finite-size crossings from the asymptotic limit.", progress)

    def build_decoder(index: int, p: float, prefix: str) -> ExactGaugeMLDecoder:
        last_print = -math.inf

        def callback(stage: str, completed: int, total: int) -> None:
            nonlocal last_print
            now = time.perf_counter()
            if completed == total or now-last_print >= 5:
                _study_log(f"{prefix} decoder {stage}: {completed}/{total} "
                           f"({100*completed/total:.0f}%)", progress)
                last_print = now

        return ExactGaugeMLDecoder(analyzed[index], *pauli_probabilities(p, weights),
                                   max_cosets=max_cosets, progress_callback=callback)

    try:
        for index, (size, code) in enumerate(zip(sizes, analyzed)):
            for p_index, point in enumerate(study["codes"][index]["results"]):
                prefix = f"[code {index+1}/{len(sizes)} {size}; p {p_index+1}/{len(values)}={point['p']:.6g}]"
                if point["complete"]:
                    _study_log(f"{prefix} SKIP completed point: LER={point['logical_error_rate']:.6g}", progress)
                    continue
                _study_log(f"{prefix} START; completed shots {point['shots']:,}/{shots:,}", progress)
                t0 = time.perf_counter()
                decoder = build_decoder(index, point["p"], prefix)
                point["decoder_build_seconds"] += time.perf_counter()-t0
                point["exact_logical_error_rate"] = decoder.exact_logical_error_rate
                save_results_json(study, json_path)
                _study_log(f"{prefix} decoder ready; exact LER={decoder.exact_logical_error_rate:.8g}", progress)
                while point["shots"] < shots:
                    count = min(batch_shots, shots-point["shots"])
                    batch_index = point["batches_completed"]
                    batch_seed = _study_batch_seed(seed, index, p_index, batch_index)
                    t0 = time.perf_counter()
                    result = estimate_logical_error_rate(
                        code, point["px"], point["py"], point["pz"], count,
                        decoder=decoder, seed=batch_seed, batch_size=count)
                    elapsed = time.perf_counter()-t0
                    # Commit a complete batch together, so Ctrl+C cannot leave
                    # counts and the corresponding seed out of step.
                    new_count = point["shots"]+count
                    new_failures = point["failures"]+result.failures
                    point.update({
                        "shots": new_count, "failures": new_failures,
                        "batches_completed": batch_index+1,
                        "batch_seeds": point["batch_seeds"]+[batch_seed],
                        "sampling_seconds": point["sampling_seconds"]+elapsed,
                        "logical_error_rate": new_failures/new_count,
                        "confidence_interval_95": list(_wilson_interval(new_failures, new_count)),
                        "complete": new_count == shots,
                    })
                    done_shots += count
                    save_results_json(study, json_path)
                    lo, hi = point["confidence_interval_95"]
                    speed = count/max(elapsed, 1e-12)
                    eta = (shots-point["shots"])/speed
                    _study_log(
                        f"{prefix} {point['shots']:,}/{shots:,} shots | "
                        f"failures={point['failures']:,} | LER={point['logical_error_rate']:.6g} "
                        f"(95% CI {lo:.6g}..{hi:.6g}) | {speed:,.0f} shots/s | "
                        f"point ETA {_duration(eta)} | overall shots {100*done_shots/total_shots:.1f}%",
                        progress)
                del decoder
        study["simulation_complete"] = True
        save_results_json(study, json_path)
        _study_log("SIMULATION COMPLETE. Locating exact finite-size curve crossings.", progress)

        cache = {(index, point["p"]): point["exact_logical_error_rate"]
                 for index, code in enumerate(study["codes"]) for point in code["results"]}
        cache.update({(item["code_index"], item["p"]): item["exact_logical_error_rate"]
                      for item in study["extra_exact_evaluations"]})

        def exact_rate(index: int, p: float) -> float:
            key = (index, p)
            if key not in cache:
                prefix = f"[crossing refinement, {sizes[index]}, p={p:.7f}]"
                decoder = build_decoder(index, p, prefix)
                cache[key] = decoder.exact_logical_error_rate
                study["extra_exact_evaluations"].append({
                    "code_index": index, "p": p, "exact_logical_error_rate": cache[key]})
                del decoder
                save_results_json(study, json_path)
            return cache[key]

        report = analyze_stacked_crossings(
            study, exact_rate=exact_rate if refine_crossings else None,
            tolerance=crossing_tolerance, progress=progress)
        study["threshold_report"] = report
        save_results_json(report, output_dir / "threshold_summary.json")
        _study_log("PLOTTING: LER comparison and crossing drift.", progress)
        figure_paths = plot_stacked_ler_study(study, report, output_dir)
        study["complete"] = True
        study["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_results_json(study, json_path)
        _study_log(f"DONE in {_duration(time.perf_counter()-start_time)} this session; "
                   f"{done_shots-initial_shots:,} new shots.", progress)
        _study_log(f"Asymptotic threshold: {report['asymptotic_threshold']['p_threshold']} "
                   f"({report['asymptotic_threshold']['status']}).", progress)
        for path in (json_path, output_dir / "threshold_summary.json", *figure_paths):
            _study_log(f"Saved: {path}", progress)
        return study
    except KeyboardInterrupt:
        save_results_json(study, json_path)
        _study_log("INTERRUPTED: completed batches are saved. Rerun with the same settings to resume.", progress)
        raise


def run_ler_sweep(
    n: int, G: Iterable[str], p_values: Iterable[float], *,
    shots: int = 100_000, noise_weights: Iterable[float] = (1, 1, 1),
    seed: int | None = 12345, batch_size: int = 4096,
    max_cosets: int = 1 << 20, json_path: str | Path | None = None,
    progress: bool = True,
) -> dict:
    """Calculate LER at precisely the supplied p values and optionally save JSON.

    p=px+py+pz. The exact ML decoder is rebuilt for each channel. Each p uses a
    separately spawned RNG seed, recorded in the JSON along with all settings.
    Results retain input order. The plot function sorts only for display.

    JSON is checkpointed after every completed p. If interrupted, complete=False
    and completed_points identify the partial result; there is no auto-resume.
    An existing output JSON at the selected path is replaced.
    """
    if isinstance(G, str):
        raise ValueError("G must be a sequence of strings.")
    gauges = list(G)
    code = analyze_code(n, gauges)
    shots = _integer(shots, "shots", 1)
    batch_size = _integer(batch_size, "batch_size", 1)
    max_cosets = _integer(max_cosets, "max_cosets", 1)
    if seed is not None:
        seed = _integer(seed, "seed")
    if isinstance(p_values, (str, bytes)):
        raise ValueError("p_values must be a nonempty sequence of numbers.")
    values = [float(p) for p in p_values]
    if not values:
        raise ValueError("p_values must not be empty.")
    weights = tuple(float(w) for w in noise_weights)
    channels = [pauli_probabilities(p, weights) for p in values]
    if (1 << code.coset_bits) > max_cosets:
        raise ValueError(f"Exact decoding needs 2**{code.coset_bits} cosets; "
                         f"max_cosets={max_cosets:,}. Choose a smaller code "
                         "or explicitly increase the cap if memory permits.")
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [int(child.generate_state(1, dtype=np.uint64)[0])
                   for child in seed_sequence.spawn(len(values))]
    results = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "completed_points": 0,
        "code": {**code.summary(), "gauge_generators": gauges},
        "noise": {
            "model": "independent_single_qubit_pauli",
            "p_definition": "p = px + py + pz",
            "weights_X_Y_Z": list(weights),
            "fractions_X_Y_Z": list(pauli_probabilities(1.0, weights)),
        },
        "simulation": {
            "model": "code_capacity",
            "syndrome_and_correction": "perfect",
            "failure_definition": "C E not in G, modulo global phase",
            "rate_definition": "logical block error: any protected logical qubit",
            "shots_per_p": shots,
            "base_seed": int(seed_sequence.entropy),
            "batch_size": batch_size,
            "decoder": "ExactGaugeMLDecoder",
            "max_cosets": max_cosets,
            "confidence_interval": "95% Wilson binomial interval",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "stim_version": stim.__version__,
        },
        "p_values": values,
        "results": [],
    }
    if progress:
        print("Code:", code.summary(), flush=True)
    if json_path is not None:
        save_results_json(results, json_path)
    for index, (p, channel, shot_seed) in enumerate(zip(values, channels, child_seeds)):
        px, py, pz = channel
        result = estimate_logical_error_rate(code, px, py, pz, shots,
                                             seed=shot_seed, batch_size=batch_size,
                                             max_cosets=max_cosets)
        results["results"].append({"p": p, "px": px, "py": py, "pz": pz,
                                   "seed": shot_seed, **asdict(result)})
        results["completed_points"] = index + 1
        results["complete"] = index + 1 == len(values)
        if json_path is not None:
            save_results_json(results, json_path)
        if progress:
            print(f"[{index+1}/{len(values)}] p={p:.6g}  "
                  f"LER={result.logical_error_rate:.6g}  "
                  f"failures={result.failures}/{shots}  "
                  f"exact={result.exact_logical_error_rate:.6g}", flush=True)
    return results


def plot_ler_results(
    results: dict | str | Path, path: str | Path = "ler_vs_p.png", *,
    scale: str = "auto", show_exact: bool = True, show: bool = False,
    title: str | None = None,
) -> Path:
    """Plot saved JSON (or a sweep dictionary), with 95% Wilson error bars.

    scale='auto' uses log-log axes if all p>0, otherwise linear axes so p=0 is
    retained. scale='log' requires all p>0. On log axes zero-failure samples are
    shown as upper-limit arrows at the Wilson upper endpoint, not fake positive
    LER estimates. Zero exact rates cannot be shown on log axes and are noted.
    Partial checkpoint files can also be plotted and are labeled incomplete.
    show=False saves a figure without needing a desktop/display server.
    """
    if isinstance(results, (str, Path)):
        with Path(results).open(encoding="utf-8") as stream:
            results = json.load(stream)
    if scale not in {"auto", "linear", "log"}:
        raise ValueError("scale must be 'auto', 'linear', or 'log'.")
    rows = sorted(results["results"], key=lambda row: row["p"])
    if not rows:
        raise ValueError("No completed p values are available to plot.")
    p = np.array([row["p"] for row in rows], dtype=float)
    ler = np.array([row["logical_error_rate"] for row in rows], dtype=float)
    intervals = np.array([row["confidence_interval_95"] for row in rows], dtype=float)
    exact = np.array([np.nan if row.get("exact_logical_error_rate") is None
                      else row["exact_logical_error_rate"] for row in rows])
    use_log = scale == "log" or (scale == "auto" and np.all(p > 0))
    if use_log and np.any(p <= 0):
        raise ValueError("Logarithmic p axes require p>0; use 'auto' or 'linear'.")

    if show:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(7.4, 5.3), layout="constrained")
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(7.4, 5.3), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()

    blue = "#2463a2"
    if use_log:
        axes.set_xscale("log")
        axes.set_yscale("log")
    visible = ler > 0 if use_log else np.ones(len(ler), dtype=bool)
    if np.any(visible):
        errors = np.maximum(0.0, np.vstack((ler - intervals[:, 0], intervals[:, 1] - ler)))
        axes.errorbar(p[visible], ler[visible], yerr=errors[:, visible], fmt="o",
                      color=blue, markerfacecolor="white", markeredgewidth=1.5,
                      markersize=6, capsize=3, elinewidth=1.25,
                      label="Stim Monte Carlo (95% CI)", zorder=3)
    zeros = ~visible
    if np.any(zeros):
        upper = intervals[zeros, 1]
        axes.errorbar(p[zeros], upper, yerr=0.35*upper, uplims=True, fmt="_",
                      color=blue, capsize=3, markersize=7,
                      label="0 failures: 95% CI upper endpoint", zorder=3)
    if show_exact:
        model = np.where((exact > 0) if use_log else np.isfinite(exact), exact, np.nan)
        if np.any(np.isfinite(model)):
            axes.plot(p, model, "--", marker=".", markersize=4, linewidth=1.5,
                      color="#222f3e", label="Exact LER", zorder=2)
        if use_log and np.any(exact == 0):
            axes.text(0.02, 0.02, "Exact zeros omitted on log axes",
                      transform=axes.transAxes, fontsize=8, color="#525d68")
    code = results["code"]
    heading = title or "Logical error rate vs physical error probability"
    if not results.get("complete", True):
        heading += " (partial sweep)"
    shots = results["simulation"]["shots_per_p"]
    weights = results["noise"]["weights_X_Y_Z"]
    bias = ":".join(f"{w:g}" for w in weights)
    figure.suptitle(heading, fontsize=13, fontweight="semibold")
    axes.set_title(f"n={code['n']}, k={code['logical_qubits']}, r={code['gauge_qubits']}"
                   f"  |  {shots:,} shots per p  |  X:Y:Z = {bias}",
                   fontsize=9.5, color="#525d68", pad=12)
    axes.set_xlabel(r"Physical error probability $p=p_x+p_y+p_z$", fontsize=11)
    axes.set_ylabel(r"Logical block error probability $P_{\mathrm{L}}$", fontsize=11)
    axes.grid(which="major", alpha=0.25, linewidth=0.7)
    axes.grid(which="minor", alpha=0.10, linewidth=0.5)
    axes.spines[["top", "right"]].set_visible(False)
    if not use_log:
        axes.set_ylim(bottom=0)
    axes.legend(loc="best", frameon=False, fontsize=9)
    axes.margins(x=0.06)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white")
    if show:
        plt.show()
        plt.close(figure)
    else:
        figure.clear()
    return path
