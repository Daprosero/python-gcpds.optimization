"""CL-RPINN (Closure-Layer Regularized Physics-Informed Neural Network) for the gas network: every hard constraint holds by construction.

This module extends the RPINN gas model in ``model_real_data_`` without
modifying it. The network makes the economic decision (raw outputs, trained); three
closures embedded as layers turn it into a physically exact state:

1. ``BalancedFlows`` (proposal Ec. 3-4): the decision arcs -- sources,
   compressors and unsupplied gas per node and demand class -- are scaled
   sigmoids shifted by one dual ``nu`` per zone (the virtual supply node is
   the reference, ``nu = 0``). ``nu`` minimises a strictly convex function
   whose stationarity is the balance of every zone, found by damped
   (Levenberg-Marquardt) Newton. Pipes stay inside a zone, so they cancel
   from the zone balance and are not part of this solve.
2. ``PipeFlows`` (Ec. 5-6): what each node still needs is delivered by the
   pipes. Any balanced split ``f0`` plus a rotation ``C t`` around each loop
   keeps the balance; ``t`` minimises ``sum |f|^3 / (3K)``, so the Weymouth
   drops around every loop add up to zero. A tree has no ``t``.
   ``CapacityScale`` then multiplies all served gas by one factor
   ``theta <= 1`` when a pipe limit or the pressure limits cannot hold; the
   problem is homogeneous (flows scale by ``theta``, drops by ``theta^2``),
   so balance and Weymouth stay exact and the gas not served becomes
   unsupplied.
3. ``ZonePressures`` (Ec. 7-8): squared pressures follow from the Weymouth
   drops up to one level per compressor-separated zone; each level is a
   scaled sigmoid inside an interval that respects pressure limits,
   compression ratios and every downstream zone. Linear, no iteration.

Each iterative solve runs without gradient and ends with one differentiable
Newton step that carries the implicit-function gradient. The training loss is
the normalised operating cost ``c^T F / (max(c) * sum z)`` alone.

Scope: the network must be connected with at least one source whose minimum
injection is zero, each zone fed by at most one compressor and compressors
forming no loops.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import BatchNormalization, Dense, Input
from scipy.linalg import null_space
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tensorflow.keras import regularizers

from gcpds.optimization.model_real_data_ import FixedWeightsConstraint, gen_w

def _default_data_dir() -> Path:
    env = os.environ.get("CLRPINN_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "CLRPINN" / "Data"


#: Networks (``Redes``) and, per scenario ``S``/``M``/``D``, the inputs and the IPOPT reference (``MAPE/*_P.mat``).
DATA_DIR = _default_data_dir()

#: Demand classes in the column order of the ``node.demcost`` sheet.
DEMAND_CLASSES = ("Res", "Ind", "Com", "NGV", "Ref", "Pet")
_SHEETS = {"inf": "node.info", "dc": "node.demcost", "well": "well", "pipe": "pipe",
           "comp": "comp", "sto": "sto"}


def read_network(path):
    """Read the sheets of a network file that the gas model uses."""
    return {k: pd.read_excel(path, sheet_name=v) for k, v in _SHEETS.items()}


def augmented_network(sheets, classes):
    """Incidence and static bounds of the augmented graph.

    Arcs are ordered ``[sources, pipes, compressors, unsupplied per class]``;
    the unsupplied arcs follow the demand input layout (one block of ``N``
    nodes per class). Row 0 of the incidence is the virtual node ``V``; rows
    ``1..N`` are the network nodes. Returns ``(A, lo, hi)`` where ``hi`` holds
    NaN for the unsupplied arcs, whose upper bound is each sample's demand.
    """
    w = gen_w(sheets["inf"], sheets["well"], sheets["pipe"], sheets["comp"], sheets["dc"], sheets["sto"])
    N = w.shape[0]
    nw, npi, nc = len(sheets["well"]), len(sheets["pipe"]), len(sheets["comp"])
    C = len(classes)
    real = np.hstack([w[:, :nw + npi + nc], np.tile(np.eye(N), (1, C))])
    virtual = np.concatenate([-np.ones(nw), np.zeros(npi + nc), -np.ones(N * C)])
    A = np.vstack([virtual, real])
    lo = np.concatenate([sheets["well"]["Imin"].values, sheets["pipe"]["Fg_min"].values,
                         np.zeros(nc), np.zeros(N * C)]).astype("float64")
    hi = np.concatenate([sheets["well"]["Imax"].values, sheets["pipe"]["Fg_max"].values,
                         sheets["comp"]["fmaxc"].values, np.full(N * C, np.nan)]).astype("float64")
    return A, lo, hi


class BalancedFlows(tf.keras.layers.Layer):
    """Arc flows inside their bounds that close every balance exactly.

    Inputs are ``(raw, z)``: one raw value per arc and the demand per node
    and class. ``groups`` (k x N, optional) sums node balances into group
    balances -- one row per zone -- and gives one ``nu`` per group; without it
    every node has its own ``nu``. The solve runs in float64; the output is
    float32.
    """

    def __init__(self, incidence, lo, hi, iterations=100, groups=None, max_step=5.0, **kwargs):
        super().__init__(**kwargs)
        self.incidence = np.asarray(incidence, dtype="float64")
        self.lo = np.asarray(lo, dtype="float64")
        self.hi = np.asarray(hi, dtype="float64")
        self.iterations = int(iterations)
        n = self.incidence.shape[0] - 1
        self.groups = np.eye(n) if groups is None else np.asarray(groups, dtype="float64")
        self.max_step = float(max_step)
        self.n_static = int(np.sum(~np.isnan(self.hi)))

    def _bounds(self, z):
        batch = tf.shape(z)[0]
        static = tf.broadcast_to(tf.constant(self.hi[:self.n_static]), [batch, self.n_static])
        hi = tf.concat([static, z], axis=-1)
        lo = tf.broadcast_to(tf.constant(self.lo), tf.shape(hi))
        return lo, hi - lo

    def call(self, inputs):
        raw = tf.cast(inputs[0], tf.float64)
        z = tf.cast(inputs[1], tf.float64)
        M = tf.constant(self.groups)
        Ar = tf.matmul(M, tf.constant(self.incidence[1:]))  # lambda_V = 0; one row per group
        N = self.incidence.shape[0] - 1
        k = self.groups.shape[0]
        nodes = tf.add_n(tf.split(z, z.shape[-1] // N, axis=-1))
        b = tf.matmul(nodes, M, transpose_b=True)          # demand of each group
        lo, span = self._bounds(z)
        rhs = b - tf.matmul(lo, Ar, transpose_b=True)
        eye = tf.eye(k, dtype=tf.float64)

        def pieces(r, lam):
            s = tf.sigmoid(r + tf.matmul(lam, Ar))
            grad = tf.matmul(span * s, Ar, transpose_b=True) - rhs
            hess = tf.einsum("ne,be,me->bnm", Ar, span * s * (1.0 - s), Ar)
            return grad, hess

        def objective(r, lam):
            return (tf.reduce_sum(span * tf.math.softplus(r + tf.matmul(lam, Ar)), axis=-1)
                    - tf.reduce_sum(rhs * lam, axis=-1))

        r0 = tf.stop_gradient(raw)
        lo, span, rhs = (tf.stop_gradient(t) for t in (lo, span, rhs))
        lam = tf.zeros_like(rhs)
        mu = tf.fill(tf.shape(rhs)[:1], tf.constant(1e-3, tf.float64))
        radius = tf.fill(tf.shape(rhs)[:1], tf.constant(self.max_step, tf.float64))

        def body(k, lam, mu, radius):
            grad, hess = pieces(r0, lam)
            scale = tf.reduce_max(tf.linalg.diag_part(hess), axis=-1) + 1e-12
            step = -tf.linalg.solve(hess + (mu * scale)[:, None, None] * eye, grad[..., None])[..., 0]
            # Trust region in logit units: a saturated sigmoid makes the Hessian nearly singular.
            # It starts at max_step, doubles after an accepted step and halves after a rejected one.
            size = tf.reduce_max(tf.abs(step), axis=-1)
            step = step * tf.minimum(tf.constant(1.0, tf.float64), radius / tf.maximum(size, 1e-300))[:, None]
            g0, g1 = objective(r0, lam), objective(r0, lam + step)
            # When G no longer changes at machine precision, accept a step that lowers the residual.
            flat = tf.abs(g1 - g0) <= 1e-12 * tf.maximum(tf.constant(1.0, tf.float64), tf.abs(g0))
            closer = (tf.norm(pieces(r0, lam + step)[0], axis=-1) < tf.norm(grad, axis=-1))
            better = (g1 < g0) | (flat & closer)
            lam = tf.where(better[:, None], lam + step, lam)
            mu = tf.where(better, tf.maximum(mu / 3.0, 1e-12), tf.minimum(mu * 4.0, 1e12))
            radius = tf.where(better, tf.minimum(2.0 * radius, 1e4), tf.maximum(0.5 * radius, 1e-8))
            return k + 1, lam, mu, radius

        _, lam, _, _ = tf.while_loop(lambda k, *_: k < self.iterations, body,
                                     (tf.constant(0), lam, mu, radius))
        lam = tf.stop_gradient(lam)
        # One differentiable Newton step: at the solution grad ~ 0, so it leaves
        # the value unchanged and gives d(lambda)/d(raw, z) by the implicit function theorem.
        lo_d, span_d = self._bounds(z)
        rhs_d = b - tf.matmul(lo_d, Ar, transpose_b=True)
        s = tf.sigmoid(raw + tf.matmul(lam, Ar))
        grad = tf.matmul(span_d * s, Ar, transpose_b=True) - rhs_d
        hess = tf.stop_gradient(tf.einsum("ne,be,me->bnm", Ar, span_d * s * (1.0 - s), Ar)) + 1e-9 * eye
        lam = lam - tf.linalg.solve(hess, grad[..., None])[..., 0]
        flows = lo_d + span_d * tf.sigmoid(raw + tf.matmul(lam, Ar))
        return tf.cast(flows, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"incidence": self.incidence.tolist(), "lo": self.lo.tolist(),
                       "hi": [None if np.isnan(v) else v for v in self.hi],
                       "iterations": self.iterations, "groups": self.groups.tolist(),
                       "max_step": self.max_step})
        return config


class FlowVector(tf.keras.layers.Layer):
    """Arrange arc flows in the column order of ``w``: sources, pipes, comps, users, storage.

    Unsupplied blocks of classes the model does not take as input, and the
    storage columns, are zero, as in v1.
    """

    def __init__(self, n_arcs, classes, nodes, n_storage, **kwargs):
        super().__init__(**kwargs)
        self.n_arcs = int(n_arcs)
        self.classes = tuple(classes)
        self.nodes = int(nodes)
        self.n_storage = int(n_storage)

    def call(self, flows):
        head = flows[:, :self.n_arcs]
        unsupplied = flows[:, self.n_arcs:]
        zeros = tf.zeros_like(unsupplied[:, :self.nodes])
        blocks = [unsupplied[:, self.classes.index(k) * self.nodes:(self.classes.index(k) + 1) * self.nodes]
                  if k in self.classes else zeros for k in DEMAND_CLASSES]
        storage = tf.zeros_like(head[:, :1]) * tf.zeros((1, self.n_storage))
        return tf.concat([head] + blocks + [storage], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"n_arcs": self.n_arcs, "classes": list(self.classes),
                       "nodes": self.nodes, "n_storage": self.n_storage})
        return config



def pipe_structure(sheets):
    """Pipe incidence pieces for ``PipeFlows``: column ranges, pseudo-inverse and loop basis."""
    w = gen_w(sheets["inf"], sheets["well"], sheets["pipe"], sheets["comp"], sheets["dc"], sheets["sto"])
    nw, npi = len(sheets["well"]), len(sheets["pipe"])
    Bq = w[:, nw:nw + npi]
    return {"Bq": Bq, "pinv": np.linalg.pinv(Bq), "loops": null_space(Bq),
            "k": sheets["pipe"]["Kij"].values.astype("float64")}


class PipeFlows(tf.keras.layers.Layer):
    """Minimum-energy pipe flows: the unique Weymouth-consistent split of the balance.

    Input: ``rhs`` (batch, N), what the pipes must deliver to each node once
    sources, compressors and unsupplied gas are fixed. Output: pipe flows
    ``f = f0 + Z t`` with ``t`` minimising ``sum |f|^3 / (3k)``; on a tree there
    is no ``t`` and the flows follow from the balance alone.
    """

    def __init__(self, pinv, loops, k, iterations=50, **kwargs):
        super().__init__(**kwargs)
        self.pinv = np.asarray(pinv, dtype="float64")
        self.loops = np.asarray(loops, dtype="float64").reshape(self.pinv.shape[0], -1)
        self.k = np.asarray(k, dtype="float64")
        self.iterations = int(iterations)

    def call(self, rhs):
        rhs = tf.cast(rhs, tf.float64)
        f0 = tf.matmul(rhs, tf.constant(self.pinv), transpose_b=True)
        if self.loops.shape[1] == 0:
            return tf.cast(f0, tf.float32)
        Z, k = tf.constant(self.loops), tf.constant(self.k)
        L = self.loops.shape[1]

        def grad_hess(f):
            g = tf.matmul(f * tf.abs(f) / k, Z)
            h = tf.einsum("el,be,em->blm", Z, 2.0 * tf.abs(f) / k, Z) + 1e-9 * tf.eye(L, dtype=tf.float64)
            return g, h

        def energy(f):
            return tf.reduce_sum(tf.abs(f) ** 3 / (3.0 * k), axis=-1)

        f0s = tf.stop_gradient(f0)

        def body(i, t):
            f = f0s + tf.matmul(t, Z, transpose_b=True)
            g, h = grad_hess(f)
            step = -tf.linalg.solve(h, g[..., None])[..., 0]
            alpha = tf.ones_like(step[:, :1])
            for _ in range(20):  # backtracking keeps every step a descent step
                worse = energy(f0s + tf.matmul(t + alpha * step, Z, transpose_b=True)) > energy(f)
                alpha = tf.where(worse[:, None], 0.5 * alpha, alpha)
            return i + 1, t + alpha * step

        _, t = tf.while_loop(lambda i, _: i < self.iterations, body,
                             (tf.constant(0), tf.zeros_like(f0[:, :L])))
        t = tf.stop_gradient(t)
        f = f0 + tf.matmul(t, Z, transpose_b=True)
        g, h = grad_hess(f)
        t = t - tf.linalg.solve(tf.stop_gradient(h), g[..., None])[..., 0]  # implicit gradient
        return tf.cast(f0 + tf.matmul(t, Z, transpose_b=True), tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"pinv": self.pinv.tolist(), "loops": self.loops.tolist(),
                       "k": self.k.tolist(), "iterations": self.iterations})
        return config


def zone_structure(sheets):
    """Compressor-separated zones, the offset map from pipe drops and the zone order."""
    inf, pipe, comp = sheets["inf"], sheets["pipe"], sheets["comp"]
    N = len(inf)
    i, j = pipe["fnode"].values.astype(int) - 1, pipe["tnode"].values.astype(int) - 1
    n_zones, zone = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(N, N)), directed=False)
    offset = np.zeros((N, len(pipe)))           # d = offset @ drop, drop_q = psi_i - psi_j
    seen = np.zeros(N, bool)
    for z in range(n_zones):
        root = int(np.where(zone == z)[0][0]); seen[root] = True; queue = [root]
        while queue:
            n = queue.pop()
            for q, (a, b) in enumerate(zip(i, j)):
                for here, there, sign in ((a, b, -1.0), (b, a, 1.0)):
                    if here == n and not seen[there]:
                        offset[there] = offset[n]; offset[there, q] += sign
                        seen[there] = True; queue.append(there)
    feeds = []
    fed = set()
    for a, b, r in zip(comp["fnode"].values.astype(int) - 1, comp["tnode"].values.astype(int) - 1,
                       comp["ratio"].values):
        if zone[b] in fed:
            raise ValueError(f"zone of node {b + 1} is fed by more than one compressor")
        fed.add(zone[b]); feeds.append((int(zone[a]), int(zone[b]), int(a), int(b), float(r) ** 2))
    order, stack = [], [z for z in range(n_zones) if z not in fed]
    while stack:
        z = stack.pop(); order.append(z)
        stack += [zb for za, zb, *_ in feeds if za == z]
    if len(order) != n_zones:
        raise ValueError("compressors form a loop between zones")
    return {"zone": zone, "offset": offset, "feeds": feeds, "order": order,
            "lo2": inf["Pmin"].values.astype("float64") ** 2,
            "hi2": inf["Pmax"].values.astype("float64") ** 2}


def _zone_intervals(d, zone, feeds, order, lo2, hi2):
    """Level interval of every zone for offsets ``d``, leaving room for every downstream zone."""
    lo, hi = {}, {}
    for z in range(int(zone.max()) + 1):
        idx = np.where(zone == z)[0]
        dz = tf.gather(d, idx, axis=1)
        lo[z] = tf.reduce_max(tf.constant(lo2[idx]) - dz, axis=1)
        hi[z] = tf.reduce_min(tf.constant(hi2[idx]) - dz, axis=1)
    for z in reversed(order):                              # backward: room downstream
        for za, zb, a, b, r2 in feeds:
            if za == z:
                lo[z] = tf.maximum(lo[z], (lo[zb] + d[:, b]) / r2 - d[:, a])
                hi[z] = tf.minimum(hi[z], hi[zb] + d[:, b] - d[:, a])
    return lo, hi


class ZonePressures(tf.keras.layers.Layer):
    """Node pressures with the Weymouth drops, pressure limits and compression windows exact.

    Inputs ``(f_pipes, raw)``: pipe flows and one raw output per zone. A
    backward pass over the zones computes the level interval that leaves room
    downstream; a forward pass picks each level with a scaled sigmoid inside
    it and the window its feeding compressor allows.
    """

    def __init__(self, zone, offset, feeds, order, lo2, hi2, k, **kwargs):
        super().__init__(**kwargs)
        self.zone = np.asarray(zone, dtype="int64")
        self.offset = np.asarray(offset, dtype="float64")
        self.feeds = [tuple(f) for f in feeds]
        self.order = [int(z) for z in order]
        self.lo2 = np.asarray(lo2, dtype="float64")
        self.hi2 = np.asarray(hi2, dtype="float64")
        self.k = np.asarray(k, dtype="float64")

    def call(self, inputs):
        f = tf.cast(inputs[0], tf.float64)
        raw = tf.cast(inputs[1], tf.float64)
        d = tf.matmul(f * tf.abs(f) / tf.constant(self.k), tf.constant(self.offset), transpose_b=True)
        lo, hi = _zone_intervals(d, self.zone, self.feeds, self.order, self.lo2, self.hi2)
        level = {}
        for z in self.order:                               # forward: sigmoid in the window
            l, u = lo[z], hi[z]
            for za, zb, a, b, r2 in self.feeds:
                if zb == z:
                    psi_a = level[za] + d[:, a]
                    l, u = tf.maximum(l, psi_a - d[:, b]), tf.minimum(u, r2 * psi_a - d[:, b])
            u = tf.maximum(u, l)                           # an empty window marks an infeasible instance
            level[z] = l + (u - l) * tf.sigmoid(raw[:, z])
        levels = tf.stack([level[z] for z in range(len(level))], axis=1)
        psi = tf.gather(levels, self.zone, axis=1) + d
        return tf.cast(tf.sqrt(tf.maximum(psi, 0.0)), tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"zone": self.zone.tolist(), "offset": self.offset.tolist(),
                       "feeds": [list(f) for f in self.feeds], "order": self.order,
                       "lo2": self.lo2.tolist(), "hi2": self.hi2.tolist(), "k": self.k.tolist()})
        return config


class PipeDemand(tf.keras.layers.Layer):
    """What the pipes must deliver to each node: demand minus the decision arcs' injections.

    The decision arcs close every zone balance, so this sums to zero over each zone.
    """

    def __init__(self, decision_incidence, classes, nodes, **kwargs):
        super().__init__(**kwargs)
        self.decision = np.asarray(decision_incidence, dtype="float64")
        self.classes = tuple(classes)
        self.nodes = int(nodes)

    def call(self, inputs):
        arcs, z = tf.cast(inputs[0], tf.float64), tf.cast(inputs[1], tf.float64)
        injected = tf.matmul(arcs, tf.constant(self.decision), transpose_b=True)
        demand = tf.add_n(tf.split(z, len(self.classes), axis=-1))
        return tf.cast(demand - injected, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"decision_incidence": self.decision.tolist(), "classes": list(self.classes),
                       "nodes": self.nodes})
        return config


class CapacityScale(tf.keras.layers.Layer):
    """Scale all served gas by one factor ``theta`` so pipe and pressure limits hold.

    Inputs ``(decisions, pipes, z)``: decision arcs ``[sources, comps,
    unsupplied per class]``, pipe flows and demand. Scaling every injection by
    ``theta`` scales every pipe flow by ``theta`` and every Weymouth drop by
    ``theta^2``, so balance and Weymouth stay exact; ``theta = 0`` (nothing
    served) is always feasible. ``theta`` is the largest value at most 1 with
    every pipe inside its limits and every zone level interval non-empty.
    The pipe part is a closed-form minimum; the pressure part, found by
    bisection on ``theta^2`` (feasibility is monotone in it), carries no
    gradient. Outputs ``(decisions, pipes, theta)`` after scaling; the gas not
    served is added to the unsupplied arcs.
    """

    def __init__(self, f_max, f_min, k, zone, offset, feeds, order, lo2, hi2, n_served,
                 bisections=50, **kwargs):
        super().__init__(**kwargs)
        self.f_max = np.abs(np.asarray(f_max, dtype="float64"))
        self.f_min = np.abs(np.asarray(f_min, dtype="float64"))
        self.k = np.asarray(k, dtype="float64")
        self.zone = np.asarray(zone, dtype="int64")
        self.offset = np.asarray(offset, dtype="float64")
        self.feeds = [tuple(f) for f in feeds]
        self.order = [int(z) for z in order]
        self.lo2 = np.asarray(lo2, dtype="float64")
        self.hi2 = np.asarray(hi2, dtype="float64")
        self.n_served = int(n_served)
        self.bisections = int(bisections)

    def _feasible(self, drop, s):
        d = s[:, None] * tf.matmul(drop, tf.constant(self.offset), transpose_b=True)
        lo, hi = _zone_intervals(d, self.zone, self.feeds, self.order, self.lo2, self.hi2)
        tol = 1e-9 * float(self.hi2.max())
        return tf.reduce_all(tf.stack([lo[z] <= hi[z] + tol for z in lo], axis=1), axis=1)

    def call(self, inputs):
        x, f, z = (tf.cast(t, tf.float64) for t in inputs)
        lim = tf.where(f >= 0, tf.constant(self.f_max), tf.constant(self.f_min))
        ratio = lim / tf.maximum(tf.abs(f), 1e-12)
        theta_q = tf.minimum(tf.constant(1.0, tf.float64), tf.reduce_min(tf.concat([ratio, tf.ones_like(ratio[:, :1])], 1), axis=1))
        drop = tf.stop_gradient(f * tf.abs(f) / tf.constant(self.k))
        top = tf.stop_gradient(theta_q) ** 2
        fits = self._feasible(drop, top)
        a, b = tf.zeros_like(top), top
        for _ in range(self.bisections):
            m = 0.5 * (a + b)
            ok = self._feasible(drop, m)
            a, b = tf.where(ok, m, a), tf.where(ok, b, m)
        theta = tf.where(fits, theta_q, tf.stop_gradient(tf.sqrt(a)))[:, None]
        served, unsupplied = x[:, :self.n_served], x[:, self.n_served:]
        unsupplied = z - theta * (z - unsupplied)
        return (tf.cast(tf.concat([theta * served, unsupplied], 1), tf.float32),
                tf.cast(theta * f, tf.float32), tf.cast(theta, tf.float32))

    def get_config(self):
        config = super().get_config()
        config.update({"f_max": self.f_max.tolist(), "f_min": self.f_min.tolist(), "k": self.k.tolist(),
                       "zone": self.zone.tolist(), "offset": self.offset.tolist(),
                       "feeds": [list(f) for f in self.feeds], "order": self.order,
                       "lo2": self.lo2.tolist(), "hi2": self.hi2.tolist(),
                       "n_served": self.n_served, "bisections": self.bisections})
        return config


class AssembleArcs(tf.keras.layers.Layer):
    """Arc flows in augmented order ``[sources, pipes, comps, unsupplied]``."""

    def __init__(self, n_sources, **kwargs):
        super().__init__(**kwargs)
        self.n_sources = int(n_sources)

    def call(self, inputs):
        decisions, pipes = inputs
        return tf.concat([decisions[:, :self.n_sources], pipes, decisions[:, self.n_sources:]], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"n_sources": self.n_sources})
        return config


def cost_vector(sheets):
    """Operating cost per column of ``F`` (sources, pipes, comps, users, storage)."""
    sto = sheets["sto"]
    return np.concatenate([sheets["well"]["Cg"].values, sheets["pipe"]["C_O"].values,
                           sheets["comp"]["costc"].values,
                           *(sheets["dc"][f"al_{k}"].values for k in DEMAND_CLASSES),
                           (sto["C_S+"] - sto["C_V"]).values,
                           (-1 * (sto["C_S-"] - sto["C_V"])).values]).astype("float32")


def mean_prediction(y_true, y_pred):
    """Loss for an output that is itself the quantity to minimise."""
    return tf.reduce_mean(y_pred)


def flow_model_v2(path, fd=None, a=1e-4, b=1e-4, c=1e-4, seed=1, classes=("Res",), iterations=200):
    """Build and compile the v2 gas model.

    The input holds the demand of each class in ``classes`` as consecutive
    blocks of one value per node (``N * len(classes)`` columns). Outputs:

    * ``F``: every flow in the column order of ``w`` (sources, pipes, comps,
      unsupplied per class, storage at zero).
    * ``P``: node pressures.
    * ``Cost``: ``c^T F / (max(c) * sum z)``, the only loss.

    ``fd`` is accepted for signature parity with v1 and unused. ``fit`` needs a
    target for ``Cost`` only (zeros work).
    """
    unknown = [k for k in classes if k not in DEMAND_CLASSES]
    if not classes or unknown or len(set(classes)) != len(classes):
        raise ValueError(f"classes must be distinct names from {DEMAND_CLASSES}, got {classes!r}")
    sh = read_network(path)
    if (sh["well"]["Imin"].values > 0).any():
        raise ValueError("capacity scaling needs every source minimum Imin to be zero")
    w = gen_w(sh["inf"], sh["well"], sh["pipe"], sh["comp"], sh["dc"], sh["sto"])
    N, C = w.shape[0], len(classes)
    nw, npi, nc = len(sh["well"]), len(sh["pipe"]), len(sh["comp"])
    A, lo, hi = augmented_network(sh, classes)
    pipes, zones = pipe_structure(sh), zone_structure(sh)
    tf.keras.utils.set_random_seed(seed)
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    In = Input(shape=(N * C,))
    x = Dense(w.shape[1] * 4, "selu", kernel_regularizer=regularizers.L1L2(a, a))(In)
    raw_sources = Dense(nw, kernel_initializer=initializer, name="raw_sources")(BatchNormalization()(x))
    raw_comps = Dense(nc, kernel_initializer=initializer, name="raw_comps")(BatchNormalization()(x))
    xt = Dense(N * C, "selu", kernel_regularizer=regularizers.L1L2(b, b), kernel_initializer=initializer)(x)
    raw_unsupplied = BatchNormalization(name="raw_unsupplied")(xt)
    raw = tf.keras.layers.Concatenate(name="raw")([raw_sources, raw_comps, raw_unsupplied])

    # Solver 1: one nu per zone closes every zone balance with the decision arcs.
    dec = [c for c in range(A.shape[1]) if not nw <= c < nw + npi]
    groups = np.zeros((int(zones["zone"].max()) + 1, N))
    groups[zones["zone"], np.arange(N)] = 1.0
    decisions = BalancedFlows(A[:, dec], lo[dec], hi[dec], iterations=iterations, groups=groups,
                              name="decisions")([raw, In])
    # Solver 2: the pipes deliver what each node still needs, rotated around each loop by Weymouth.
    rhs = PipeDemand(A[1:, dec], classes, N, name="pipe_demand")([decisions, In])
    f_pipes = PipeFlows(pipes["pinv"], pipes["loops"], pipes["k"], name="pipe_flows")(rhs)
    decisions, f_pipes, _ = CapacityScale(sh["pipe"]["Fg_max"].values, sh["pipe"]["Fg_min"].values, pipes["k"],
                                          zones["zone"], zones["offset"], zones["feeds"], zones["order"],
                                          zones["lo2"], zones["hi2"], nw + nc,
                                          name="capacity")([decisions, f_pipes, In])
    arcs = AssembleArcs(nw, name="arcs")([decisions, f_pipes])
    F = FlowVector(nw + npi + nc, classes, N, 2 * len(sh["sto"]), name="F")(arcs)

    x1 = BatchNormalization()(In)
    for _ in range(3):
        x1 = Dense(w.shape[1] * 4, "selu", kernel_regularizer=regularizers.L1L2(c, c),
                   kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
    raw_levels = Dense(int(zones["zone"].max()) + 1, kernel_initializer=initializer, name="raw_levels")(x1)
    P = ZonePressures(zones["zone"], zones["offset"], zones["feeds"], zones["order"],
                      zones["lo2"], zones["hi2"], pipes["k"], name="P")([f_pipes, raw_levels])

    cost = cost_vector(sh)
    scale = float(cost.max())
    total = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=-1, keepdims=True), name="total_demand")(In)
    spent = Dense(1, use_bias=False, trainable=False, name="spent",
                  kernel_initializer=tf.keras.initializers.Constant(cost.reshape(-1, 1)))(F)
    Out_cost = tf.keras.layers.Lambda(lambda t: t[0] / (scale * tf.maximum(t[1], 1e-6)), name="Cost")([spent, total])

    model = Model(In, {"F": F, "P": P, "Cost": Out_cost})
    model.compile(loss={"Cost": mean_prediction}, optimizer=tf.keras.optimizers.Adamax(1e-2))
    return model


def node_balance_residual(path, F, z, classes=("Res",)):
    """Largest absolute node-balance residual ``|w F - z|`` per sample."""
    sh = read_network(path)
    w = gen_w(sh["inf"], sh["well"], sh["pipe"], sh["comp"], sh["dc"], sh["sto"])
    N = w.shape[0]
    nodes = np.asarray(z).reshape(len(z), len(classes), N).sum(axis=1)
    return np.abs(np.asarray(F) @ w.T - nodes).max(axis=1)
