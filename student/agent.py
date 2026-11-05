"""Expectimax agent for 2048: deterministic, pure Python, single thread, no clock.

A board is one 64-bit int (16 nibbles = tile exponents, 0 = empty). Slides and the
heuristic are 16-bit row look-up tables built once at import time.
MAX node = best legal slide; chance node = average over every empty cell
(2 with p=0.9, 4 with p=0.1); branches rarer than CPROB_MIN stop expanding.
Leaves are scored right after the player's slide.
Effort per move is a node budget set by the number of empty cells (iterative deepening).
"""

from __future__ import annotations

from game2048.observation import Observation
from game2048.types import Move

# ---------------------------------------------------------------- row tables
MAX_EXP = 15  # a nibble holds at most 15 -> 32768


def _slide_left(cells: list[int]) -> list[int]:
    vals = [v for v in cells if v]
    out: list[int] = []
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(min(vals[i] + 1, MAX_EXP))
            i += 2
        else:
            out.append(vals[i])
            i += 1
    return out + [0] * (4 - len(out))


EMPTY_WEIGHT, MERGE_WEIGHT = 270.0, 700.0    # empty cells; equal neighbours (merge chance)
MONO_WEIGHT, MONO_POWER = 47.0, 4.0          # penalty for lines not sorted in one direction
MERGE_REWARD = 5.0                           # weight on game-score points earned by merging (bigger merge = more)
BASE = 200000.0                              # constant offset keeping live boards above DEAD


def _build_tables():
    pw_mono = [float(r) ** MONO_POWER for r in range(16)]
    # game-score points a tile of exponent r is worth: (r-1)*2^r; the gain of a merge is the
    # difference of this between boards, so rewarding it = rewarding merges (bigger = more)
    pts = [0.0, 0.0] + [(r - 1) * 2.0 ** r for r in range(2, 16)]
    left, right, empty_bits, shape = [0] * 65536, [0] * 65536, [()] * 65536, [0.0] * 65536
    for row in range(65536):
        ranks = [(row >> (4 * i)) & 15 for i in range(4)]
        o = _slide_left(ranks)
        left[row] = o[0] | (o[1] << 4) | (o[2] << 8) | (o[3] << 12)
        r = _slide_left(ranks[::-1])
        right[row] = r[3] | (r[2] << 4) | (r[1] << 8) | (r[0] << 12)
        empty_bits[row] = tuple(1 << (4 * i) for i in range(4) if ranks[i] == 0)

        total, merges, prev, counter = 0.0, 0, 0, 0
        for v in ranks:
            total += pts[v]
            if v:
                if prev == v:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = v
        if counter > 0:
            merges += 1 + counter
        mono_l = mono_r = 0.0
        for i in range(1, 4):
            if ranks[i - 1] > ranks[i]:
                mono_l += pw_mono[ranks[i - 1]] - pw_mono[ranks[i]]
            else:
                mono_r += pw_mono[ranks[i]] - pw_mono[ranks[i - 1]]
        shape[row] = (
            BASE
            + EMPTY_WEIGHT * len(empty_bits[row])
            + MERGE_WEIGHT * merges
            - MONO_WEIGHT * (mono_l if mono_l < mono_r else mono_r)
            + MERGE_REWARD * total
        )
    return left, right, empty_bits, shape


LEFT, RIGHT, EMPTY_BITS, SHAPE = _build_tables()


def _transpose(x: int) -> int:
    a = (x & 0xF0F00F0FF0F00F0F) | ((x & 0x0000F0F00000F0F0) << 12) | ((x & 0x0F0F00000F0F0000) >> 12)
    return (a & 0xFF00FF0000FF00FF) | ((a & 0x00FF00FF00000000) >> 24) | ((a & 0x00000000FF00FF00) << 24)


def _slide(b: int, t: list[int]) -> int:
    return t[b & 0xFFFF] | (t[(b >> 16) & 0xFFFF] << 16) | (t[(b >> 32) & 0xFFFF] << 32) | (t[b >> 48] << 48)


def _evaluate(b: int) -> float:
    S = SHAPE
    t = _transpose(b)
    return (
        S[b & 0xFFFF] + S[(b >> 16) & 0xFFFF] + S[(b >> 32) & 0xFFFF] + S[b >> 48]
        + S[t & 0xFFFF] + S[(t >> 16) & 0xFFFF] + S[(t >> 32) & 0xFFFF] + S[t >> 48]
    )


# ---------------------------------------------------------------- search
DEAD = -1.0e6
MAX_DEPTH = 8
CPROB_MIN = 1e-3
MOVE_ORDER = (Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN)


class _OutOfBudget(Exception):
    pass


class Agent:
    name = "expectimax-2048"

    def __init__(self) -> None:
        self._nodes = 0
        self._cap = 0
        self._cache: dict[int, float] = {}
        self.last_nodes = 0  # telemetry read by the benchmark
        self.last_completed_depth = 0

    def reset(self, seed: int | None = None) -> None:
        pass  # stateless between moves

    def _chance(self, b: int, k: int, cp: float) -> float:
        """Board right after the player's slide; a random tile is about to appear."""
        self._nodes += 1
        if k == 0 or cp < CPROB_MIN:
            return _evaluate(b)
        key = (b << 4) | k
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self._nodes > self._cap:
            raise _OutOfBudget
        cells = [bit << s for s in (0, 16, 32, 48) for bit in EMPTY_BITS[(b >> s) & 0xFFFF]]
        n = len(cells)  # >= 1: a legal slide always leaves an empty cell
        p2, p4 = cp * 0.9 / n, cp * 0.1 / n
        mx = self._max
        total = 0.0
        for bit in cells:
            total += 0.9 * mx(b | bit, k, p2) + 0.1 * mx(b | (bit << 1), k, p4)
        val = total / n
        self._cache[key] = val
        return val

    def _max(self, b: int, k: int, cp: float) -> float:
        """Player to move with k moves left (UP, LEFT, RIGHT, DOWN inlined for speed)."""
        L, R, ch, km = LEFT, RIGHT, self._chance, k - 1
        best, found = DEAD, False
        t = _transpose(b)
        nt = L[t & 0xFFFF] | (L[(t >> 16) & 0xFFFF] << 16) | (L[(t >> 32) & 0xFFFF] << 32) | (L[t >> 48] << 48)
        if nt != t:  # UP
            best, found = ch(_transpose(nt), km, cp), True
        nb = L[b & 0xFFFF] | (L[(b >> 16) & 0xFFFF] << 16) | (L[(b >> 32) & 0xFFFF] << 32) | (L[b >> 48] << 48)
        if nb != b:  # LEFT
            v = ch(nb, km, cp)
            if not found or v > best:
                best, found = v, True
        nb = R[b & 0xFFFF] | (R[(b >> 16) & 0xFFFF] << 16) | (R[(b >> 32) & 0xFFFF] << 32) | (R[b >> 48] << 48)
        if nb != b:  # RIGHT
            v = ch(nb, km, cp)
            if not found or v > best:
                best, found = v, True
        nt = R[t & 0xFFFF] | (R[(t >> 16) & 0xFFFF] << 16) | (R[(t >> 32) & 0xFFFF] << 32) | (R[t >> 48] << 48)
        if nt != t:  # DOWN
            v = ch(_transpose(nt), km, cp)
            if not found or v > best:
                best, found = v, True
        return best if found else DEAD

    def _root(self, after: list[tuple[Move, int]], depth: int) -> Move:
        best_move, best_val = after[0][0], float("-inf")
        for move, nb in after:
            v = self._chance(nb, depth - 1, 1.0)
            if v > best_val:
                best_val, best_move = v, move
        return best_move

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        legal = obs.legal_moves
        if len(legal) == 1:
            self.last_nodes = self.last_completed_depth = 0
            return legal[0]

        b = 0  # read-only conversion of obs.board (tile values -> exponents)
        for i, v in enumerate(obs.board.ravel().tolist()):
            if v:
                b |= min(int(v).bit_length() - 1, MAX_EXP) << (4 * i)
        slide = {
            Move.UP: lambda: _transpose(_slide(_transpose(b), LEFT)),
            Move.DOWN: lambda: _transpose(_slide(_transpose(b), RIGHT)),
            Move.LEFT: lambda: _slide(b, LEFT),
            Move.RIGHT: lambda: _slide(b, RIGHT),
        }
        after = [(m, slide[m]()) for m in MOVE_ORDER if m in legal]

        empties = sum(1 for i in range(16) if not (b >> (4 * i)) & 15)
        budget = 7000 if empties <= 3 else 4000 if empties <= 5 else 1250 if empties <= 8 else 300

        # depth 1 always finishes (uncapped); then deepen while the next depth is predicted to fit
        self._cache, self._nodes, self._cap = {}, 0, 1 << 60
        best_move = self._root(after, 1)
        completed, total = 1, self._nodes
        last = max(1, self._nodes)
        prev = last
        for depth in range(2, MAX_DEPTH + 1):
            growth = 6.0 if depth == 2 else max(3.0, last / prev)
            if total + last * growth > budget:
                break
            self._cache, self._nodes, self._cap = {}, 0, budget - total
            try:
                move = self._root(after, depth)
            except _OutOfBudget:
                total += self._nodes
                break
            best_move, completed = move, depth
            total += self._nodes
            prev, last = last, max(1, self._nodes)
        self.last_nodes, self.last_completed_depth = total, completed
        return best_move
