"""Hand-written search over time-respecting trails (Units I-II).

State: (account, arrival_time_ns). Action: forward money along an outgoing edge
that is strictly later than the arrival and within the hop window. Goal: a known
cash-out account (B-D1). Path cost: hops for the uninformed searches, and
-log(risk of the receiver) per edge (B-D4) for UCS / A*.

Conventions (documented in reports/b_02_search.md):
- BFS / DFS / DLS / IDDFS test the goal when a node is generated; UCS / A* when popped.
- DFS, DLS and IDDFS are tree searches by default: time strictly increases along a
  path, so they cannot loop forever. `graph_search=True` adds a depth-aware visited set.
- `find_all=True` keeps searching and collects one path to every goal reached.
- `peak_stored` (frontier + reached/visited) is the deterministic memory measure;
  `peak_bytes` comes from tracemalloc when `measure_memory=True`.
- A run that hits `max_expansions` is `truncated`; that is never reported as "not found".
- `cutoff` means the depth limit stopped at least one branch (DLS / IDDFS).

This module never reads account labels; the goal set is passed in by the caller.
"""

import heapq
import math
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mulehunt.b_params import DEFAULT_RISK, DLS_LIMIT, MAX_DEPTH, MAX_EXPANSIONS


@dataclass
class SearchResult:
    algorithm: str
    found: bool = False
    goal: str | None = None
    path: list = field(default_factory=list)
    goal_paths: dict = field(default_factory=dict)
    cost: float = math.inf
    nodes_expanded: int = 0
    nodes_generated: int = 0
    peak_stored: int = 0
    elapsed_s: float = 0.0
    peak_bytes: int | None = None
    cutoff: bool = False
    truncated: bool = False

    @property
    def hops(self) -> int | None:
        return len(self.path) if self.found else None


# ---------------------------------------------------------------- costs

class CostFn:
    """Edge cost that depends only on the receiving account, so heuristics can be
    computed per account. Calling it on an Edge returns that edge's cost."""

    def __init__(self, account_cost, name: str):
        self.account_cost = account_cost
        self.name = name

    def __call__(self, edge) -> float:
        return self.account_cost(edge.receiver)


UNIT_COST = CostFn(lambda account: 1.0, "unit")


def make_cost_fn(risk: dict | None, default_risk: float = DEFAULT_RISK) -> CostFn:
    """B-D4: cost of an edge = -log(risk_prob of the receiver)."""
    risk = risk or {}
    return CostFn(lambda account: -math.log(risk.get(account, default_risk)), "neg_log_risk")


def path_cost(path, cost_fn=UNIT_COST) -> float:
    return float(sum(cost_fn(e) for e in path))


# ---------------------------------------------------------------- helpers

def _rebuild(node) -> list:
    """node = (edge, parent_node) linked list -> list of edges from the start."""
    path = []
    while node is not None:
        path.append(node[0])
        node = node[1]
    return path[::-1]


class _Run:
    """Shared bookkeeping: counters, budget, timing and optional tracemalloc."""

    def __init__(self, name, max_expansions, measure_memory):
        self.r = SearchResult(name)
        self.max_expansions = max_expansions
        self.measure_memory = measure_memory

    def __enter__(self):
        if self.measure_memory:
            tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.r.elapsed_s = time.perf_counter() - self.t0
        if self.measure_memory:
            self.r.peak_bytes = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        return False

    def over_budget(self) -> bool:
        if self.max_expansions is not None and self.r.nodes_expanded >= self.max_expansions:
            self.r.truncated = True
            return True
        return False

    def stored(self, n: int) -> None:
        self.r.peak_stored = max(self.r.peak_stored, n)

    def hit(self, goal, path, find_all, cost_fn) -> bool:
        """Record a goal; returns True when the search should stop."""
        if goal not in self.r.goal_paths:
            self.r.goal_paths[goal] = path
        if not self.r.found:
            self.r.found, self.r.goal, self.r.path = True, goal, path
            self.r.cost = path_cost(path, cost_fn)
        return not find_all


def _start_is_goal(run, start, goals, cost_fn) -> bool:
    if start[0] in goals:
        run.r.found, run.r.goal, run.r.path, run.r.cost = True, start[0], [], 0.0
        run.r.goal_paths[start[0]] = []
        run.stored(1)
        return True
    return False


# ---------------------------------------------------------------- uninformed

def bfs(graph, start, goals, window=None, find_all=False, max_expansions=MAX_EXPANSIONS,
        cost_fn=UNIT_COST, measure_memory=False) -> SearchResult:
    """Breadth-first graph search over states; finds the fewest-hop trail."""
    goals = set(goals)
    with _Run("bfs", max_expansions, measure_memory) as run:
        if _start_is_goal(run, start, goals, cost_fn):
            return run.r
        frontier = deque([(start, None)])
        reached = {start}
        run.stored(2)
        while frontier:
            if run.over_budget():
                break
            (account, t), node = frontier.popleft()
            run.r.nodes_expanded += 1
            for e in graph.successors(account, t, window):
                child = (e.receiver, e.time_ns)
                if child in reached:
                    continue
                run.r.nodes_generated += 1
                reached.add(child)
                cnode = (e, node)
                if e.receiver in goals:
                    if run.hit(e.receiver, _rebuild(cnode), find_all, cost_fn):
                        return run.r
                    continue  # a cash-out is terminal
                frontier.append((child, cnode))
            run.stored(len(frontier) + len(reached))
    return run.r


def _depth_first(name, graph, start, goals, limit, window, find_all, max_expansions, cost_fn,
                 graph_search, run=None):
    """Iterative DFS with an optional depth limit. Successors are tried in time order."""
    stack = [(start, 0, None)]
    visited = {start: 0} if graph_search else None
    while stack:
        if run.over_budget():
            break
        (account, t), depth, node = stack.pop()
        succ = graph.successors(account, t, window)
        if limit is not None and depth >= limit:
            if succ:
                run.r.cutoff = True
            continue
        run.r.nodes_expanded += 1
        children = []
        for e in succ:
            child = (e.receiver, e.time_ns)
            if graph_search:
                seen = visited.get(child)
                if seen is not None and seen <= depth + 1:
                    continue
                visited[child] = depth + 1
            run.r.nodes_generated += 1
            cnode = (e, node)
            if e.receiver in goals:
                if run.hit(e.receiver, _rebuild(cnode), find_all, cost_fn):
                    return True
                continue
            children.append((child, depth + 1, cnode))
        stack.extend(reversed(children))  # earliest successor on top
        run.stored(len(stack) + (len(visited) if graph_search else 0))
    return False


def dfs(graph, start, goals, window=None, find_all=False, max_expansions=MAX_EXPANSIONS,
        cost_fn=UNIT_COST, graph_search=False, measure_memory=False) -> SearchResult:
    goals = set(goals)
    with _Run("dfs", max_expansions, measure_memory) as run:
        if not _start_is_goal(run, start, goals, cost_fn):
            _depth_first("dfs", graph, start, goals, None, window, find_all, max_expansions, cost_fn,
                         graph_search, run)
    return run.r


def dls(graph, start, goals, limit=DLS_LIMIT, window=None, find_all=False, max_expansions=MAX_EXPANSIONS,
        cost_fn=UNIT_COST, graph_search=False, measure_memory=False) -> SearchResult:
    """Depth-limited search: trails of at most `limit` hops. `cutoff` = the limit was binding."""
    goals = set(goals)
    with _Run(f"dls({limit})", max_expansions, measure_memory) as run:
        if not _start_is_goal(run, start, goals, cost_fn):
            _depth_first("dls", graph, start, goals, limit, window, find_all, max_expansions, cost_fn,
                         graph_search, run)
    return run.r


def iddfs(graph, start, goals, max_depth=MAX_DEPTH, window=None, find_all=False, max_expansions=MAX_EXPANSIONS,
          cost_fn=UNIT_COST, graph_search=False, measure_memory=False) -> SearchResult:
    """Iterative deepening: DLS with limits 1, 2, ... max_depth; counters accumulate."""
    goals = set(goals)
    with _Run("iddfs", max_expansions, measure_memory) as run:
        if _start_is_goal(run, start, goals, cost_fn):
            return run.r
        for limit in range(1, max_depth + 1):
            run.r.cutoff = False
            _depth_first("iddfs", graph, start, goals, limit, window, find_all, max_expansions, cost_fn,
                         graph_search, run)
            if (run.r.found and not find_all) or run.r.truncated or not run.r.cutoff:
                break
    return run.r


# ---------------------------------------------------------------- informed

def reverse_bfs_heuristic(graph, goals, cost_fn):
    """h1(n) = m * d(n): d = reverse-BFS hops to the nearest goal (time ignored, so
    d never exceeds the true remaining hops); m = cheapest cost into any account that
    can still reach a goal. Accounts with no route get infinity."""
    d = graph.hop_distance_to(goals)
    costs = [cost_fn.account_cost(a) for a in d if graph.in_degree(a) > 0]
    m = min(costs) if costs else 0.0
    h = lambda account: m * d[account] if account in d else math.inf
    h.m, h.d = m, d
    return h


def layered_heuristic(graph, goals, cost_fn):
    """h2(n) = sum_{j<d(n)} L_j, L_j = cheapest cost into an account at distance j.
    Any trail from n must enter every layer d(n)-1, ..., 0, so h2 is admissible; h2 >= h1."""
    d = graph.hop_distance_to(goals)
    depth = max(d.values(), default=0)
    layer = [math.inf] * (depth + 1)
    for a, j in d.items():
        if graph.in_degree(a) > 0:
            layer[j] = min(layer[j], cost_fn.account_cost(a))
    layer = [0.0 if x == math.inf else x for x in layer]  # an empty layer can't be crossed anyway
    prefix = np.concatenate([[0.0], np.cumsum(layer)])
    h = lambda account: float(prefix[d[account]]) if account in d else math.inf
    h.layers, h.d = layer, d
    return h


def _best_first(name, graph, start, goals, cost_fn, h, window, find_all, max_expansions, measure_memory):
    goals = set(goals)
    with _Run(name, max_expansions, measure_memory) as run:
        if _start_is_goal(run, start, goals, cost_fn):
            return run.r
        tie = 0
        frontier = [(h(start[0]), 0.0, tie, start, None)]
        best_g = {start: 0.0}
        closed = set()
        run.stored(2)
        while frontier:
            f, g, _, state, node = heapq.heappop(frontier)
            if state in closed or g > best_g.get(state, math.inf):
                continue
            if state[0] in goals:
                if run.hit(state[0], _rebuild(node), find_all, cost_fn):
                    return run.r
                closed.add(state)
                continue
            if run.over_budget():
                break
            closed.add(state)
            run.r.nodes_expanded += 1
            for e in graph.successors(state[0], state[1], window):
                child = (e.receiver, e.time_ns)
                hc = h(e.receiver)
                if hc == math.inf or child in closed:
                    continue
                g2 = g + cost_fn(e)
                if g2 < best_g.get(child, math.inf):
                    best_g[child] = g2
                    tie += 1
                    run.r.nodes_generated += 1
                    heapq.heappush(frontier, (g2 + hc, g2, tie, child, (e, node)))
            run.stored(len(frontier) + len(closed))
    return run.r


def ucs(graph, start, goals, cost_fn=UNIT_COST, window=None, find_all=False, max_expansions=MAX_EXPANSIONS,
        measure_memory=False) -> SearchResult:
    return _best_first("ucs", graph, start, goals, cost_fn, lambda a: 0.0, window, find_all, max_expansions,
                       measure_memory)


def astar(graph, start, goals, cost_fn=UNIT_COST, heuristic="reverse_bfs", window=None, find_all=False,
          max_expansions=MAX_EXPANSIONS, measure_memory=False) -> SearchResult:
    """A*; `heuristic` is "reverse_bfs" (h1), "layered" (h2), "zero" or a callable account -> h.
    Pass a prebuilt callable when running many searches against the same goals."""
    if heuristic == "reverse_bfs":
        h, name = reverse_bfs_heuristic(graph, goals, cost_fn), "astar(h1)"
    elif heuristic == "layered":
        h, name = layered_heuristic(graph, goals, cost_fn), "astar(h2)"
    elif heuristic == "zero":
        h, name = (lambda a: 0.0), "astar(h0)"
    else:
        h, name = heuristic, f"astar({getattr(heuristic, 'label', 'h')})"
    return _best_first(name, graph, start, goals, cost_fn, h, window, find_all, max_expansions, measure_memory)


ALGORITHMS = {"bfs": bfs, "dfs": dfs, "dls": dls, "iddfs": iddfs, "ucs": ucs, "astar": astar}
