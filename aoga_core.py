import math
import random
import numpy as np
import networkx as nx

# Default parameters
WAIT_TIME_WEIGHT = 2
FIRST_TRANSFER_PENALTY = 15
SECOND_TRANSFER_PENALTY = 20
NUM_ROUTES = 10

def compute_utility(waiting_time, in_vehicle_time, x1t, x2t,
                    Cwt=WAIT_TIME_WEIGHT, C1t=FIRST_TRANSFER_PENALTY, C2t=SECOND_TRANSFER_PENALTY):
    return waiting_time * Cwt + in_vehicle_time + x1t * C1t + x2t * C2t

def softmax_utilities(utilities):
    exp_util = np.exp(utilities)
    probs = exp_util / np.sum(exp_util)
    return probs

class TransitOption:
    def __init__(self, path_segments, route_ids, waiting_time, in_vehicle_time, x1t, x2t):
        self.path_segments = path_segments
        self.route_ids = route_ids
        self.waiting_time = waiting_time
        self.in_vehicle_time = in_vehicle_time
        self.x1t = x1t
        self.x2t = x2t
        self.utility = compute_utility(waiting_time, in_vehicle_time, x1t, x2t)

def estimate_waiting_time(frequencies):
    if not frequencies:
        return float('inf')
    total_freq = sum(frequencies)
    return 30 / total_freq if total_freq > 0 else float('inf')

def assign_demand_for_od(o, d, routes, route_freqs, g):
    options = []
    for rid, route in enumerate(routes):
        if o in route and d in route:
            idx_o, idx_d = route.index(o), route.index(d)
            if idx_o != idx_d:
                seg = route[min(idx_o, idx_d):max(idx_o, idx_d)+1]
                edges = [(seg[i], seg[i+1]) for i in range(len(seg)-1)]
                travel_time = sum(g[u][v]['weight'] for (u, v) in edges)
                wt = estimate_waiting_time([route_freqs[rid]])
                options.append(TransitOption(edges, (rid,), wt, travel_time, 0, 0))
    for r1_id, r1 in enumerate(routes):
        if o not in r1:
            continue
        for r2_id, r2 in enumerate(routes):
            if r2_id == r1_id or d not in r2:
                continue
            for t in set(r1) & set(r2):
                try:
                    seg1 = r1[r1.index(o):r1.index(t)+1]
                    seg2 = r2[r2.index(t):r2.index(d)+1]
                    edges1 = [(seg1[i], seg1[i+1]) for i in range(len(seg1)-1)]
                    edges2 = [(seg2[i], seg2[i+1]) for i in range(len(seg2)-1)]
                    travel_time = sum(g[u][v]['weight'] for u, v in edges1 + edges2)
                    wt_total = estimate_waiting_time([route_freqs[r1_id]]) + estimate_waiting_time([route_freqs[r2_id]])
                    options.append(TransitOption(edges1 + edges2, (r1_id, r2_id), wt_total, travel_time, 1, 0))
                except:
                    continue
    for r1_id, r1 in enumerate(routes):
        if o not in r1:
            continue
        for r2_id, r2 in enumerate(routes):
            if r2_id == r1_id:
                continue
            for r3_id, r3 in enumerate(routes):
                if r3_id in (r1_id, r2_id) or d not in r3:
                    continue
                for t1 in set(r1) & set(r2):
                    for t2 in set(r2) & set(r3):
                        try:
                            seg1 = r1[r1.index(o):r1.index(t1)+1]
                            seg2 = r2[r2.index(t1):r2.index(t2)+1]
                            seg3 = r3[r3.index(t2):r3.index(d)+1]
                            edges = [(seg1[i], seg1[i+1]) for i in range(len(seg1)-1)] +                                     [(seg2[i], seg2[i+1]) for i in range(len(seg2)-1)] +                                     [(seg3[i], seg3[i+1]) for i in range(len(seg3)-1)]
                            travel_time = sum(g[u][v]['weight'] for u, v in edges)
                            wt_total = (estimate_waiting_time([route_freqs[r1_id]]) +
                                        estimate_waiting_time([route_freqs[r2_id]]) +
                                        estimate_waiting_time([route_freqs[r3_id]]))
                            options.append(TransitOption(edges, (r1_id, r2_id, r3_id), wt_total, travel_time, 1, 1))
                        except:
                            continue
    if not options:
        return None
    utilities = np.array([opt.utility for opt in options])
    probs = softmax_utilities(-utilities)
    return np.random.choice(options, p=probs)

def mutate_individual(individual, od_routes_dict, g, num_routes=NUM_ROUTES):
    mutated = individual.copy()
    for _ in range(3):
        i = random.randint(0, num_routes - 1)
        route = mutated[i]
        found_od = None
        for od, route_list in od_routes_dict.items():
            if any(route == r[0] for r in route_list):
                found_od = od
                break
        if not found_od:
            continue
        alternatives = [r[0] for r in od_routes_dict[found_od] if r[0] != route]
        random.shuffle(alternatives)
        for new_route in alternatives:
            mutated[i] = new_route
            if is_feasible(mutated, g):
                return mutated
        mutated[i] = route
    return individual

def is_feasible(individual, g):
    sub_g = nx.Graph()
    for route in individual:
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            sub_g.add_edge(u, v)
    if not nx.is_connected(sub_g):
        return False
    for route in individual:
        if len(route) != len(set(route)):
            return False
    covered = set()
    for route in individual:
        covered.update(route)
    if covered != set(g.nodes()):
        return False
    seen = set()
    for route in individual:
        norm = tuple(route)
        rev = tuple(reversed(route))
        if norm in seen or rev in seen:
            return False
        seen.add(norm)
    return True
