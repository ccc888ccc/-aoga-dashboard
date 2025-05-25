import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import math
import argparse

# 預設參數（可被命令列覆寫）
NUM_ROUTES = 10
POP_SIZE = 20
MAX_GEN = 50
MAX_ROUTE_LEN = 10
BUS_CAPACITY = 40
LOAD_FACTOR = 1.25
WAIT_TIME_WEIGHT = 2
FIRST_TRANSFER_PENALTY = 15
SECOND_TRANSFER_PENALTY = 20

# 命令列參數解析
parser = argparse.ArgumentParser()
parser.add_argument('--od', type=str, default='uploaded_od.csv')
parser.add_argument('--edge', type=str, default='uploaded_edges.csv')
parser.add_argument('--detour_min', type=float, default=1.0)
parser.add_argument('--detour_max', type=float, default=1.5)
parser.add_argument('--capacity', type=int, default=40)
parser.add_argument('--load_factor', type=float, default=1.25)
args = parser.parse_args()

BUS_CAPACITY = args.capacity
LOAD_FACTOR = args.load_factor

# 建立圖
g = nx.Graph()
edges_df = pd.read_csv(args.edge)
for _, row in edges_df.iterrows():
    g.add_edge(int(row['from']), int(row['to']), weight=float(row['weight']))

NUM_NODES = len(g.nodes())

# 讀入 OD
od_df = pd.read_csv(args.od)
OD = {(i + 1, j + 1): od_df.iloc[i, j] for i in range(od_df.shape[0]) for j in range(od_df.shape[1]) if i != j and od_df.iloc[i, j] > 0}

def generate_route_db(detour_factor):
    db = []
    for i, j in combinations(g.nodes(), 2):
        try:
            path = nx.shortest_path(g, source=i, target=j, weight='weight')
            weight = nx.path_weight(g, path, weight='weight')
            if len(path) <= MAX_ROUTE_LEN and weight <= detour_factor * nx.shortest_path_length(g, source=i, target=j, weight='weight'):
                db.append((tuple(path), weight))
        except:
            continue
    return db

def generate_individual(route_db):
    return [r[0] for r in random.sample(route_db, NUM_ROUTES)]

def evaluate(ind):
    user_cost = 0
    pf, rl = defaultdict(int), defaultdict(int)
    t0 = t1 = t2 = un = 0

    for (o, d), demand in OD.items():
        best_cost = float('inf')
        best_path = None
        best_route_ids = ()
        for rid, route in enumerate(ind):
            if o in route and d in route:
                idx_o, idx_d = route.index(o), route.index(d)
                if idx_o != idx_d:
                    seg = route[min(idx_o, idx_d):max(idx_o, idx_d)+1]
                    cost = WAIT_TIME_WEIGHT * 5 + sum(g[seg[i]][seg[i+1]]['weight'] for i in range(len(seg)-1))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = [(seg[i], seg[i+1]) for i in range(len(seg)-1)]
                        best_route_ids = (rid,)
        for r1_id, r1 in enumerate(ind):
            if o not in r1:
                continue
            for r2_id, r2 in enumerate(ind):
                if r1_id == r2_id or d not in r2:
                    continue
                common = set(r1) & set(r2)
                for t in common:
                    try:
                        seg1 = r1[r1.index(o):r1.index(t)+1]
                        seg2 = r2[r2.index(t):r2.index(d)+1]
                        time1 = sum(g[seg1[i]][seg1[i+1]]['weight'] for i in range(len(seg1)-1))
                        time2 = sum(g[seg2[i]][seg2[i+1]]['weight'] for i in range(len(seg2)-1))
                        cost = WAIT_TIME_WEIGHT * 10 + time1 + time2 + FIRST_TRANSFER_PENALTY
                        if cost < best_cost:
                            best_cost = cost
                            best_path = (
                                [(seg1[i], seg1[i+1]) for i in range(len(seg1)-1)] +
                                [(seg2[i], seg2[i+1]) for i in range(len(seg2)-1)]
                            )
                            best_route_ids = (r1_id, r2_id)
                    except:
                        continue
        for r1_id, r1 in enumerate(ind):
            if o not in r1:
                continue
            for r2_id, r2 in enumerate(ind):
                if r2_id in (r1_id,):
                    continue
                for r3_id, r3 in enumerate(ind):
                    if r3_id in (r1_id, r2_id) or d not in r3:
                        continue
                    common12 = set(r1) & set(r2)
                    common23 = set(r2) & set(r3)
                    for t1 in common12:
                        for t2 in common23:
                            try:
                                seg1 = r1[r1.index(o):r1.index(t1)+1]
                                seg2 = r2[r2.index(t1):r2.index(t2)+1]
                                seg3 = r3[r3.index(t2):r3.index(d)+1]
                                time1 = sum(g[seg1[i]][seg1[i+1]]['weight'] for i in range(len(seg1)-1))
                                time2 = sum(g[seg2[i]][seg2[i+1]]['weight'] for i in range(len(seg2)-1))
                                time3 = sum(g[seg3[i]][seg3[i+1]]['weight'] for i in range(len(seg3)-1))
                                cost = (WAIT_TIME_WEIGHT * 15 + time1 + time2 + time3 +
                                        FIRST_TRANSFER_PENALTY + SECOND_TRANSFER_PENALTY)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_path = (
                                        [(seg1[i], seg1[i+1]) for i in range(len(seg1)-1)] +
                                        [(seg2[i], seg2[i+1]) for i in range(len(seg2)-1)] +
                                        [(seg3[i], seg3[i+1]) for i in range(len(seg3)-1)]
                                    )
                                    best_route_ids = (r1_id, r2_id, r3_id)
                            except:
                                continue

        if best_cost == float('inf'):
            best_cost = 100
            un += demand
        else:
            if len(best_route_ids) == 1:
                t0 += demand
            elif len(best_route_ids) == 2:
                t1 += demand
            elif len(best_route_ids) == 3:
                t2 += demand
            for seg in best_path:
                for rid in best_route_ids:
                    pf[(rid, seg)] += demand
        user_cost += best_cost * demand

    for (rid, seg), load in pf.items():
        rl[rid] = max(rl[rid], load)

    fleet = sum(math.ceil(rl[rid] / (BUS_CAPACITY * LOAD_FACTOR)) for rid in rl)
    headways = [60 / (rl[rid] / (BUS_CAPACITY * LOAD_FACTOR) + 1e-6) for rid in rl]
    avg_headway = round(sum(headways) / len(headways), 2) if headways else 0
    max_headway = max(headways) if headways else 0
    num_routes = len(ind)

    return user_cost, fleet, t0, t1, t2, un, avg_headway, max_headway, num_routes

def AOGA(detour):
    route_db = generate_route_db(detour)
    population = [generate_individual(route_db) for _ in range(POP_SIZE)]
    history = []
    use_user_cost = True

    for gen in range(MAX_GEN):
        fitness = [evaluate(ind) for ind in population]
        ranked = sorted(zip(population, fitness), key=lambda x: x[1][0 if use_user_cost else 1])
        use_user_cost = not use_user_cost
        best = min(fitness, key=lambda x: x[0])
        best_index = fitness.index(best)
        best_individual = population[best_index]
        history.append({
            'generation': gen,
            'user_cost': best[0],
            'fleet_size': best[1],
            '0-transfer': best[2],
            '1-transfer': best[3],
            '2-transfer': best[4],
            'unserved': best[5],
            'avg_headway': best[6],
            'max_headway': best[7],
            'num_routes': best[8],
            'detour_factor': detour,
            'routes': str(best_individual)
        })
        parents = [ind for ind, _ in ranked[:POP_SIZE//2]]
        new_pop = []
        while len(new_pop) < POP_SIZE:
            if len(parents) < 2: break
            p1, p2 = random.sample(parents, 2)
            cut = random.randint(1, NUM_ROUTES-1)
            child = p1[:cut] + p2[cut:]
            if random.random() < 0.2:
                child[random.randint(0, NUM_ROUTES-1)] = random.choice(route_db)[0]
            new_pop.append(child)
        population = [ind for ind, _ in ranked[:2]] + new_pop[:POP_SIZE-2]
    return history

if __name__ == '__main__':
    all_results = []
    detour_range = np.arange(args.detour_min, args.detour_max + 0.01, 0.1)
    for DETOUR_FACTOR in detour_range:
        results = AOGA(round(DETOUR_FACTOR, 2))
        for r in results:
            r['detour_factor'] = round(DETOUR_FACTOR, 2)
        all_results.extend(results)
        pd.DataFrame(results).to_csv(f"results_{round(DETOUR_FACTOR,2)}.csv", index=False)
        print(f"Finished DETOUR_FACTOR={round(DETOUR_FACTOR, 2)}")

    pd.DataFrame(all_results).to_csv("results_all.csv", index=False)

    # Optional: visualization preview
    user_costs = [r['user_cost'] for r in all_results]
    fleets = [r['fleet_size'] for r in all_results]
    colors = [r['detour_factor'] for r in all_results]
    plt.scatter(fleets, user_costs, c=colors, cmap='viridis')
    plt.colorbar(label='Detour Factor')
    plt.xlabel('Fleet Size')
    plt.ylabel('User Cost')
    plt.title('Pareto Frontier by Detour Factor')
    plt.grid(True)
    plt.show()
