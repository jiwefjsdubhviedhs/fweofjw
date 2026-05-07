import pulp

# ========== DATA ==========
zones = ["Stage", "Booth", "Fair", "Entrance"]
waste = [1100, 1000, 700, 300]
max_bins_per_zone = [5, 10, 10, 5]      # space limits

bin_caps = [100, 170, 240]              # liters
bin_costs = [150, 280, 400]             # Php
type_names = ["S", "M", "L"]

n_zones = len(zones)
n_types = len(bin_caps)
MAX_BINS = 20   # upper bound per type per zone

# ---------- Helper 1: Minimum cost for full coverage (cheapest) ----------
def min_cost_full_coverage():
    prob = pulp.LpProblem("MinCost_Full", pulp.LpMinimize)
    bins = pulp.LpVariable.dicts("bins",
                                 ((i, k) for i in range(n_zones) for k in range(n_types)),
                                 lowBound=0, upBound=MAX_BINS, cat='Integer')
    total_cost = pulp.lpSum(bins[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
    total_bins = pulp.lpSum(bins[i, k] for i in range(n_zones) for k in range(n_types))
    prob.objective = total_cost
    for i in range(n_zones):
        cap = pulp.lpSum(bin_caps[k] * bins[i, k] for k in range(n_types))
        prob += cap >= waste[i]
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) >= 1
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) <= max_bins_per_zone[i]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        return None, None, None
    bins_sol = [[int(bins[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
    return bins_sol, int(total_bins.value()), total_cost.value()

# ---------- Helper 2: Maximise number of bins subject to budget (full coverage) ----------
def max_bins_with_budget(budget):
    prob = pulp.LpProblem("MaxBins_Budget", pulp.LpMaximize)
    bins = pulp.LpVariable.dicts("bins",
                                 ((i, k) for i in range(n_zones) for k in range(n_types)),
                                 lowBound=0, upBound=MAX_BINS, cat='Integer')
    total_cost = pulp.lpSum(bins[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
    total_bins = pulp.lpSum(bins[i, k] for i in range(n_zones) for k in range(n_types))
    prob.objective = total_bins   # maximise bin count
    for i in range(n_zones):
        cap = pulp.lpSum(bin_caps[k] * bins[i, k] for k in range(n_types))
        prob += cap >= waste[i]
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) >= 1
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) <= max_bins_per_zone[i]
    prob += total_cost <= budget
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        return None, None, None
    bins_sol = [[int(bins[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
    return bins_sol, int(total_bins.value()), total_cost.value()

# ---------- Helper 3: Partial coverage – maximise number of zones fully covered ----------
def max_coverage_with_budget(budget):
    prob = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
    bins = pulp.LpVariable.dicts("b",
                                 ((i, k) for i in range(n_zones) for k in range(n_types)),
                                 lowBound=0, upBound=MAX_BINS, cat='Integer')
    covered = pulp.LpVariable.dicts("cov", range(n_zones), cat='Binary')
    total_cost = pulp.lpSum(bins[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
    total_bins = pulp.lpSum(bins[i, k] for i in range(n_zones) for k in range(n_types))
    M = 10000
    for i in range(n_zones):
        capacity = pulp.lpSum(bin_caps[k] * bins[i, k] for k in range(n_types))
        prob += capacity >= waste[i] - M * (1 - covered[i])
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) <= max_bins_per_zone[i]
    prob += total_cost <= budget
    prob.objective = pulp.lpSum(covered[i] for i in range(n_zones))
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status == pulp.LpStatusOptimal:
        n_covered = sum(covered[i].varValue for i in range(n_zones))
        bins_sol = [[int(bins[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
        return bins_sol, int(total_bins.value()), int(n_covered)
    return None, None, None

# ========== PHASE I ==========
print("=" * 60)

bins_opt, total_bins_opt, min_cost = min_cost_full_coverage()
if bins_opt is None:
    print("No feasible solution")
    exit()
print(f"Minimum total cost: Php {min_cost:.2f}")
print(f"Total number of bins used: {total_bins_opt}")
print("Bin placement:")
for i, zone in enumerate(zones):
    parts = []
    for k, name in enumerate(type_names):
        cnt = bins_opt[i][k]
        if cnt > 0:
            parts.append(f"{cnt}{name}")
    print(f"  {zone}: {', '.join(parts)}")

# ========== PHASE III ==========
print("\n" + "=" * 60)


budget_list = [2000, 3000, 4000, 4880, 5000, 6000, 8000, 10000]

# Table header (no Actual Cost column)
print(f"{'Budget (Php)':>12} | {'# of Bins':>9} | {'Coverage':>12} | {'Bins Used (Zone: S,M,L)'}")
print("-" * 80)

for B in budget_list:
    if B == min_cost:
        # Exactly minimum budget: use cheapest solution
        bins_sol, tot_bins, _ = bins_opt, total_bins_opt, min_cost
        coverage = "Full"
        bin_str = []
        for i, zone in enumerate(zones):
            parts = []
            for k, name in enumerate(type_names):
                cnt = bins_sol[i][k]
                if cnt > 0:
                    parts.append(f"{cnt}{name}")
            bin_str.append(f"{zone}: {','.join(parts)}")
        print(f"{B:12.0f} | {tot_bins:9} | {coverage:12} | {'; '.join(bin_str)}")
    elif B > min_cost:
        # Budget above minimum: maximise number of bins
        bins_sol, tot_bins, actual_cost = max_bins_with_budget(B)
        if bins_sol is not None:
            coverage = "Full"
            bin_str = []
            for i, zone in enumerate(zones):
                parts = []
                for k, name in enumerate(type_names):
                    cnt = bins_sol[i][k]
                    if cnt > 0:
                        parts.append(f"{cnt}{name}")
                bin_str.append(f"{zone}: {','.join(parts)}")
            print(f"{B:12.0f} | {tot_bins:9} | {coverage:12} | {'; '.join(bin_str)}")
        else:
            print(f"{B:12.0f} | {'N/A':9} | {'Infeasible':12} | No feasible solution")
    else:
        # Budget below minimum: partial coverage (maximise zones)
        bins_part, tot_bins_part, n_covered = max_coverage_with_budget(B)
        if bins_part is not None:
            coverage = f"Partial ({n_covered}/4 zones)"
            bin_str = []
            for i, zone in enumerate(zones):
                parts = []
                for k, name in enumerate(type_names):
                    cnt = bins_part[i][k]
                    if cnt > 0:
                        parts.append(f"{cnt}{name}")
                bin_str.append(f"{zone}: {','.join(parts)}")
            print(f"{B:12.0f} | {tot_bins_part:9} | {coverage:12} | {'; '.join(bin_str)}")
        else:
            print(f"{B:12.0f} | {'N/A':9} | {'Infeasible':12} | No feasible solution")

print("\nNote: For budgets above the minimum (4880), the solver maximises the number of bins")
print("      while still achieving full coverage, thus using the extra budget.")
print("Per‑zone bin limits: Stage 5, Booth 10, Fair 10, Entrance 5.")