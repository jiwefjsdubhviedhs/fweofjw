import pulp

zones = ["Stage", "Booth", "Fair", "Entrance"]
waste = [1100, 1000, 700, 300]
n_zones = len(zones)

bin_caps = [100, 170, 240]
bin_costs = [150, 280, 400]
type_names = ["S", "M", "L"]
n_types = len(bin_caps)

MAX_BINS_PER_ZONE = 20

def solve_model(objective_type, extra_constraints=None, budget=None):
    prob = pulp.LpProblem("Waste_Bin_Optimization", pulp.LpMinimize)
    bins = pulp.LpVariable.dicts("bins",
                                 ((i, k) for i in range(n_zones) for k in range(n_types)),
                                 lowBound=0, upBound=MAX_BINS_PER_ZONE, cat='Integer')
    total_cost = pulp.lpSum(bins[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
    total_bins = pulp.lpSum(bins[i, k] for i in range(n_zones) for k in range(n_types))
    if objective_type == 'min_bins':
        prob.objective = total_bins
    else:
        prob.objective = total_cost
    for i in range(n_zones):
        capacity = pulp.lpSum(bin_caps[k] * bins[i, k] for k in range(n_types))
        prob += capacity >= waste[i]
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) >= 1
    if extra_constraints:
        for constr in extra_constraints:
            prob += constr
    if budget is not None:
        prob += total_cost <= budget
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        return None, None, None, None
    bins_sol = [[int(bins[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
    return prob.status, bins_sol, int(total_bins.value()), total_cost.value()

def solve_with_fixed_bins(target_bins):
    prob = pulp.LpProblem("Fixed_Bins_Count", pulp.LpMinimize)
    bins = pulp.LpVariable.dicts("bins",
                                 ((i, k) for i in range(n_zones) for k in range(n_types)),
                                 lowBound=0, upBound=MAX_BINS_PER_ZONE, cat='Integer')
    total_cost = pulp.lpSum(bins[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
    total_bins = pulp.lpSum(bins[i, k] for i in range(n_zones) for k in range(n_types))
    prob.objective = total_cost
    for i in range(n_zones):
        capacity = pulp.lpSum(bin_caps[k] * bins[i, k] for k in range(n_types))
        prob += capacity >= waste[i]
    for i in range(n_zones):
        prob += pulp.lpSum(bins[i, k] for k in range(n_types)) >= 1
    prob += total_bins == target_bins
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        return None, None, None
    bins_sol = [[int(bins[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
    return bins_sol, int(total_bins.value()), total_cost.value()

def format_bins_used(bins_sol):
    parts = []
    for i, loc in enumerate(zones):
        loc_parts = []
        for k, name in enumerate(type_names):
            cnt = bins_sol[i][k]
            if cnt > 0:
                loc_parts.append(f"{cnt}{name}")
        parts.append(f"{loc}: {','.join(loc_parts) if loc_parts else 'none'}")
    return "; ".join(parts)

# Phase I
print("=" * 60)
print("PHASE I: Minimum number of bins")
_, _, min_bins, _ = solve_model('min_bins')
print(f"Minimum number of bins: {min_bins}")

# Phase II
print("\n" + "=" * 60)
print("PHASE II: Minimum cost using exactly that many bins")
bins_fixed, actual_bins, phase2_cost = solve_with_fixed_bins(min_bins)
print(f"Minimum budget (Php): {phase2_cost:.2f}")
print("Placement:", format_bins_used(bins_fixed))
print(f"Total bins: {actual_bins}")

# Phase III
print("\n" + "=" * 60)
print("PHASE III: Budget scenarios")
_, _, _, min_full_cost = solve_model('min_cost')
print(f"Absolute minimum budget for full coverage (all small bins): {min_full_cost:.2f} Php")

budget_list = sorted(set([
    max(0, min_full_cost - 2000),
    max(0, min_full_cost - 1000),
    max(0, min_full_cost - 500),
    min_full_cost,
    phase2_cost,          # Add the Phase II budget explicitly
    min_full_cost + 1000,
    min_full_cost + 3000,
    min_full_cost + 6000
]))

print("\n| Budget (Php) | # of Bins | Coverage (Partial/Full) | Bins Used (Zone: S,M,L) |")
print("|-------------|-----------|-------------------------|-------------------------|")

for B in budget_list:
    if B == phase2_cost and B != min_full_cost:
        # Show the Phase II solution directly (mix of bins)
        print(f"| {B:11.0f} | {actual_bins:9} | Full (Phase II)          | {format_bins_used(bins_fixed)} |")
        continue

    status, bins_sol, tot_bins, cost = solve_model('min_cost', budget=B)
    if status is not None:
        print(f"| {B:11.0f} | {tot_bins:9} | Full                     | {format_bins_used(bins_sol)} |")
    else:
        # Partial coverage (maximize number of zones)
        prob = pulp.LpProblem("Partial", pulp.LpMaximize)
        bins_var = pulp.LpVariable.dicts("b", ((i, k) for i in range(n_zones) for k in range(n_types)),
                                         lowBound=0, upBound=MAX_BINS_PER_ZONE, cat='Integer')
        covered = pulp.LpVariable.dicts("c", range(n_zones), cat='Binary')
        total_cost_expr = pulp.lpSum(bins_var[i, k] * bin_costs[k] for i in range(n_zones) for k in range(n_types))
        total_bins_expr = pulp.lpSum(bins_var[i, k] for i in range(n_zones) for k in range(n_types))
        M = 10000
        for i in range(n_zones):
            capacity = pulp.lpSum(bin_caps[k] * bins_var[i, k] for k in range(n_types))
            prob += capacity >= waste[i] - M * (1 - covered[i])
        prob += total_cost_expr <= B
        prob.objective = pulp.lpSum(covered[i] for i in range(n_zones))
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        if prob.status == pulp.LpStatusOptimal:
            n_covered = sum(covered[i].varValue for i in range(n_zones))
            cov_text = f"Partial ({int(n_covered)}/4 zones)"
            bins_part = [[int(bins_var[i, k].varValue) for k in range(n_types)] for i in range(n_zones)]
            print(f"| {B:11.0f} | {int(total_bins_expr.value()):9} | {cov_text:<23} | {format_bins_used(bins_part)} |")
        else:
            print(f"| {B:11.0f} | {'N/A':9} | Infeasible                | N/A |")
