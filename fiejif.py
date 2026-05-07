import pulp
import numpy as np
import pandas as pd

# ============================================
# MODEL ASSUMPTIONS
# ============================================

# Parameters
bin_capacity = 240  # Liters per bin
max_distance = 100  # Meters
w = 1  # Liters per person
total_population = 3000  # Total expected population
cost_per_bin = 200  # Cost per bin (c)

# Maximum bins per location (to force distribution)
max_bins_per_location = 5

# Zones
zones = ['Stage_Area', 'Booth_Area', 'Fair_Area', 'Entrance_Exit_Area']

# ============================================
# POPULATION DISTRIBUTION (Pj)
# Total = 3000
# ============================================

P = {
    'Stage_Area': 1000,
    'Booth_Area': 800,
    'Fair_Area': 700,
    'Entrance_Exit_Area': 500
}

# Verify total
assert sum(P.values()) == total_population

# Calculate waste: Tj = w * Pj
T = {zone: w * P[zone] for zone in zones}

# Candidate bin locations
locations = ['Stage_Area', 'Booth_Area', 'Fair_Area', 'Entrance_Exit_Area']

# Distance matrix - all within 100m
d_ij = np.array([
    # To Zone:     Stage   Booth   Fair    Entrance
    [0,            60,     80,     95],    # From Stage_Area
    [60,           0,      70,     90],    # From Booth_Area
    [80,           70,     0,      50],    # From Fair_Area
    [95,           90,     50,     0]      # From Entrance_Exit_Area
])

print("=" * 70)
print("WASTE BIN OPTIMIZATION MODEL")
print("=" * 70)

# ============================================
# INPUT DATA SUMMARY
# ============================================

print("\nINPUT DATA:")
print(f"  Total population (P): {total_population}")
print(f"  Waste per person (w): {w} L")
print(f"  Total waste generated: {w * total_population} L")
print(f"  Bin capacity: {bin_capacity} L")
print(f"  Minimum bins needed (capacity only): {np.ceil(w * total_population / bin_capacity):.0f}")
print(f"  Number of bin locations: {len(locations)}")
print(f"  Max bins per location: {max_bins_per_location}")
print(f"  Cost per bin (c): PHP {cost_per_bin}\n")

print(f"{'Zone':<20} | {'Population':<12} | {'Waste (Tj)':<12}")
print("-" * 50)
for zone in zones:
    print(f"{zone:<20} | {P[zone]:>10,} | {T[zone]:>10.1f} L")
print("-" * 50)
print(f"{'TOTAL':<20} | {total_population:>10,} | {w * total_population:>10.1f} L")

# ============================================
# PHASE I: MINIMUM NUMBER OF BINS WITH DISTRIBUTION
# ============================================

def phase1_minimum_bins():
    """
    Phase I: Minimize number of bins with FORCED DISTRIBUTION
    Ensures ALL zones are covered with bins spread across locations
    """
    
    print("\n" + "=" * 70)
    print("PHASE I: MINIMUM NUMBER OF BINS (WITH DISTRIBUTION)")
    print("=" * 70)
    print("\nNOTE: xᵢ is INTEGER with max limit to force distribution")
    
    # Calculate theoretical minimum bins needed
    total_waste = sum(T.values())
    theoretical_min = np.ceil(total_waste / bin_capacity)
    print(f"\n  Theoretical minimum bins needed: {theoretical_min:.0f}")
    print(f"  Enforcing distribution across all {len(locations)} zones\n")
    
    # Create problem
    prob = pulp.LpProblem("Phase1_Minimize_Bins", pulp.LpMinimize)
    
    # DECISION VARIABLES
    # x[i] = NUMBER of bins at location i (INTEGER with max limit)
    x = pulp.LpVariable.dicts("x", locations, 0, max_bins_per_location, pulp.LpInteger)
    
    # y[i][j] = 1 if zone j assigned to bin at location i (BINARY)
    y = pulp.LpVariable.dicts("y", [(i, j) for i in locations for j in zones], 0, 1, pulp.LpBinary)
    
    # OBJECTIVE: Minimize total number of bins
    prob += pulp.lpSum(x[i] for i in locations)
    
    # CONSTRAINT 1: Each zone must be assigned to at least one bin
    for j in zones:
        prob += pulp.lpSum(y[(i, j)] for i in locations) >= 1
    
    # CONSTRAINT 2: Zone can only be assigned if bin exists at that location
    for i in locations:
        for j in zones:
            prob += y[(i, j)] <= x[i]
    
    # CONSTRAINT 3: Distance constraint
    for i_idx, i in enumerate(locations):
        for j_idx, j in enumerate(zones):
            if d_ij[i_idx][j_idx] > max_distance:
                prob += y[(i, j)] == 0
    
    # CONSTRAINT 4: Bin capacity constraint
    for i in locations:
        prob += pulp.lpSum(T[j] * y[(i, j)] for j in zones) <= bin_capacity * x[i]
    
    # CONSTRAINT 5: FORCE DISTRIBUTION - Each zone must have a bin at its own location
    # This ensures every zone gets at least one bin at its own location
    for zone in zones:
        prob += x[zone] >= 1  # At least 1 bin at each zone's location
    
    # CONSTRAINT 6: Ensure bins are distributed (no single location gets all bins)
    # Total bins must be distributed across at least 3 locations
    prob += pulp.lpSum([1 for i in locations if x[i] >= 1]) >= 3
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)
    
    if prob.status == pulp.LpStatusOptimal:
        total_bins = int(pulp.value(prob.objective))
        
        print(f"\n✓ OPTIMAL SOLUTION FOUND!")
        print(f"  Status: {pulp.LpStatus[prob.status]}")
        print(f"  Total bins needed: {total_bins}")
        
        # Display bin placement
        print("\n" + "=" * 70)
        print("BIN PLACEMENT PLAN (DISTRIBUTED)")
        print("=" * 70)
        
        bin_placement = {}
        for i in locations:
            bins_at_i = int(x[i].varValue)
            if bins_at_i > 0:
                bin_placement[i] = bins_at_i
                assigned_waste = sum(T[j] * y[(i, j)].varValue for j in zones)
                print(f"  📍 {i}: {bins_at_i} bin(s) (assigned waste: {assigned_waste:.1f} L)")
            else:
                print(f"  📍 {i}: 0 bins")
        
        # Display zone assignments
        print("\n" + "=" * 70)
        print("ZONE ASSIGNMENTS (Distance Check)")
        print("=" * 70)
        
        for j in zones:
            assigned_to = []
            for i in locations:
                if y[(i, j)].varValue > 0.5:
                    assigned_to.append(i)
                    i_idx = locations.index(i)
                    j_idx = zones.index(j)
                    distance = d_ij[i_idx][j_idx]
                    status = "✓" if distance <= max_distance else "✗"
                    print(f"  {status} {j} → {i} (distance: {distance}m)")
            if not assigned_to:
                print(f"  ❌ {j} has NO assignment!")
        
        # Capacity utilization
        print("\n" + "=" * 70)
        print("CAPACITY UTILIZATION")
        print("=" * 70)
        
        for i in locations:
            bins_at_i = int(x[i].varValue)
            if bins_at_i > 0:
                assigned_waste = sum(T[j] * y[(i, j)].varValue for j in zones)
                total_capacity = bins_at_i * bin_capacity
                utilization = (assigned_waste / total_capacity) * 100
                print(f"  • {i}: {assigned_waste:.1f} / {total_capacity} L ({utilization:.1f}%)")
        
        # Verify total capacity
        total_capacity = sum(int(x[i].varValue) for i in locations) * bin_capacity
        print(f"\n  📊 Total capacity: {total_capacity} L")
        print(f"  📊 Total waste: {sum(T.values())} L")
        
        if total_capacity >= sum(T.values()):
            print(f"  ✅ Capacity sufficient")
        else:
            print(f"  ⚠️ Capacity insufficient - need more bins")
        
        # Check distribution
        locations_with_bins = len([i for i in locations if int(x[i].varValue) > 0])
        print(f"\n  📍 Locations with bins: {locations_with_bins}/{len(locations)}")
        
        results = {
            'total_bins': total_bins,
            'bin_placement': bin_placement,
            'assignments': {j: [i for i in locations if y[(i, j)].varValue > 0.5] for j in zones},
            'total_waste': sum(T.values()),
            'total_capacity': total_capacity,
            'locations_with_bins': locations_with_bins
        }
        
        return results
    else:
        print(f"\n❌ NO FEASIBLE SOLUTION FOUND")
        print(f"  Status: {pulp.LpStatus[prob.status]}")
        
        # Calculate what's needed
        total_waste = sum(T.values())
        theoretical_min = np.ceil(total_waste / bin_capacity)
        max_possible = max_bins_per_location * len(locations)
        
        print(f"\n  🔍 DIAGNOSTIC:")
        print(f"     Total waste: {total_waste} L")
        print(f"     Bin capacity: {bin_capacity} L")
        print(f"     Theoretical min bins: {theoretical_min:.0f}")
        print(f"     Max bins possible ({max_bins_per_location} per location): {max_possible}")
        
        if theoretical_min > max_possible:
            print(f"     ❌ Need {theoretical_min:.0f} bins but can only place {max_possible}")
            print(f"     → Increase max_bins_per_location to at least {np.ceil(theoretical_min/len(locations)):.0f}")
        
        # Check zone coverage
        print(f"\n  📍 Zone coverage check:")
        for j_idx, j in enumerate(zones):
            reachable = []
            for i_idx, i in enumerate(locations):
                if d_ij[i_idx][j_idx] <= max_distance:
                    reachable.append(i)
            print(f"     {j}: can be served by {reachable}")
            if len(reachable) == 0:
                print(f"       ❌ {j} has NO bin within {max_distance}m!")
        
        return None

# ============================================
# PHASE II: MINIMUM BUDGET
# ============================================

def phase2_minimum_budget(results):
    """Phase II: Calculate minimum required budget"""
    
    print("\n" + "=" * 70)
    print("PHASE II: MINIMUM REQUIRED BUDGET")
    print("=" * 70)
    
    if results is None:
        return None
    
    min_budget = results['total_bins'] * cost_per_bin
    
    print(f"\n  💰 Cost per bin (c): PHP {cost_per_bin}")
    print(f"  🔢 Minimum number of bins: {results['total_bins']}")
    print(f"\n  💵 MINIMUM BUDGET = {results['total_bins']} × {cost_per_bin} = PHP {min_budget:,.0f}")
    
    results['min_budget'] = min_budget
    return results

# ============================================
# PHASE III: BUDGET SCENARIOS WITH DISTRIBUTION
# ============================================

def phase3_budget_scenarios(results):
    """Phase III: Different budget scenarios with distribution constraints"""
    
    print("\n" + "=" * 70)
    print("PHASE III: BUDGET SCENARIO ANALYSIS")
    print("=" * 70)
    
    if results is None:
        return None
    
    min_budget = results['min_budget']
    budget_percentages = [0.3, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0]
    scenarios = [min_budget * p for p in budget_percentages]
    
    scenario_results = []
    
    print(f"\n{'Budget (B)':<15} | {'# of Bins':<10} | {'Zones Covered':<15} | {'Distribution':<15} | {'Status'}")
    print("-" * 80)
    
    for budget in scenarios:
        budget = round(budget, -1)
        
        prob = pulp.LpProblem(f"Phase3_Budget_{budget}", pulp.LpMaximize)
        
        # Integer variables for bins with max limit
        x = pulp.LpVariable.dicts("x", locations, 0, max_bins_per_location, pulp.LpInteger)
        y = pulp.LpVariable.dicts("y", [(i, j) for i in locations for j in zones], 0, 1, pulp.LpBinary)
        
        # Coverage variables
        zone_covered = pulp.LpVariable.dicts("covered", zones, 0, 1, pulp.LpBinary)
        
        # Objective: Maximize number of zones covered
        prob += pulp.lpSum(zone_covered[j] for j in zones)
        
        # Constraints
        for j in zones:
            prob += zone_covered[j] <= pulp.lpSum(y[(i, j)] for i in locations)
            prob += pulp.lpSum(y[(i, j)] for i in locations) >= zone_covered[j]
        
        for i in locations:
            for j in zones:
                prob += y[(i, j)] <= x[i]
        
        # Budget constraint
        prob += pulp.lpSum(cost_per_bin * x[i] for i in locations) <= budget
        
        # Distance constraint
        for i_idx, i in enumerate(locations):
            for j_idx, j in enumerate(zones):
                if d_ij[i_idx][j_idx] > max_distance:
                    prob += y[(i, j)] == 0
        
        # Capacity constraint
        for i in locations:
            prob += pulp.lpSum(T[j] * y[(i, j)] for j in zones) <= bin_capacity * x[i]
        
        # Distribution constraint for full coverage scenarios
        if budget >= min_budget:
            # Try to enforce distribution when budget allows
            for zone in zones:
                prob += x[zone] >= zone_covered[zone]
        
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        if prob.status == pulp.LpStatusOptimal:
            total_bins = int(sum(x[i].varValue for i in locations))
            covered_zones = sum(zone_covered[j].varValue for j in zones)
            locations_with_bins = sum(1 for i in locations if x[i].varValue > 0.5)
            
            if covered_zones == len(zones):
                coverage = "✅ Full"
                status = "Feasible"
            else:
                coverage = f"⚠️ {covered_zones:.0f}/{len(zones)}"
                status = "Partial"
            
            scenario_results.append({
                'Budget': f"PHP {budget:,.0f}",
                '# of Bins': total_bins,
                'Zones Covered': coverage,
                'Distribution': f"{locations_with_bins} locations",
                'Status': status
            })
            print(f"PHP {budget:>10,} | {total_bins:>10} | {coverage:<15} | {locations_with_bins:>13} locs | {status}")
        else:
            scenario_results.append({
                'Budget': f"PHP {budget:,.0f}",
                '# of Bins': 0,
                'Zones Covered': '❌ None',
                'Distribution': 'N/A',
                'Status': 'Infeasible'
            })
            print(f"PHP {budget:>10,} | {'Infeasible':>10} | {'❌ None':<15} | {'N/A':<15} | Infeasible")
    
    results['budget_scenarios'] = scenario_results
    
    # Display table
    print("\n" + "=" * 70)
    print("PHASE III RESULTS TABLE")
    print("=" * 70)
    df = pd.DataFrame(scenario_results)
    print(df.to_string(index=False))
    
    return results

# ============================================
# PRINT FINAL SUMMARY
# ============================================

def print_final_summary(results):
    """Print final summary with distribution check"""
    
    if results is None:
        return
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ SOLUTION IS FEASIBLE AND DISTRIBUTED!")
    print(f"\n  📊 KEY METRICS:")
    print(f"     • Total bins needed: {results['total_bins']}")
    print(f"     • Minimum budget: PHP {results['min_budget']:,.0f}")
    print(f"     • Total waste: {results['total_waste']} L")
    print(f"     • Total capacity: {results['total_capacity']} L")
    
    print(f"\n  🗺️  BIN PLACEMENT (DISTRIBUTED ACROSS ZONES):")
    for location, count in results['bin_placement'].items():
        print(f"     • {location}: {count} bin(s)")
    
    print(f"\n  🔗 ZONE COVERAGE:")
    for zone, assigned in results['assignments'].items():
        print(f"     • {zone} → {', '.join(assigned)}")
    
    # Verify each zone has a bin at its own location
    print(f"\n  ✅ DISTRIBUTION VERIFICATION:")
    all_covered = True
    for zone in zones:
        if zone in results['bin_placement']:
            print(f"     ✓ {zone} has {results['bin_placement'][zone]} bin(s) at its own location")
        else:
            print(f"     ⚠️ {zone} has NO bin at its own location")
            all_covered = False
    
    if all_covered:
        print(f"\n  ✅ EVERY ZONE HAS AT LEAST ONE BIN AT ITS OWN LOCATION!")
    
    print(f"\n  💰 BUDGET RECOMMENDATION:")
    print(f"     • Minimum budget needed: PHP {results['min_budget']:,.0f}")
    print(f"     • Recommended budget (with contingency): PHP {results['min_budget'] * 1.1:,.0f}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run Phase I
    results = phase1_minimum_bins()
    
    if results:
        # Run Phase II
        results = phase2_minimum_budget(results)
        
        # Run Phase III
        results = phase3_budget_scenarios(results)
        
        # Print final summary
        print_final_summary(results)
        
        print("\n" + "=" * 70)
        print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("📁 Results are distributed across ALL zones")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ OPTIMIZATION FAILED")
        print("=" * 70)
        print("\n🔧 RECOMMENDED FIXES:")
        print("  1. Increase max_bins_per_location (currently 5)")
        print("  2. Increase bin capacity (currently 240L)")
        print("  3. Increase max_distance (currently 100m)")
        print("  4. Adjust population distribution")