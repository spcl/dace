#!/usr/bin/env python3
import sys
import pulp
import math
import random

def solve_noc_lp(p=4):
    # Create the MILP problem (minimization)
    prob = pulp.LpProblem("NoC_Channel_Assignment", pulp.LpMinimize)
    
    # Decision variables: x[i,j] = 1 if cell (i,j) is assigned to L; 0 if assigned to B.
    # (Assume grid indices: i = column index, j = row index, with (0,0) at bottom left)
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(p) for j in range(p)],
                                cat="Binary")
    
    # M is a continuous variable representing the maximum channel cost.
    M = pulp.LpVariable("M", lowBound=0, cat="Continuous")
    
    # Objective: minimize M.
    prob += M, "Minimize_Max_Channel_Cost"
    
    half = p // 2  # number of L's (and B's) per row/column.
    
    # Constraint: each row must have exactly half L's.
    for j in range(p):
        prob += pulp.lpSum(x[i, j] for i in range(p)) == half, f"Row_{j}_L_Count"
    
    # Constraint: each column must have exactly half L's.
    for i in range(p):
        prob += pulp.lpSum(x[i, j] for j in range(p)) == half, f"Col_{i}_L_Count"
    
    # For each row j (left channel), cost is sum_{i} (i * x[i,j]).
    for j in range(p):
        prob += pulp.lpSum(i * x[i, j] for i in range(p)) <= M, f"LeftChannelCost_Row_{j}"
    
    # For each column i (bottom channel), cost is sum_{j} (j * (1 - x[i,j])).
    for i in range(p):
        prob += pulp.lpSum(j * (1 - x[i, j]) for j in range(p)) <= M, f"BottomChannelCost_Col_{i}"
    
    # Solve the problem.
    solver = pulp.PULP_CBC_CMD(msg=0)  # You can set msg=1 to see solver output.
    result = prob.solve(solver)
    
    print("Solver Status:", pulp.LpStatus[prob.status])
    print("Minimum maximum channel cost (M):", pulp.value(M))
    
    # Build and print the assignment grid.
    assignment = {}
    grid = [["" for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p):
            val = pulp.value(x[i, j])
            assignment[(i, j)] = val
            # Using threshold 0.5: if 1 then L; otherwise B.
            grid[p - 1 - j][i] = "L" if val >= 0.5 else "B"
    
    print("\nAssignment grid (with (0,0) at bottom left):")
    for row in grid:
        print(" ".join(row))
    
    return assignment, pulp.value(M)





# ----------------------------------------
# Channel-based cost functions.
# For a left channel tied to a row, each processor at (c, r)
# contributes a cost equal to its horizontal distance (c).
def left_channel_cost(processors):
    return sum(c for (c, r) in processors)

# For a bottom channel tied to a column, each processor at (c, r)
# contributes a cost equal to its vertical distance (r).
def bottom_channel_cost(processors):
    return sum(r for (c, r) in processors)

def compute_channel_costs(assignment, p):
    """
    Computes the channel costs for each left (row) and bottom (column) channel.
    For each row r, the left channel cost is the sum of the x–coordinates (c) of
    all processors in that row assigned "L".
    For each column c, the bottom channel cost is the sum of the y–coordinates (r)
    of all processors in that column assigned "B".
    """
    left_costs = {r: [] for r in range(p)}
    bottom_costs = {c: [] for c in range(p)}
    
    for (c, r), typ in assignment.items():
        if typ == "L":
            left_costs[r].append((c, r))
        else:  # typ == "B"
            bottom_costs[c].append((c, r))
    
    left_channel_vals = {r: left_channel_cost(left_costs[r]) for r in range(p)}
    bottom_channel_vals = {c: bottom_channel_cost(bottom_costs[c]) for c in range(p)}
    return left_channel_vals, bottom_channel_vals

def objective(assignment, p):
    """
    The overall objective is defined here as the maximum cost among all channels.
    (You can change this to a sum or any other metric.)
    """
    left_channels, bottom_channels = compute_channel_costs(assignment, p)
    max_left = max(left_channels.values()) if left_channels else 0
    max_bottom = max(bottom_channels.values()) if bottom_channels else 0
    return max(max_left, max_bottom)

# ----------------------------------------
# Balanced initial assignment for any even p.
def initial_assignment(p):
    """
    For a p x p grid (with p even), assign each cell as "L" or "B" so that each row
    and each column gets exactly p/2 L's and p/2 B's.
    
    Method: For each row r, assign "L" to those columns c for which (c - r) mod p is less than p/2,
    and "B" otherwise.
    """
    assignment = {}
    half = p // 2
    for r in range(p):
        for c in range(p):
            if (c - r) % p < half:
                assignment[(c, r)] = "L"
            else:
                assignment[(c, r)] = "B"
    return assignment

# ----------------------------------------
# Neighbor generation using a 2x2 switch.
def get_neighbor(assignment, p):
    """
    Generates a neighboring assignment by performing a 2x2 switch.
    Randomly select two distinct rows and two distinct columns to form a 2x2 submatrix.
    If the submatrix forms a checkerboard pattern (diagonally opposite cells match, but
    are different from the other diagonal), flip the pattern.
    This move preserves the number of "L" and "B" in every row and column.
    """
    new_assignment = assignment.copy()
    for _ in range(100):  # Try up to 100 times to find a valid move.
        r1, r2 = random.sample(range(p), 2)
        c1, c2 = random.sample(range(p), 2)
        cell_a = (c1, r1)
        cell_b = (c2, r1)
        cell_c = (c1, r2)
        cell_d = (c2, r2)
        # Check for a valid checkerboard pattern.
        if (new_assignment[cell_a] == new_assignment[cell_d] and
            new_assignment[cell_b] == new_assignment[cell_c] and
            new_assignment[cell_a] != new_assignment[cell_b]):
            # Swap the assignments in the 2x2 submatrix.
            val_a = new_assignment[cell_a]
            val_b = new_assignment[cell_b]
            new_assignment[cell_a] = val_b
            new_assignment[cell_b] = val_a
            new_assignment[cell_c] = val_a
            new_assignment[cell_d] = val_b
            return new_assignment
    return new_assignment  # Return the original if no move was found.

# ----------------------------------------
# Simulated Annealing search.
def simulated_annealing(p, initial_temp=100.0, cooling_rate=0.995, iterations=10000):
    current = initial_assignment(p)
    current_obj = objective(current, p)
    best = current
    best_obj = current_obj
    temp = initial_temp

    for i in range(iterations):
        neighbor = get_neighbor(current, p)
        neighbor_obj = objective(neighbor, p)
        delta = neighbor_obj - current_obj

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_obj = neighbor_obj
            if current_obj < best_obj:
                best = current
                best_obj = current_obj

        temp *= cooling_rate

    return best, best_obj

# ----------------------------------------
# Utility functions to print the grid and channel costs.
def print_assignment(assignment, p):
    grid = [["" for _ in range(p)] for _ in range(p)]
    for (c, r), typ in assignment.items():
        grid[p - 1 - r][c] = typ
    for row in grid:
        print(" ".join(row))

def print_channel_costs(assignment, p):
    left_channels, bottom_channels = compute_channel_costs(assignment, p)
    print("Left channel costs (row: cost):")
    for r in range(p):
        print(f"Row {r}: {left_channels[r]}")
    print("Bottom channel costs (col: cost):")
    for c in range(p):
        print(f"Col {c}: {bottom_channels[c]}")

# # ----------------------------------------
# # Main execution.
# if __name__ == "__main__":
#     p = 32  # Default grid size is 4x4. For p=8, each row/column will have 4 L's and 4 B's.
        
#     best_assignment, best_obj = simulated_annealing(p)
#     print("Best objective (maximum channel cost):", best_obj)
#     print("\nAssignment grid (with (0,0) at bottom left):")
#     print_assignment(best_assignment, p)
#     print("\nChannel costs:")
#     print_channel_costs(best_assignment, p)
