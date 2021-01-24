from ..problem.optimal_dispatch import OptimalDispatch


# Exports
__all__ = ["encode_problem", "encode_solution", "decode_solution", "decode_indices", "draw_best"]


def encode_problem(problem:OptimalDispatch):
    """
    Transform an instance of the Optimal Dispatch problem to the structures
    used by the Differential Evolution algorithm.
    :param problem: Instance of the problem to be encoded to the structures
    used by the Differential Evolution algorithm.
    :return: It returns three values: the function for evaluating an individual,
    a list of lower bounds, and a list of upper bounds.
    """

    # Evaluation function
    feval = lambda x: problem.evaluate(decode_solution(problem, x))

    # Lower-bounds, upper-bounds and variables' types
    var_lb, var_ub = [], []

    # Variables related to the generators
    for i in range(0, problem.n_generators):

        # Generator rate
        var_lb += [problem.generators[i]["lower_rate"] * 0.5] * problem.n_intervals
        var_ub += [problem.generators[i]["upper_rate"]] * problem.n_intervals

        # Generators fuel composition
        var_lb += [1] * problem.n_intervals
        var_ub += [3] * problem.n_intervals

    # Variables related to the battery
    var_lb += [-problem.battery["max_load"]] * problem.n_intervals
    var_ub += [problem.battery["max_load"]] * problem.n_intervals

    # Variables related to biogas production
    if problem.enable_cvar:
        var_lb += [problem.biogas["mean_production"] - 4 * problem.biogas["deviation_production"]]
        var_ub += [problem.biogas["mean_production"] + 4 * problem.biogas["deviation_production"]]
    else:
        var_lb += [problem.biogas["mean_production"]]
        var_ub += [problem.biogas["mean_production"]]

    return feval, var_lb, var_ub


def encode_solution(problem, solution):
    """
    Transform a solution of the Optimal Dispatch problem to the type of encoding
    expected by the differential evolution algorithm.
    :param problem: an instance of the Optimal Dispatch problem.
    :param solution: a solution of the Optimal Dispatch problem.
    :return: an individual of the Differential Evolution.
    """

    # Get solution data
    generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

    x = []
    for i in range(0, problem.n_generators):
        x += [generators_status[i][j] * generators_rate[i][j] for j in range(0, problem.n_intervals)]
        x += generators_fuel_composition[i]
    x += battery_energy
    x += biogas_production

    return x


def decode_solution(problem, x):
    """
    Transform an individual (a solution returned by the Differential Evolution) to
    the type of encoding used by the original Optimal Dispatch problem.
    :param problem: an instance of the Optimal Dispatch problem.
    :param x: an individual of the Differential Evolution.
    :return: a solution encoded as expected by the original Optimal Dispatch problem.
    """

    idx = 0
    generators_status = []
    generators_rate = []
    generators_fuel_composition = []
    battery_load = [0] * problem.n_intervals
    biogas_production = []

    # Generators
    for i in range(0, problem.n_generators):

        generators_rate.append([0.0] * problem.n_intervals)
        generators_status.append([0] * problem.n_intervals)
        generators_fuel_composition.append([0] * problem.n_intervals)

        for j in range(0, problem.n_intervals):
            generators_rate[i][j] = x[idx]
            generators_status[i][j] = 1 if x[idx] >= problem.generators[i]["lower_rate"] else 0
            idx += 1

        for j in range(0, problem.n_intervals):
            generators_fuel_composition[i][j] = round(x[idx])
            idx += 1

    # Battery load
    for j in range(0, problem.n_intervals):
        battery_load[j] = x[idx]
        idx += 1

    # Biogas
    biogas_production = x[idx]

    return generators_status, generators_rate, generators_fuel_composition, battery_load, biogas_production


def decode_indices(problem, x):
    """
    Transform an individual (a solution returned by the Differential Evolution) to
    the type of encoding used by the original Optimal Dispatch problem.
    :param problem: an instance of the Optimal Dispatch problem.
    :param x: an individual of the Differential Evolution.
    :return: a solution encoded as expected by the original Optimal Dispatch problem.
    """

    idx = 0
    idx_generators_status = []
    idx_generators_rate = []
    idx_generators_fuel_composition = []
    idx_battery_load = [0] * problem.n_intervals
    idx_biogas_production = []

    # Generators
    for i in range(0, problem.n_generators):

        idx_generators_status.append([0] * problem.n_intervals)
        for j in range(0, problem.nintervals):
            idx_generators_status[i][j] = idx
            idx += 1

        idx_generators_rate.append([0] * problem.n_intervals)
        for j in range(0, problem.n_intervals):
            idx_generators_rate[i][j] = idx
            idx += 1

        idx_generators_fuel_composition.append([0] * problem.n_intervals)
        for j in range(0, problem.n_intervals):
            idx_generators_fuel_composition[i][j] = idx
            idx += 1

    # Battery load
    for j in range(0, problem.n_intervals):
        idx_battery_load[j] = idx
        idx += 1

    # Biogas
    idx_biogas_production = idx

    return idx_generators_status, idx_generators_rate, idx_generators_fuel_composition, idx_battery_load, idx_biogas_production


def draw_best(problem, population_x, population_f, block=False, show=True, interactive=True):

    """
    Draw the best solution in the population.
    :param problem: an instance of the Optimal Dispatch problem.
    :param population_x: the current population of the Differential Evolution.
    :param population_f: evaluation of the current population of the Differential Evolution.
    """

    # Find best solution
    best_idx = 0
    best_obj, best_g, best_h = population_f[0]
    best_inf = sum([sum(value) for value in problem.calculate_infeasibility(best_g, best_h)])

    for i in range(1, len(population_f)):
        obj, g, h = population_f[i]
        inf = sum([sum(value) for value in problem.calculate_infeasibility(g, h)])
        if (inf, obj) < (best_inf, best_obj):
            best_idx, best_obj, best_inf, best_g, best_h = i, obj, inf, g, h

    # Decode the best solution
    solution = decode_solution(problem, population_x[best_idx])

    # Draw solution
    problem.draw_solution(solution, block=block, show=show, interactive=interactive)
