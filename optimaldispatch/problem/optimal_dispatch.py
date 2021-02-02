import math
import json
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.lines as mlines
from scipy.stats import norm

# Exports
__all__ = ["OptimalDispatch"]


class OptimalDispatch(object):
    """
    Class that defines the Optimal Dispatch problem.
    """

    def __init__(self, instance_file, cvar_epsilon=None):
        """
        Constructor.
        :param instance_file: JSON file with instance data
        """

        # Parse JSON file
        data = json.loads(instance_file.read())

        # Demand and energy price
        self.n_intervals = data["n_intervals"]
        self.demand = data["demand"]
        self.selling_price = data["selling_price"]
        self.buying_price = data["buying_price"]

        # Generators
        self.n_generators = data["n_generators"]
        self.generators = data["generators"]

        # Battery
        self.battery = data["battery"]

        # Fuel
        self.biogas = data["biogas"]
        self.ethanol = data["ethanol"]
        self.biomethane = data["biomethane"]
        self.gnv = data["gnv"]
        self.fuel_buses_demand = data["fuel_buses_demand"]

        # Solar energy
        self.solar_energy = data["solar_energy"]

        # CVaR e-constraint
        self.enable_cvar = cvar_epsilon is not None
        self.epsilon = cvar_epsilon

    def evaluate(self, solution):
        """
        Calculate objective function of the given solution.
        :param solution: Solution to evaluate.
        :return: Objective function value.
        """

        # Get solution attributes
        generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

        # Evaluate biogas cost
        biogas_cost = ((self.biogas["maintenance_cost"][0] * biogas_production + self.biogas["maintenance_cost"][1] +
                        self.biogas["input_cost"] + self.biogas["transport_cost"]) / biogas_production)

        # Battery energy cost
        battery_load = ([0.0] * self.n_intervals) + [self.battery["initial_load"]]
        battery_use_cost = [0.0] * self.n_intervals

        for i in range(0, self.n_intervals):
            if battery_energy[i] > 0:
                battery_load[i] = battery_load[i - 1] + battery_energy[i] * self.battery["eff_charge"]
            else:
                battery_load[i] = battery_load[i - 1] + battery_energy[i] / self.battery["eff_discharge"]

            # Compute battery use cost
            battery_use_cost[i] = abs(battery_energy[i] * self.battery["cost"])

        # Generators cost
        generators_status_changed = [[]] * self.n_generators
        generators_up_down_cost = [0.0] * self.n_generators
        generators_efficiency = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_fuel_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_biogas_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_ethanol_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_fuel_cost = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_total_fuel_cost = 0.0
        generators_total_used_biogas = 0.0
        generators_total_used_ethanol = 0.0

        for i in range(0, self.n_generators):

            # Status changing (Up / Down)
            generators_status_aux = generators_status[i] + [self.generators[i]["initial_state"]]
            generators_status_changed[i] = [generators_status_aux[j] != generators_status_aux[j - 1]
                                            for j in range(0, self.n_intervals)]

            for j in range(0, self.n_intervals):

                # Generators efficiency
                eff_key = None
                if generators_fuel_composition[i][j] == 1:
                    eff_key = "efficiency_1"
                elif generators_fuel_composition[i][j] == 2:
                    eff_key = "efficiency_2"
                else:
                    eff_key = "efficiency_3"

                a = self.generators[i][eff_key][0]
                b = self.generators[i][eff_key][1]
                generators_efficiency[i][j] = a * (generators_rate[i][j] / self.generators[i]["upper_rate"]) + b

                # Generators consumption and fuel cost
                generators_fuel_consumption[i][j] = (generators_rate[i][j] * generators_status[i][j] /
                                                     generators_efficiency[i][j])
                generators_biogas_consumption[i][j] = (generators_fuel_consumption[i][j] *
                                                       (0.20 * 2 ** (generators_fuel_composition[i][j] - 1) +
                                                        0.10 * (generators_fuel_composition[i][j] - 1)))
                generators_ethanol_consumption[i][j] = (generators_fuel_consumption[i][j] -
                                                        generators_biogas_consumption[i][j])
                generators_fuel_cost[i][j] = (generators_biogas_consumption[i][j] * biogas_cost +
                                              generators_ethanol_consumption[i][j] * self.ethanol["cost"])

                generators_total_used_biogas += generators_biogas_consumption[i][j]
                generators_total_used_ethanol += generators_ethanol_consumption[i][j]
                generators_total_fuel_cost += generators_fuel_cost[i][j]

                # Cost of changing generator status (Up / Down)
                if generators_status_changed[i][j]:
                    if generators_status[i][j] == 1:
                        generators_up_down_cost[i] += self.generators[i]["up_cost"]
                    else:
                        generators_up_down_cost[i] += self.generators[i]["down_cost"]

        # Fuel Commercialization
        remaining_biogas = biogas_production - generators_total_used_biogas
        biomethane_disp = (remaining_biogas * self.biomethane["efficiency"]) / 10.92
        biomethane_cost = abs(self.biomethane["maintenance_cost"][0] * biomethane_disp +
                              self.biomethane["maintenance_cost"][1])

        biogas_energy_consumption = (self.biogas["consumption"][0] * biogas_production +
                                     self.biogas["consumption"][1]) / 8

        demand = self.demand.copy()
        for j in range(0, self.n_intervals):
            if 16 <= j <= 34:
                demand[j] = demand[j] + biogas_energy_consumption

        if biomethane_disp < 60 * 8:
            biomethane_energy_consumption = (self.biomethane["consumption"] * biomethane_disp) / 8
            for j in range(0, self.n_intervals):
                if 16 <= j <= 29:
                    demand[j] = demand[j] + abs(biomethane_energy_consumption)
        else:
            biomethane_energy_time = math.ceil(biomethane_disp / 60)
            biomethane_energy_consumption = (self.biomethane["consumption"] * biomethane_disp) / biomethane_energy_time
            for j in range(0, self.n_intervals):
                if 16 <= j <= (16 + biomethane_energy_time):
                    demand[j] = demand[j] + biomethane_energy_consumption

        if biomethane_disp >= 0 and biomethane_disp > self.fuel_buses_demand:
            fuel_buses_cost = biomethane_cost * self.fuel_buses_demand
        elif 0 <= biomethane_disp < self.fuel_buses_demand:
            fuel_buses_cost = (biomethane_cost * biomethane_disp +
                               self.gnv["cost"] * (self.fuel_buses_demand - biomethane_disp))
        else:
            fuel_buses_cost = self.gnv["cost"] * self.fuel_buses_demand

        # Electrical Energy Commercialization (buy / sell)
        commercialization_electric_energy = demand.copy()
        commercialization_electric_energy_cost = [0.0] * self.n_intervals

        for j in range(0, self.n_intervals):

            # Energy commercialized
            commercialization_electric_energy[j] += battery_energy[j]
            commercialization_electric_energy[j] -= self.solar_energy[j]

            for i in range(0, self.n_generators):
                commercialization_electric_energy[j] -= generators_status[i][j] * generators_rate[i][j]

            # Cost of energy commercialized
            if commercialization_electric_energy[j] > 0:
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.buying_price[j]
            else:
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.selling_price[j]

        # Total cost
        total_cost = (sum(commercialization_electric_energy_cost) +
                      sum(generators_up_down_cost) +
                      generators_total_fuel_cost +
                      fuel_buses_cost +
                      sum(battery_use_cost))

        for i in range(0, self.n_generators):
            total_cost += sum(generators_fuel_cost[i])

        # CONSTRAINTS

        # Constraint: ramp rate limit
        constraints_ramp_rate_limit = []
        for i in range(0, self.n_generators):

            generators_rate_aux = generators_rate[i] + [generators_rate[i][0]]
            generators_status_aux = generators_status[i] + [self.generators[i]["initial_state"]]

            # Changing in generator's rate
            generators_rate_changing = [
                abs((generators_rate_aux[j] * generators_status_aux[j]) -
                    (generators_rate_aux[j - 1] * generators_status_aux[j - 1]))
                for j in range(0, self.n_intervals)]

            # Ramp rate constraints: (change - limit) * status_not_changed <= 0
            for j in range(0, self.n_intervals):
                constraint = ((generators_rate_changing[j] - self.generators[i]["ramp_rate_limit"]) *
                              int(generators_status_changed[i][j]))
                constraints_ramp_rate_limit.append(constraint)

        # Constraint: window without changing generator status
        constraints_generator_window = []
        for i in range(0, self.n_generators):

            generators_status_changed_aux = generators_status_changed[i]

            for k in range(1, self.generators[i]["up_down_window"]):
                start_window = 0
                end_window = k

                # Window constraint: changes - 1 <= 0
                constraint = sum(generators_status_changed_aux[start_window:end_window]) - 1
                constraints_generator_window.append(constraint)

            for j in range(0, len(generators_status_changed_aux)):
                start_window = j
                end_window = min(len(generators_status_changed_aux), j + self.generators[i]["up_down_window"])

                # Window constraint: changes - 1 <= 0
                constraint = sum(generators_status_changed_aux[start_window:end_window]) - 1
                constraints_generator_window.append(constraint)

        # Constraint: Fuel Limit: used_used - fuel_limit <= 0
        constraints_biogas_limit = abs(min(0, remaining_biogas))
        constraints_ethanol_limit = abs(min(0, (self.ethanol["disponibility"] - generators_total_used_ethanol)))
        constraints_fuel_limit = constraints_biogas_limit + constraints_ethanol_limit

        # Constraint: Battery Limit
        battery_free_load = [
            (self.battery["max_load"] - battery_load[i - 1]) / self.battery["eff_charge"]
            for i in range(0, self.n_intervals)]
        battery_available_load = [
            battery_load[i - 1] * self.battery["eff_discharge"]
            for i in range(0, self.n_intervals)]

        constraint_battery1 = 0
        constraint_battery2 = 0
        for i in range(0, self.n_intervals):
            if battery_energy[i] < 0:
                constraint_battery1 += max((abs(battery_energy[i]) - battery_available_load[i]), 0)
            else:
                constraint_battery1 += max((battery_energy[i] - battery_free_load[i]), 0)

            constraint_battery2 = sum(
                [(abs(battery_load[i]) if (battery_load[i] < 0 or battery_load[i] > self.battery["max_load"]) else 0)
                 for i in range(0, self.n_intervals)])

        # Arrays of equality and inequality constraints
        equality_constraints = []
        inequality_constraints = constraints_ramp_rate_limit + constraints_generator_window + \
                                 [constraints_fuel_limit, constraint_battery1, constraint_battery2]

        # CVaR (if enabled)
        if self.enable_cvar:
            z = (biogas_production - self.biogas["mean_production"]) / self.biogas["deviation_production"]
            alpha = norm.cdf(z)
            biogas_cvar = abs(alpha ** -1 * norm.pdf(z) * self.biogas["deviation_production"] - self.biogas["mean_production"])
            diff_biogas = self.biogas["mean_production"] - biogas_cvar
            cvar = diff_biogas * 0.27 * self.buying_price[40]

            constraint_cvar = min((cvar - self.epsilon * 82), 0)
            inequality_constraints.append(constraint_cvar)

        # Return total cost and constraints
        return total_cost, inequality_constraints, equality_constraints

    def calculate_infeasibility(self, g, h, eps=1e-5):
        """
        Calculate infeasibility for all constraints.
        :param g: inequality constraints.
        :param h: equality constraints.
        :param eps: precision.
        :return: two lists: the first regarding to the inequality constraints and the
        second regarding to the equality constraints.
        """
        inf_g = [max(0, value - eps) for value in g]
        inf_h = [abs(value) > eps for value in h]
        return inf_g, inf_h

    def is_feasible(self, solution, eps=1e-5):
        """
        Check if a give solution is feasible.
        :param solution: Solution to check feasibility.
        :return: True if the solution is feasible, or False otherwise.
        """
        _, g, h = self.evaluate(solution)
        inf_g, inf_h = self.calculate_infeasibility(g, h, eps)
        return (sum(inf_g) + sum(inf_h)) > 0

    def draw_solution(self, solution, label="optimaldispatch", block=False, interactive=False, show=True):
        """
        Draw a solution.
        """

        # Get solution attributes
        generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

        # Evaluate solution
        cost, g, h = self.evaluate(solution)
        infeasibility = sum([sum(value) for value in self.calculate_infeasibility(g, h)])

        # Evaluate biogas cost
        biogas_cost = ((self.biogas["maintenance_cost"][0] * biogas_production + self.biogas["maintenance_cost"][1] +
                        self.biogas["input_cost"] + self.biogas["transport_cost"]) / biogas_production)

        # Battery energy cost
        battery_load = ([0.0] * self.n_intervals) + [self.battery["initial_load"]]
        battery_use_cost = [0.0] * self.n_intervals

        for i in range(0, self.n_intervals):
            if battery_energy[i] > 0:
                battery_load[i] = battery_load[i - 1] + battery_energy[i] * self.battery["eff_charge"]
            else:
                battery_load[i] = battery_load[i - 1] + battery_energy[i] / self.battery["eff_discharge"]

            # Compute battery use cost
            battery_use_cost[i] = abs(battery_energy[i] * self.battery["cost"])

        # Generators cost
        generators_status_changed = [[]] * self.n_generators
        generators_up_down_cost = [0.0] * self.n_generators
        generators_efficiency = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_fuel_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_biogas_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_ethanol_consumption = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_fuel_cost = [[0.0 for _ in range(0, self.n_intervals)] for _ in range(0, self.n_generators)]
        generators_total_fuel_cost = 0.0
        generators_total_used_biogas = 0.0
        generators_total_used_ethanol = 0.0

        for i in range(0, self.n_generators):

            # Status changing (Up / Down)
            generators_status_aux = generators_status[i] + [self.generators[i]["initial_state"]]
            generators_status_changed[i] = [generators_status_aux[j] != generators_status_aux[j - 1]
                                            for j in range(0, self.n_intervals)]

            for j in range(0, self.n_intervals):

                # Generators efficiency
                eff_key = None
                if generators_fuel_composition[i][j] == 1:
                    eff_key = "efficiency_1"
                elif generators_fuel_composition[i][j] == 2:
                    eff_key = "efficiency_2"
                else:
                    eff_key = "efficiency_3"

                a = self.generators[i][eff_key][0]
                b = self.generators[i][eff_key][1]
                generators_efficiency[i][j] = a * (generators_rate[i][j] / self.generators[i]["upper_rate"]) + b

                # Generators consumption and fuel cost
                generators_fuel_consumption[i][j] = (generators_rate[i][j] * generators_status[i][j] /
                                                     generators_efficiency[i][j])
                generators_biogas_consumption[i][j] = (generators_fuel_consumption[i][j] *
                                                       (0.20 * 2 ** (generators_fuel_composition[i][j] - 1) +
                                                        0.10 * (generators_fuel_composition[i][j] - 1)))
                generators_ethanol_consumption[i][j] = (generators_fuel_consumption[i][j] -
                                                        generators_biogas_consumption[i][j])
                generators_fuel_cost[i][j] = (generators_biogas_consumption[i][j] * biogas_cost +
                                              generators_ethanol_consumption[i][j] * self.ethanol["cost"])

                generators_total_used_biogas += generators_biogas_consumption[i][j]
                generators_total_used_ethanol += generators_ethanol_consumption[i][j]
                generators_total_fuel_cost += generators_fuel_cost[i][j]

                # Cost of changing generator status (Up / Down)
                if generators_status_changed[i][j]:
                    if generators_status[i][j] == 1:
                        generators_up_down_cost[i] += self.generators[i]["up_cost"]
                    else:
                        generators_up_down_cost[i] += self.generators[i]["down_cost"]

        # Fuel Commercialization
        remaining_biogas = biogas_production - generators_total_used_biogas
        biomethane_disp = (remaining_biogas * self.biomethane["efficiency"]) / 10.92
        biomethane_cost = abs(self.biomethane["maintenance_cost"][0] * biomethane_disp +
                              self.biomethane["maintenance_cost"][1])

        biogas_energy_consumption = (self.biogas["consumption"][0] * biogas_production +
                                     self.biogas["consumption"][1]) / 8

        demand = self.demand.copy()
        for j in range(0, self.n_intervals):
            if 16 <= j <= 34:
                demand[j] = demand[j] + biogas_energy_consumption

        if biomethane_disp < 60 * 8:
            biomethane_energy_consumption = (self.biomethane["consumption"] * biomethane_disp) / 8
            for j in range(0, self.n_intervals):
                if 16 <= j <= 29:
                    demand[j] = demand[j] + abs(biomethane_energy_consumption)
        else:
            biomethane_energy_time = math.ceil(biomethane_disp / 60)
            biomethane_energy_consumption = (self.biomethane["consumption"] * biomethane_disp) / biomethane_energy_time
            for j in range(0, self.n_intervals):
                if 16 <= j <= (16 + biomethane_energy_time):
                    demand[j] = demand[j] + biomethane_energy_consumption

        if biomethane_disp >= 0 and biomethane_disp > self.fuel_buses_demand:
            fuel_buses_cost = biomethane_cost * self.fuel_buses_demand
        elif 0 <= biomethane_disp < self.fuel_buses_demand:
            fuel_buses_cost = (biomethane_cost * biomethane_disp +
                               self.gnv["cost"] * (self.fuel_buses_demand - biomethane_disp))
        else:
            fuel_buses_cost = self.gnv["cost"] * self.fuel_buses_demand

        # Electrical Energy Commercialization (buy / sell)
        commercialization_electric_energy = demand.copy()
        commercialization_electric_energy_cost = [0.0] * self.n_intervals

        for j in range(0, self.n_intervals):

            # Energy commercialized
            commercialization_electric_energy[j] += battery_energy[j]
            commercialization_electric_energy[j] -= self.solar_energy[j]

            for i in range(0, self.n_generators):
                commercialization_electric_energy[j] -= generators_status[i][j] * generators_rate[i][j]

            # Cost of energy commercialized
            if commercialization_electric_energy[j] > 0:
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.buying_price[j]
            else:
                commercialization_electric_energy_cost[j] = commercialization_electric_energy[j] * self.selling_price[j]

        # CVaR (if enabled)
        if self.enable_cvar:
            z = (biogas_production - self.biogas["mean_production"]) / self.biogas["deviation_production"]
            alpha = norm.cdf(z)
            biogas_cvar = abs(alpha ** -1 * norm.pdf(z) * self.biogas["deviation_production"] - self.biogas["mean_production"])
            diff_biogas = self.biogas["mean_production"] - biogas_cvar
            cvar = diff_biogas * 0.27 * self.buying_price[40]

        # Set interactivity on plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()

        # Intervals (timeline: x-axis)
        intervals = list(range(0, self.n_intervals))
        xtick_label = [dates.num2date(1 + (i * (1.0 / self.n_intervals))).strftime('%H:%M') for i in intervals]
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        # Create a figure (or set it as the current one if already exists) and set its size
        plt.figure(label, figsize=(15, 10), clear=True)

        # Plot electrical demands
        plt_demands = plt.plot(intervals, demand, color='black')
        previous = [0] * self.n_intervals

        # Plot solar energy
        plt_solar_energy = plt.bar(intervals, self.solar_energy, color='tab:orange', bottom=previous)
        previous = [previous[j] + self.solar_energy[j] for j in range(0, self.n_intervals)]

        # Plot generators (biogas and ethanol)
        engine_biogas = [0] * self.n_intervals
        engine_ethanol = [0] * self.n_intervals
        for i in range(0, self.n_generators):
            rate = [generators_rate[i][j] * generators_status[i][j] for j in range(0, self.n_intervals)]
            rate_biogas = [rate[j] * (0.20 * 2 ** (generators_fuel_composition[i][j] - 1) + 0.1 * (generators_fuel_composition[i][j] - 1)) for j in range(0, self.n_intervals)]
            rate_ethanol = [rate[j] - rate_biogas[j] for j in range(0, self.n_intervals)]
            engine_biogas = [engine_biogas[j] + rate_biogas[j] for j in range(0, self.n_intervals)]
            engine_ethanol = [engine_ethanol[j] + rate_ethanol[j] for j in range(0, self.n_intervals)]

        plt_engine_biogas = plt.bar(intervals, engine_biogas, color='limegreen', bottom=previous)
        previous = [previous[j] + engine_biogas[j] for j in range(0, self.n_intervals)]

        plt_engine_ethanol = plt.bar(intervals, engine_ethanol, color='tab:green', bottom=previous)
        previous = [previous[j] + engine_ethanol[j] for j in range(0, self.n_intervals)]

        # Plot battery energy
        battery_energy_use = [max(0, -x) for x in battery_energy]
        plt_battery_energy = plt.bar(intervals, battery_energy_use, color='tab:blue', bottom=previous)
        previous = [previous[j] + battery_energy_use[j] for j in range(0, self.n_intervals)]

        # Purchased electricity
        purchased_electric_energy = [max(0, x) for x in commercialization_electric_energy]
        plt_purchased = plt.bar(intervals, purchased_electric_energy, color='gold', bottom=previous)
        previous = [previous[j] + purchased_electric_energy[j] for j in range(0, self.n_intervals)]

        # Defines the basis for plotting energy use beyond demand
        previous_excess = demand.copy()

        # Battery charging
        battery_charging_energy = [max(0, x) for x in battery_energy]

        battery_charging_values = [battery_charging_energy[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]
        battery_charging_intervals = [intervals[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]
        battery_charging_previous = [previous_excess[j] for j in range(0, self.n_intervals) if battery_charging_energy[j] > 0]

        plt_battery_charging = plt.bar(battery_charging_intervals, battery_charging_values, facecolor=None,
                                       edgecolor="tab:red", hatch="///", fill=False, bottom=battery_charging_previous)

        previous_excess = [previous_excess[j] + battery_charging_energy[j] for j in range(0, self.n_intervals)]

        # Sold electricity
        sold_electric_energy = [max(0, -x) for x in commercialization_electric_energy]

        sold_electric_values = [sold_electric_energy[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]
        sold_electric_intervals = [intervals[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]
        sold_electric_previous = [previous_excess[j] for j in range(0, self.n_intervals) if sold_electric_energy[j] > 0]

        plt_sold = plt.bar(sold_electric_intervals, sold_electric_values, facecolor=None, edgecolor="black",
                           hatch="...", fill=False, bottom=sold_electric_previous)

        previous_excess = [previous_excess[j] + sold_electric_energy[j] for j in range(0, self.n_intervals)]

        # Legend
        plt_demands_proxy = mlines.Line2D([], [], color='black', marker=None)
        plt_objects = [plt_demands_proxy, plt_solar_energy, plt_engine_biogas, plt_engine_ethanol, plt_battery_energy,
                       plt_purchased, plt_sold, plt_battery_charging]
        legends = ["Electrical demands", "PV", "Engine (biogas)", "Engine (ethanol)", "Battery use",
                   "Purchased electricity", "Sold electricity", "Battery charging"]

        plt.legend(plt_objects, legends, loc="upper left", ncol=2, fontsize=14)

        # Other attributes
        plt.xlabel("Time (HH:MM)", fontsize=14)
        plt.ylabel("Energy (kWh/2)", fontsize=14)
        #plt.xticks(intervals, xtick_label, rotation='vertical')
        plt.xticks(intervals, xtick_label, rotation=45)

        # Solution details
        details = ""
        details += "Total cost: {:.2f} {}\n".format(cost, "(infeasible solution)" if infeasibility > 1E-6 else "")
        details += "Biogas production: {:.2f} kWh\n".format(biogas_production)
        details += "Engine biogas consumption: {:.2f} kWh ({:.2f}%)\n".format(biogas_production, (biogas_production / biogas_production) * 100)
        details += "Bus fleet biomethane: {:.2f} kWh ({:.2f}%)\n".format(remaining_biogas, (remaining_biogas / biogas_production) * 100)
        details += "Purchased VNG: {:.2f} m³\n".format(max(0, self.fuel_buses_demand - biomethane_disp))
        details += "Battery initial load: {:.2f} KWh\n".format(self.battery["initial_load"])
        details += "Battery final load: {:.2f} KWh\n".format(battery_load[len(battery_load) - 2])

        if self.enable_cvar:
            details += "CVaR: {:.2f} (Probability level: {:.2f})".format(cvar, alpha)

        #plt.figtext(0.0, 0.0, details, horizontalalignment='left', color='black', fontsize=16)
        #plt.text(0.0, -100, details, ha="left", fontsize=16, wrap=True)
        plt.gcf().text(0.05, 0.005, details, fontsize=14)
        plt.subplots_adjust(bottom=0.26, top=0.99, left=0.05, right=0.99)

        # Show
        if show:
            plt.show(block=block)

            # Allow time to draw
            if not block:
                plt.pause(1e-3)


    def solution_to_json(self, solution, file):

        # Get solution attributes
        generators_status, generators_rate, generators_fuel_composition, battery_energy, biogas_production = solution

        cost, g, h = self.evaluate(solution)
        inf_g, inf_h = self.calculate_infeasibility(g, h)
        infeasibility = sum(inf_g) + sum(inf_h)

        data = dict()

        # cost and infeasibility
        data["cost"] = cost
        data["infeasibility"] = infeasibility
        data["cvar"] = {
            "enabled": self.enable_cvar,
            "epsilon": self.epsilon
        }


        # Generators' data
        data["generators"] = []
        for i in range(0, self.n_generators):
            data_generator = dict()
            data_generator["status"] = generators_status[i]
            data_generator["rate"] = generators_rate[i]
            data_generator["fuel_composition"] = generators_fuel_composition[i]
            data["generators"].append(data_generator)

        data["battery_energy"] = battery_energy
        data["biogas_production"] = biogas_production

        json.dump(data, file, indent=2, )
