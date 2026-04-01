"""
Genetic Algorithm optimiser using DEAP.

Encodes the agriculture resource allocation problem as a real-valued GA
with:
  - Tournament selection
  - Simulated binary crossover (SBX)
  - Polynomial mutation
  - Elitism (best individual preserved each generation)
"""

from __future__ import annotations

import random
import time
import numpy as np

from deap import base, creator, tools, algorithms

from src.optimization.problem import AgricultureOptimizationProblem, ProblemConfig


# DEAP requires module-level creator registration
def _register_deap_types() -> None:
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


_register_deap_types()


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for agriculture resource allocation.

    Parameters
    ----------
    config          : Problem configuration (bounds, weights, etc.)
    pop_size        : Population size
    n_generations   : Number of generations
    cx_prob         : Crossover probability
    mut_prob        : Mutation probability
    tournament_size : Tournament selection size
    eta_cx          : SBX crossover crowding factor
    eta_mut         : Polynomial mutation crowding factor
    """

    def __init__(
        self,
        config: ProblemConfig | None = None,
        pop_size: int = 100,
        n_generations: int = 200,
        cx_prob: float = 0.7,
        mut_prob: float = 0.2,
        tournament_size: int = 3,
        eta_cx: float = 20.0,
        eta_mut: float = 20.0,
    ):
        self.problem       = AgricultureOptimizationProblem(config)
        self.pop_size      = pop_size
        self.n_generations = n_generations
        self.cx_prob       = cx_prob
        self.mut_prob      = mut_prob
        self.tournament_size = tournament_size
        self.eta_cx        = eta_cx
        self.eta_mut       = eta_mut

        self._best_solution:   np.ndarray | None = None
        self._best_fitness:    float = float("inf")
        self._convergence:     list[float] = []
        self._elapsed_seconds: float = 0.0

        self._setup_toolbox()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_toolbox(self) -> None:
        cfg = self.problem.config
        lb  = cfg.lower_bounds.tolist()
        ub  = cfg.upper_bounds.tolist()

        self.toolbox = base.Toolbox()

        # Individual initialiser: uniform random within bounds
        for i, (lo, hi) in enumerate(zip(lb, ub)):
            self.toolbox.register(f"attr_{i}", random.uniform, lo, hi)

        attrs = [getattr(self.toolbox, f"attr_{i}") for i in range(cfg.n_vars)]
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            attrs,
            n=1,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Operators
        self.toolbox.register("evaluate",  self._evaluate)
        self.toolbox.register("mate",      tools.cxSimulatedBinaryBounded,
                              low=lb, up=ub, eta=self.eta_cx)
        self.toolbox.register("mutate",    tools.mutPolynomialBounded,
                              low=lb, up=ub, eta=self.eta_mut,
                              indpb=1.0 / cfg.n_vars)
        self.toolbox.register("select",    tools.selTournament,
                              tournsize=self.tournament_size)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, individual) -> tuple[float]:
        x = np.array(individual)
        return (self.problem.objective(x),)

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimise(self, verbose: bool = True) -> dict:
        """Run the GA and return the best solution summary."""
        random.seed(42)
        np.random.seed(42)

        t0  = time.perf_counter()
        pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min",  np.min)
        stats.register("mean", np.mean)
        hof   = tools.HallOfFame(1)

        self._convergence = []

        for gen in range(1, self.n_generations + 1):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Re-evaluate invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Elitism: preserve best
            hof.update(offspring)
            pop[:] = offspring

            best_gen = min(ind.fitness.values[0] for ind in pop)
            self._convergence.append(best_gen)

            if verbose and gen % 50 == 0:
                mean_f = np.mean([ind.fitness.values[0] for ind in pop])
                print(f"  GA Gen {gen:4d} | best={best_gen:.6f} | mean={mean_f:.6f}")

        self._elapsed_seconds = time.perf_counter() - t0
        best_ind = hof[0]
        self._best_solution  = np.array(best_ind)
        self._best_fitness   = best_ind.fitness.values[0]

        result = self.problem.summarise(self._best_solution, algorithm="Genetic Algorithm")
        result["convergence"]      = self._convergence
        result["elapsed_seconds"]  = round(self._elapsed_seconds, 3)
        result["n_generations"]    = self.n_generations
        result["pop_size"]         = self.pop_size

        if verbose:
            print(f"\n  GA finished in {self._elapsed_seconds:.2f}s")
            print(f"  Best objective: {self._best_fitness:.6f}")
            print(f"  Predicted yield: {result['predicted_yield_t_ha']} t/ha")
            print(f"  Water usage: {result['irrigation_water_l_ha']} L/ha")

        return result
