import concurrent.futures
import concurrent.futures
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Generic, Callable, Iterator, Iterable, Literal

import numpy as np
from tqdm import tqdm

from search.base import AbstractParameterSearcher, ParamType


class Agent:  # immutable
    """Class representing an agent in a genetic algorithm."""

    def __init__(self, index_it: Iterable[int]):
        self._indexes = tuple(index_it)
        assert isinstance(self._indexes, tuple), (type(self._indexes), self._indexes)
        assert all(isinstance(gene, int) for gene in self._indexes), self._indexes

    def __len__(self) -> int:
        """Return the number of genes."""
        return len(self._indexes)

    def __getitem__(self, index: int) -> int:
        """Return the gene value at the specified index."""
        return self._indexes[index]

    def to_list(self) -> list[int]:
        return list(self._indexes)


class AgentUtil(Generic[ParamType]):
    """
    Utility class for handling agents in a genetic algorithm.
    Provides methods for creating, mutating, crossing over, and evaluating agents.
    """

    def __init__(
            self,
            *,
            domain: OrderedDict[str, Sequence[Any]],
            rng: np.random.RandomState,
            is_feasible: Callable[[ParamType], bool],
            param_mapper: Callable[[dict[str, Any]], ParamType],
            param_scorer: Callable[[ParamType], float],
    ):
        """
        Initialize the AgentUtil.

        Args:
            domain: OrderedDict mapping parameter names to lists of possible values.
            rng: Random number generator (numpy RandomState).
            is_feasible: Function to check if a parameter set is feasible.
            param_mapper: Function to map a parameter dictionary to a parameter object.
            param_scorer: Function to score a parameter object.
        """
        assert isinstance(domain, OrderedDict), (type(domain), domain)
        self._domain = domain
        self._rng = rng
        self._is_feasible = is_feasible
        self._param_mapper = param_mapper
        self._param_scorer = param_scorer

        self._max_attempt_error = 10000  # Maximum number of attempts to generate a feasible agent before raising an error
        self._max_attempt_warning = 1000  # Number of attempts after which a warning is printed

    def as_param(self, a: Agent) -> ParamType:
        """
        Convert an Agent to a parameter dictionary.

        Args:
            a: The agent to convert.

        Returns:
            A dictionary mapping parameter names to their values.
        """
        # Make _as_dict public so it can be used from Population as well
        dct = {key: list(values)[a[i]] for i, (key, values) in enumerate(self._domain.items())}
        return self._param_mapper(dct)

    def is_feasible(self, a: Agent) -> bool:
        """
        Check if an agent is feasible.

        Args:
            a: The agent to check.

        Returns:
            True if the agent is feasible, False otherwise.
        """
        # Check if each gene is within the domain
        for i, (_, values) in enumerate(self._domain.items()):
            if a[i] < 0 or len(values) <= a[i]:
                return False
        # Check additional feasibility constraints
        return self._is_feasible(self.as_param(a))

    def eval_score(self, a: Agent) -> float:
        """
        Score an agent.

        Args:
            a: The agent to score.

        Returns:
            The score as a float.
        """
        return self._param_scorer(self.as_param(a))

    def _create_random_impl(self) -> Agent:
        """
        Create a random agent (not guaranteed to be feasible).

        Returns:
            A new Agent instance.
        """
        indexes = tuple(int(self._rng.randint(0, len(values))) for values in self._domain.values())
        return Agent(indexes)

    def create_random(self) -> Agent:
        """
        Create a random feasible agent.

        Returns:
            A feasible Agent instance.

        Raises:
            RuntimeError: If a feasible agent cannot be created after max attempts.
        """
        for attempt in range(self._max_attempt_error):
            agent = self._create_random_impl()
            if self.is_feasible(agent):
                return agent
            if attempt >= self._max_attempt_warning:
                warnings.warn(
                    f"Could not create a feasible agent after {attempt} attempts, continuing to try...",
                    UserWarning,
                )
        raise RuntimeError(
            f"Failed to create a feasible agent after {self._max_attempt_error} attempts"
        )

    def _create_from_param_impl(self, param: ParamType) -> Agent:
        """
        Create an agent from a parameter object (not guaranteed to be feasible).

        Args:
            param: The parameter object.

        Returns:
            An Agent instance.

        Raises:
            ValueError: If the parameter is not in the domain.
        """
        indexes = []
        for key, values in self._domain.items():
            if not hasattr(param, key):
                raise ValueError(f"param.{key} not found")
            if getattr(param, key) not in values:
                raise ValueError(
                    f"param.{key} = {getattr(param, key)} not in domain {list(values)}")
            indexes.append(values.index(getattr(param, key)))
        return Agent(tuple(indexes))

    def create_from_param(self, param: ParamType) -> Agent:
        """
        Create a feasible agent from a parameter object.

        Args:
            param: The parameter object.

        Returns:
            A feasible Agent instance.

        Raises:
            ValueError: If the parameter is not feasible.
        """
        a = self._create_from_param_impl(param)
        if not self.is_feasible(a):
            raise ValueError(f"param = {param} is not feasible")
        return a

    def _uniform_crossover_impl(self, a_1: Agent, a_2: Agent) -> tuple[Agent, Agent]:
        """
        Perform uniform crossover between two agents (not guaranteed to be feasible).

        Args:
            a_1: The first parent agent.
            a_2: The second parent agent.

        Returns:
            A tuple of two new Agent instances.
        """
        assert len(a_1) == len(a_2), (len(a_1), len(a_2))
        g_1 = a_1.to_list()
        g_2 = a_2.to_list()
        for i in range(len(a_1)):
            if self._rng.random() < 0.5:
                g_1[i], g_2[i] = g_2[i], g_1[i]
        c_1, c_2 = Agent(g_1), Agent(g_2)
        assert len(c_1) == len(c_2) == len(a_1)
        return c_1, c_2

    def _two_point_crossover_impl(self, a_1: Agent, a_2: Agent) -> tuple[Agent, Agent]:
        """
        Perform two-point crossover between two agents (not guaranteed to be feasible).

        Args:
            a_1: The first parent agent.
            a_2: The second parent agent.

        Returns:
            A tuple of two new Agent instances.
        """
        assert len(a_1) == len(a_2), (len(a_1), len(a_2))
        g_1 = a_1.to_list()
        g_2 = a_2.to_list()
        # Select two random points in the genome
        i, j = self._rng.choice(range(len(a_1)), size=2, replace=False)
        if i > j:
            i, j = j, i
        # Swap the genes between the two points
        g_1[i:j], g_2[i:j] = g_2[i:j], g_1[i:j]
        c_1, c_2 = Agent(g_1), Agent(g_2)
        assert len(c_1) == len(c_2) == len(a_1)
        return c_1, c_2

    def crossover(
            self,
            a_1: Agent,
            a_2: Agent,
            *,
            crossover_type: Literal["uniform", "two-point"],
    ) -> tuple[Agent, Agent]:
        """
        Perform crossover between two agents and ensure the result is feasible.

        Args:
            a_1: The first parent agent.
            a_2: The second parent agent.
            crossover_type: Type of crossover to use.

        Returns:
            A tuple of two feasible Agent instances.

        Raises:
            RuntimeError: If feasible children cannot be created after max attempts.
            ValueError: If the crossover type is invalid.
        """
        for attempt in range(self._max_attempt_error):
            if crossover_type == "uniform":
                c_1, c_2 = self._uniform_crossover_impl(a_1, a_2)
            elif crossover_type == "two-point":
                c_1, c_2 = self._two_point_crossover_impl(a_1, a_2)
            else:
                raise ValueError(f"Invalid crossover type: {crossover_type}")
            if self.is_feasible(c_1) and self.is_feasible(c_2):
                return c_1, c_2
            if attempt >= self._max_attempt_warning:
                warnings.warn(
                    f"Could not crossover agents after {attempt} attempts, continuing to try...",
                    UserWarning,
                )
        raise RuntimeError(f"Failed to crossover agents after {self._max_attempt_error} attempts")

    def _mutate_impl(self, a: Agent) -> Agent:
        """
        Mutate an agent (not guaranteed to be feasible).

        Args:
            a: The agent to mutate.

        Returns:
            A new Agent instance (possibly mutated).
        """
        mutated_genes = a.to_list()
        # Extract indices whose domain has multiple values (i.e., mutable genes)
        mutable_indices = [i for i, values in enumerate(self._domain.values()) if len(values) > 1]
        # Randomly select a mutable gene to mutate
        i = self._rng.choice(mutable_indices)
        values = list(self._domain.values())[i]
        current_index = mutated_genes[i]
        # Select a value different from the current one
        candidates: list[int] = [idx for idx in range(len(values)) if idx != current_index]
        mutated_genes[i] = int(self._rng.choice(candidates))
        # If no mutable indices or mutation occur, return as is
        new_a = Agent(mutated_genes)
        assert len(new_a) == len(a)
        return new_a

    def mutate(self, a: Agent) -> Agent:
        """
        Mutate an agent and ensure the result is feasible.

        Args:
            a: The agent to mutate.

        Returns:
            A feasible Agent instance.

        Raises:
            RuntimeError: If a feasible mutated agent cannot be created after max attempts.
        """
        for attempt in range(self._max_attempt_error):
            a = self._mutate_impl(a)
            if self.is_feasible(a):
                return a
            if attempt >= self._max_attempt_warning:
                warnings.warn(
                    f"Could not mutate agent after {attempt} attempts, continuing to try...",
                    UserWarning,
                )
        raise RuntimeError(f"Failed to mutate agent after {self._max_attempt_error} attempts")


class Population:  # mutable
    """Class representing a population in a genetic algorithm."""

    def __init__(
            self,
            *,
            agent_util: AgentUtil,
            rng: np.random.RandomState,
    ):
        self._agents: list[Agent] = []
        self._agent_util = agent_util
        self._rng = rng

        # Cache of agent scores from the last evaluation (single or parallel)
        self._score_cache: dict[Agent, float] = {}

    def initialize_population(
            self,
            *,
            size: int,
    ) -> None:
        """
        Initialize the population with random agents.

        Args:
            size: The number of agents in the population. Must be positive and even.
        """
        assert size > 0, "size must be greater than 0"
        assert size % 2 == 0, "size must be even"

        while len(self._agents) < size:
            agent = self._agent_util.create_random()
            self._agents.append(agent)

    def evaluate_single(self) -> Iterable[tuple[ParamType, float]]:
        """
        Evaluate all agents in the population sequentially.

        Yields:
            Tuples of (agent, score).
        """
        scores = {}
        for agent in self._agents:
            score = self._agent_util.eval_score(agent)
            scores[agent] = score
            yield agent, score
        self._score_cache = scores

    def evaluate_parallel(self, n_workers: int) -> Iterable[tuple[ParamType, float]]:
        """
        Evaluate all agents in the population in parallel.

        Args:
            n_workers: Number of worker processes.

        Yields:
            Tuples of (agent, score).
        """
        scores = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._agent_util.eval_score, agent): agent for agent in
                       self._agents}
            for future in concurrent.futures.as_completed(futures):
                agent_key = futures[future]
                try:
                    score = future.result()
                except Exception:
                    warnings.warn(f"Error occurred while evaluating {agent_key}", UserWarning)
                    raise
                scores[agent_key] = score
                yield agent_key, score
        self._score_cache = scores

    def selection(self, *, total_size: int, tournament_size: int) -> list[tuple[Agent, Agent]]:
        """
        Select parents for the next generation using tournament selection.
        Agents with higher scores are more likely to be selected.

        Args:
            total_size: Number of parent pairs to select.
            tournament_size: Number of agents in each tournament.

        Returns:
            List of tuples, each containing two selected parent agents.
        """
        assert tournament_size >= 2, "tournament_size must be greater than 2"
        pop_indexes = list(range(len(self._agents)))
        selected_parents: list[tuple[Agent, Agent]] = []
        for _ in range(total_size):
            # Randomly select indices for the tournament, then get the corresponding agents
            sub_indexes = self._rng.choice(
                pop_indexes,
                size=min(tournament_size, len(pop_indexes)),
                replace=False,
            )
            sub_agents = [self._agents[i] for i in sub_indexes]
            scores = [self._score_cache[agent] for agent in sub_agents]
            a_second_index, a_top_index = np.argsort(scores)[-2:]
            selected_parents.append((sub_agents[a_top_index], sub_agents[a_second_index]))

        assert len(selected_parents) == total_size

        return selected_parents

    def evolve(
            self,
            *,
            crossover_rate: float,
            crossover_type: Literal["uniform", "two-point"],
            mutation_rate: float,
            tournament_size: int,
    ) -> None:
        """
        Evolve the population to the next generation.

        Args:
            crossover_rate: Probability of crossover between parents.
            crossover_type: Type of crossover to use.
            mutation_rate: Probability of mutation for each child.
            tournament_size: Number of agents in each tournament for selection.
        """
        assert len(self._agents) % 2 == 0
        selected_parents \
            = self.selection(total_size=len(self._agents) // 2, tournament_size=tournament_size)
        next_agents = []
        for p_1, p_2 in selected_parents:
            if self._rng.random() < crossover_rate:
                c_1, c_2 = self._agent_util.crossover(p_1, p_2, crossover_type=crossover_type)
            else:
                c_1, c_2 = p_1, p_2
            if self._rng.random() < mutation_rate:
                c_1 = self._agent_util.mutate(c_1)
            if self._rng.random() < mutation_rate:
                c_2 = self._agent_util.mutate(c_2)
            next_agents.append(c_1)
            next_agents.append(c_2)
        # Two children are added for each parent pair, so len(next_agents) is always even.
        # Since size is even in initialize_population, this assertion always holds.
        assert len(self._agents) == len(next_agents)
        self._agents = next_agents

    def iter_param_and_scores(self) -> Iterator[tuple[ParamType, float]]:
        """
        Iterator over parameters and their scores for all agents in the last evaluation.

        Yields:
            Tuples of (parameter, score).
        """
        for agent, score in self._score_cache.items():
            param = self._agent_util.as_param(agent)
            yield param, score


class GAParameterSearcher(AbstractParameterSearcher, Generic[ParamType]):
    """
    Parameter searcher using Genetic Algorithm (GA).
    """

    def __init__(
            self,
            scorer: Callable[[ParamType], float],
            param_grid: dict[str, Sequence[Any]],
            param_mapper: Callable[[dict[str, Any]], ParamType],
            *,
            n_pop: int = 30,  # Number of individuals per generation
            n_gen: int = 10,  # Number of generations
            mutation_rate: float = 0.01,  # Mutation rate
            crossover_rate: float = 0.9,  # Crossover rate
            crossover_type: Literal["uniform", "two-point"],  # Type of crossover to use
            tournament_size: int = 3,  # Tournament selection size
            seed: int | None = None,  # Random seed
            constraint_predicate: Callable[[ParamType], bool] | None = None,
            # Feasibility predicate
    ):
        """
        Initialize the GAParameterSearcher.

        Args:
            scorer: Function to evaluate the score of a parameter.
            param_grid: Dictionary mapping parameter names to possible values.
            param_mapper: Function to map a parameter dictionary to a parameter object.
            n_pop: Number of individuals per generation.
            n_gen: Number of generations.
            mutation_rate: Probability of mutation for each child.
            crossover_rate: Probability of crossover between parents.
            crossover_type: Type of crossover to use.
                If the combination of adjacent parameters is important, use "two-point".
                Otherwise, "uniform" is recommended.
            tournament_size: Number of agents in each tournament for selection.
            seed: Random seed for reproducibility.
            constraint_predicate: Function to check if a parameter set is feasible.
        """
        super().__init__(scorer, param_grid, param_mapper,
                         constraint_predicate=constraint_predicate)
        self._n_pop = n_pop
        self._n_gen = n_gen
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._crossover_type = crossover_type
        self._tournament_size = tournament_size

        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()

    def _run_search(self, *, n_workers: int) -> None:
        """
        Run the genetic algorithm search.

        Args:
            n_workers: Number of parallel workers to use for evaluation.
        """
        # Generate the initial population
        current_population = Population(
            agent_util=AgentUtil(
                domain=OrderedDict(self._param_grid),
                rng=self._rng,
                is_feasible=self._is_feasible,
                param_mapper=self._param_mapper,
                param_scorer=self._eval_score,
            ),
            rng=self._rng,
        )
        current_population.initialize_population(
            size=self._n_pop,
        )

        for generation in range(self._n_gen):
            # Evaluate the current population (single or parallel)
            if n_workers == 1:
                it = current_population.evaluate_single()
            else:
                it = current_population.evaluate_parallel(n_workers=n_workers)
            bar = tqdm(it, total=self._n_pop,
                       desc=f"[GA] Evaluating Population (gen={generation + 1}, n_workers={n_workers})")
            scores = {}
            try:
                for agent, score in bar:
                    scores[agent] = score
                    bar.set_description(
                        f"[GA] Evaluating Population (gen={generation + 1}, best={max(scores.values()):.4f})")
            except KeyboardInterrupt:
                warnings.warn("KeyboardInterrupt", UserWarning)
                break

            # Record all agents' parameters and scores
            for param, score in current_population.iter_param_and_scores():
                self.add_record(param, score)

            print(
                f"\nGeneration {generation + 1}/{self._n_gen}\n"
                f" - best: {self.best_score:.3f}\n"
                f" - param: {self.best_param!r}\n"
            )

            # Do not generate the next generation for the last generation
            if generation < self._n_gen - 1:
                # Evolve the population to generate the next generation
                current_population.evolve(
                    crossover_rate=self._crossover_rate,
                    crossover_type=self._crossover_type,
                    mutation_rate=self._mutation_rate,
                    tournament_size=self._tournament_size,
                )

        if self.is_empty:
            raise RuntimeError("No valid parameter found after GA search")
