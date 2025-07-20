import concurrent.futures
from typing import Any, Generic, Callable, Collection

import numpy as np
from tqdm import tqdm

from search.base import AbstractParameterSearcher, ParamType


class Agent:
    """遺伝的アルゴリズムのエージェントを表すクラス"""

    def __init__(self, indexes: tuple[int, ...], keys: tuple[str, ...]):
        self._indexes = indexes
        self._keys = keys

        # バリデーション
        assert isinstance(self._indexes, tuple), (type(self._indexes), self._indexes)
        assert all(isinstance(gene, int) for gene in self._indexes), self._indexes
        assert isinstance(self._keys, tuple), (type(self._keys), self._keys)
        assert len(self._indexes) == len(self._keys), (len(self._indexes), len(self._keys))

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> int:
        return self._indexes[index]

    def __iter__(self):
        return iter(self._indexes)

    def to_list(self) -> list[int]:
        """リスト形式に変換"""
        return list(self._indexes)

    @classmethod
    def from_list(cls, indexes: list[int], keys: list[str]) -> "Agent":
        """リストからエージェントを作成"""
        return cls(tuple(indexes), tuple(keys))

    def to_param_dict(self, values_lst: dict[str, list[Any]]) -> dict[str, Any]:
        """エージェントをパラメータ辞書に変換"""
        param_dict = {}
        for i, key in enumerate(self._keys):
            param_dict[key] = values_lst[key][self._indexes[i]]
        return param_dict

    def crossover(self, other: "Agent", rng: np.random.RandomState) -> tuple["Agent", "Agent"]:
        """
        ランダム交叉（Random crossover）
        親の遺伝子座をランダムに選んで交換する
        """
        child1_genes = list(self._indexes)
        child2_genes = list(other._indexes)
        for i in range(len(self._indexes)):
            if rng.random() < 0.5:  # 各遺伝子座で50%の確率で交換
                child1_genes[i], child2_genes[i] = child2_genes[i], child1_genes[i]
        return Agent.from_list(child1_genes, list(self._keys)), \
            Agent.from_list(child2_genes, list(self._keys))

    def mutate(self, rng: np.random.RandomState, values_lst: dict[str, list[Any]],
               mutation_rate: float = 0.1) -> "Agent":
        """
        突然変異
        各遺伝子座で突然変異率に基づいてランダムな値に変更
        """
        mutated_genes = list(self._indexes)
        for i, key in enumerate(self._keys):
            if rng.random() < mutation_rate:
                # 該当するパラメータの取りうる値の範囲内でランダムに選択
                mutated_genes[i] = rng.randint(0, len(values_lst[key]))
        return Agent.from_list(mutated_genes, list(self._keys))


class Population:
    """遺伝的アルゴリズムの集団を表すクラス"""

    def __init__(self, agents: list[Agent], values_lst: dict[str, list[Any]],
                 param_mapper: Callable[[dict[str, Any]], ParamType],
                 rng: np.random.RandomState):
        self._agents = agents
        self._values_lst = values_lst
        self._param_mapper = param_mapper
        self._rng = rng
        self._scores: dict[Agent, float] = {}

    @classmethod
    def generate_random(cls, size: int, keys: list[str], values_lst: dict[str, list[Any]],
                        param_mapper: Callable[[dict[str, Any]], ParamType],
                        rng: np.random.RandomState) -> "Population":
        """ランダムな集団を生成"""
        agents = []
        for _ in range(size):
            indexes = []
            for key in keys:
                indexes.append(rng.randint(0, len(values_lst[key])))
            agents.append(Agent.from_list(indexes, keys))
        return cls(agents, values_lst, param_mapper, rng)

    def evaluate_parallel(self, scorer: Callable[[ParamType], float], *, n_workers: int) \
            -> dict[Agent, float]:
        """集団を評価し、スコアを返す"""
        scores = {}
        param_objects_to_eval = []
        agent_to_param_dict_map = {}

        for agent in self._agents:
            param_dict = agent.to_param_dict(self._values_lst)
            param_obj = self._param_mapper(param_dict)
            param_objects_to_eval.append(param_obj)
            agent_to_param_dict_map[agent] = param_obj

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(scorer, p_obj): (agent, p_obj)
                       for agent, p_obj in agent_to_param_dict_map.items()}

            bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                       desc="Evaluating Population")
            try:
                for future in bar:
                    agent_key, p_obj = futures[future]
                    try:
                        score = future.result()
                    except:
                        print("Error occurred while evaluating", p_obj)
                        raise
                    scores[agent_key] = score
            except KeyboardInterrupt:
                print("Stopping evaluation ...")
                executor.shutdown(wait=True, cancel_futures=True)
                raise

        self._scores = scores
        return scores

    def selection(self, tournament_size: int) -> list[Agent]:
        """
        トーナメント選択により次世代の親を選択する。
        スコアが高いほど選ばれやすい。
        """
        selected_parents = []
        for _ in range(len(self._agents)):
            # インデックスをランダムに選択してから、対応するエージェントを取得
            population_indices = list(range(len(self._agents)))
            tournament_indices = self._rng.choice(population_indices,
                                                  size=min(tournament_size, len(self._agents)),
                                                  replace=False)
            tournament_candidates = [self._agents[i] for i in tournament_indices]

            # 各候補のスコアを取得し、最も良いスコアを持つ候補を選ぶ
            best_candidate = None
            best_score_in_tournament = -float('inf')

            for candidate_agent in tournament_candidates:
                current_score = self._scores.get(candidate_agent, -float('inf'))
                if current_score > best_score_in_tournament:
                    best_score_in_tournament = current_score
                    best_candidate = candidate_agent

            if best_candidate is not None:
                selected_parents.append(best_candidate)
        return selected_parents

    def evolve(self, crossover_rate: float, mutation_rate: float) -> "Population":
        """集団を進化させて新しい集団を生成"""
        selected_parents = self.selection(tournament_size=5)  # デフォルト値

        next_agents = []
        # 交叉と突然変異で次世代を生成
        for i in range(0, len(self._agents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1] if i + 1 < len(self._agents) else \
                selected_parents[i]  # 奇数個体数の場合に対応

            # crossover_rateに基づいて交叉を実行するかどうかを判定
            if self._rng.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2, self._rng)
            else:
                child1, child2 = parent1, parent2  # 交叉しない場合は親をそのまま使用

            next_agents.append(child1.mutate(self._rng, self._values_lst, mutation_rate))
            if len(next_agents) < len(self._agents):  # 個体数を超えないように
                next_agents.append(child2.mutate(self._rng, self._values_lst, mutation_rate))

        return Population(next_agents, self._values_lst, self._param_mapper, self._rng)

    def get_best_agent(self) -> tuple[Agent, float]:
        """最良のエージェントとそのスコアを取得"""
        if not self._scores:
            raise ValueError("Population has not been evaluated yet")

        best_agent = max(self._scores.items(), key=lambda x: x[1])[0]
        best_score = self._scores[best_agent]
        return best_agent, best_score


class GAParameterSearcher(AbstractParameterSearcher, Generic[ParamType]):
    """
    遺伝的アルゴリズム (GA) を用いたパラメータ探索器。
    """

    def __init__(
            self,
            scorer: Callable[[ParamType], float],
            param_grid: dict[str, Collection[Any]],
            param_mapper: Callable[[dict[str, Any]], ParamType],
            *,
            n_pop: int = 30,  # 世代ごとの個体数
            n_gen: int = 10,  # 世代数
            mutation_rate: float = 0.1,  # 突然変異率
            crossover_rate: float = 0.8,  # 交叉率
            tournament_size: int = 5,  # トーナメント選択のサイズ
            seed: int | None = None  # 乱数シード
    ):
        super().__init__(scorer, param_grid, param_mapper)
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self._values_lst = {key: list(values) for key, values in self._param_grid.items()}
        self._keys = list(param_grid.keys())

    def _run_search(self, *, n_workers: int) -> None:
        # 初期集団を生成
        current_population = Population.generate_random(
            size=self.n_pop,
            keys=self._keys,
            values_lst=self._values_lst,
            param_mapper=self._param_mapper,
            rng=self.rng
        )

        for generation in range(self.n_gen):
            # 集団を評価
            try:
                current_population.evaluate_parallel(scorer=self._eval_score, n_workers=n_workers)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break

            # 現在世代のベストスコアとベストエージェントを更新
            current_best_agent, current_best_score = current_population.get_best_agent()
            best_param_dict = current_best_agent.to_param_dict(self._values_lst)
            best_param = self._param_mapper(best_param_dict)
            self._history.append((best_param, current_best_score))

            print(
                f"Generation {generation + 1}/{self.n_gen}: "
                f"Best score = {current_best_score:.3f}"
            )
            print(best_param)

            if generation < self.n_gen - 1:  # 最終世代では次世代を生成しない
                # 集団を進化させて新しい集団を生成
                current_population = current_population.evolve(
                    crossover_rate=self.crossover_rate,
                    mutation_rate=self.mutation_rate
                )

        if not self._history:
            raise RuntimeError("No valid parameter found after GA search")
