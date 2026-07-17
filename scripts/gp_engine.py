# -*- coding: utf-8 -*-
"""
gp_engine.py — warm-start 遗传编程（GP）进化引擎

面向单标的时序 alpha 挖掘的标准 GP：
    评估 → 锦标赛选择 → 子树交叉 → 子树变异 → 精英保留

核心特点：
    - warm-start：外部（如 LLM 生成的种子公式）可作为初始种群注入，
      不足部分用 random_tree 补齐；不给则整代随机初始化。
    - fitness_fn / validator_fn 都是外部传入的闭包：
        fitness_fn(node) -> float        （越大越好，例如 IC）
        validator_fn(node) -> bool        （可选，非法个体不进种群）
    - 交叉/变异直接在 expression.Node 树上做「子树交换 / 子树替换」，
      并保护时序算子的「窗口整数常数」参数位——那个位置只能是 int 常数，
      不能被普通子树替换掉，否则时序算子求值会拿不到合法窗口。

依赖地基三模块（裸模块名 import，运行时需 scripts/ 在 sys.path）：
    operators.py / expression.py / feature_registry.py
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from expression import (
    Node,
    random_tree,
    to_formula_string,
    depth as node_depth,
)
# 时序类算子的「窗口位」在最后一个参数，交叉/变异时必须保护该位置
from expression import _WINDOW_OPS  # noqa: 复用地基定义，避免两处清单漂移


@dataclass
class _Individual:
    """种群中的一个个体：表达式树 + 缓存的适应度。"""
    node: Node
    fitness: float = field(default=float('-inf'))


class GPEngine:
    """warm-start 遗传编程进化引擎。"""

    def __init__(
        self,
        feature_names,
        fitness_fn,
        validator_fn=None,
        pop_size: int = 200,
        n_gen: int = 30,
        max_depth: int = 6,
        elite_frac: float = 0.05,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        seed: int = 42,
    ):
        """初始化引擎。

        参数：
            feature_names:  可用特征名列表/集合（供 random_tree 生成子树）
            fitness_fn:     (node)->float，适应度函数，越大越好（外部把
                            fitness.evaluate_fitness 包成只吃 node 的闭包传入）
            validator_fn:   (node)->bool，可选合法性校验；非法个体不进种群
            pop_size:       种群规模
            n_gen:          进化代数
            max_depth:      个体最大深度（随机生成 / 变异子树的深度上限）
            elite_frac:     精英比例（每代直接保留的最优个体占比）
            crossover_rate: 交叉概率
            mutation_rate:  变异概率
            seed:           随机种子（复现）
        """
        self.feature_names = list(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names 不能为空")
        self.fitness_fn = fitness_fn
        self.validator_fn = validator_fn
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.max_depth = int(max_depth)
        self.elite_frac = float(elite_frac)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.rng = np.random.default_rng(seed)

        # 每代至少保留 1 个精英
        self.n_elite = max(1, int(round(self.pop_size * self.elite_frac)))
        # 锦标赛规模：种群不大时取 3，避免过强选择压
        self.tournament_size = max(2, min(3, self.pop_size))

    # ------------------------------------------------------------------
    # 适应度评估
    # ------------------------------------------------------------------
    def _safe_fitness(self, node: Node) -> float:
        """安全计算适应度：异常 / NaN 一律记为 -inf（视为最差个体）。"""
        try:
            val = self.fitness_fn(node)
        except Exception:
            return float('-inf')
        if val is None:
            return float('-inf')
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return float('-inf')
        if not np.isfinite(fval):
            return float('-inf')
        return fval

    def _is_valid(self, node: Node) -> bool:
        """合法性校验：无 validator_fn 时恒 True；有则调用并吞掉异常（异常记非法）。"""
        if self.validator_fn is None:
            return True
        try:
            return bool(self.validator_fn(node))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 初始种群（warm-start）
    # ------------------------------------------------------------------
    def _make_random(self) -> Node:
        """生成一棵合法随机树（若 validator 存在，最多重试若干次）。"""
        for _ in range(20):
            node = random_tree(self.feature_names, self.max_depth, rng=self.rng)
            if self._is_valid(node):
                return node
        # 兜底：即便校验不过也返回最后一棵，避免死循环
        return node

    def _init_population(self, warm_start_pop) -> list[_Individual]:
        """构造初始种群。

        warm_start_pop 给了就作初始种群（深拷贝，过滤非法，超量截断，
        不足用随机树补齐）；没给则全随机。
        """
        pop: list[_Individual] = []
        if warm_start_pop:
            for node in warm_start_pop:
                if len(pop) >= self.pop_size:
                    break
                if node is None:
                    continue
                nd = copy.deepcopy(node)
                if self._is_valid(nd):
                    pop.append(_Individual(node=nd))
        # 不足补随机树
        while len(pop) < self.pop_size:
            pop.append(_Individual(node=self._make_random()))
        return pop

    # ------------------------------------------------------------------
    # 子树定位工具（用于交叉 / 变异）
    # ------------------------------------------------------------------
    def _collect_nodes(self, root: Node):
        """收集树中所有「可作为子树替换点」的节点及其父信息。

        返回 list[(node, parent, child_index)]：
            - node:        节点本身
            - parent:      父节点（根节点的 parent 为 None）
            - child_index: node 在 parent.children 中的下标（根为 -1）

        为保护时序算子的窗口整数常数位，凡是「父节点是 _WINDOW_OPS 且下标为
        最后一个参数」的位置一律排除——这些位置只能是 int 常数窗口，不参与
        普通子树的交叉/变异。
        """
        result = []

        def _walk(node: Node, parent, idx: int):
            # 判断当前位置是否是被保护的窗口位
            protected = (
                parent is not None
                and parent.node_type == 'op'
                and parent.op in _WINDOW_OPS
                and idx == len(parent.children) - 1
            )
            if not protected:
                result.append((node, parent, idx))
            if node.node_type == 'op':
                for i, c in enumerate(node.children):
                    _walk(c, node, i)

        _walk(root, None, -1)
        return result

    def _random_subtree_point(self, root: Node):
        """在树中随机挑一个可替换的子树点（返回 (node,parent,idx)）。"""
        candidates = self._collect_nodes(root)
        i = int(self.rng.integers(0, len(candidates)))
        return candidates[i]

    # ------------------------------------------------------------------
    # 遗传算子
    # ------------------------------------------------------------------
    def _crossover(self, a: Node, b: Node) -> Node:
        """子树交叉：把 b 的一个随机子树复制过来，替换 a 的一个随机子树位置。

        返回新个体（a 的深拷贝上做替换）。窗口整数位受保护、不会被选中。
        """
        child = copy.deepcopy(a)
        donor_node, _, _ = self._random_subtree_point(b)
        donor = copy.deepcopy(donor_node)

        node, parent, idx = self._random_subtree_point(child)
        if parent is None:
            # 选中了根：整棵树被替换为供体子树
            return donor
        parent.children[idx] = donor
        return child

    def _mutate(self, a: Node) -> Node:
        """子树变异：把 a 的一个随机子树替换成一棵新的 random_tree。

        窗口整数位受保护、不会被选中，故不会破坏时序算子的窗口约定。
        """
        child = copy.deepcopy(a)
        node, parent, idx = self._random_subtree_point(child)
        # 新子树深度控制在一个较小的范围，避免树无限膨胀
        sub_depth = int(self.rng.integers(1, max(2, self.max_depth // 2 + 1)) + 1)
        sub_depth = min(sub_depth, self.max_depth)
        new_sub = random_tree(self.feature_names, sub_depth, rng=self.rng)
        if parent is None:
            return new_sub
        parent.children[idx] = new_sub
        return child

    def _make_valid_offspring(self, producer) -> Node:
        """反复调用 producer() 直到产出合法且不过深的个体；重试上限后退回随机树。

        producer: 无参可调用，返回一个 Node。
        """
        for _ in range(10):
            child = producer()
            if node_depth(child) > self.max_depth:
                continue
            if self._is_valid(child):
                return child
        # 退回一棵合法随机树，保证种群始终填满且合法
        return self._make_random()

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------
    def _tournament(self, pop: list[_Individual]) -> _Individual:
        """锦标赛选择：随机抽 tournament_size 个，取适应度最高者。"""
        k = min(self.tournament_size, len(pop))
        idxs = self.rng.integers(0, len(pop), size=k)
        best = pop[int(idxs[0])]
        for j in idxs[1:]:
            cand = pop[int(j)]
            if cand.fitness > best.fitness:
                best = cand
        return best

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    @staticmethod
    def _diversity(pop: list[_Individual]) -> float:
        """种群多样性：不同公式串占种群规模的比例（0~1）。"""
        if not pop:
            return 0.0
        forms = {to_formula_string(ind.node) for ind in pop}
        return len(forms) / len(pop)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, warm_start_pop=None) -> dict:
        """执行进化。

        参数：
            warm_start_pop: list[Node] 或 None。给了作初始种群（不足补随机），
                            没给则整代随机初始化。

        返回 dict：
            {
              'best':         Node,                  # 全局最优个体
              'best_fitness': float,
              'top_k':        list[(Node, float)],   # 末代前 k 名（按适应度降序）
              'trajectory':   list[dict],            # 每代统计 + 当代/全局最优公式
            }
        """
        pop = self._init_population(warm_start_pop)
        # 初始评估
        for ind in pop:
            ind.fitness = self._safe_fitness(ind.node)

        trajectory: list[dict] = []
        # 全局最优（精英保留保证 best_fitness 单调不降）
        global_best = max(pop, key=lambda x: x.fitness)
        global_best = _Individual(copy.deepcopy(global_best.node), global_best.fitness)

        for gen in range(self.n_gen):
            # ---- 记录当代统计 ----
            fits = np.array([ind.fitness for ind in pop], dtype=float)
            finite = fits[np.isfinite(fits)]
            cur_best = float(np.max(fits)) if fits.size else float('-inf')
            mean_fit = float(np.mean(finite)) if finite.size else float('-inf')
            current_best = max(pop, key=lambda x: x.fitness)
            # 全局最优跟踪（当代最优若更好则更新）
            if cur_best > global_best.fitness:
                global_best = _Individual(copy.deepcopy(current_best.node), current_best.fitness)

            trajectory.append({
                'gen': gen,
                # best_fitness 是截至当前的全局最优，保持旧字段语义。
                'best_fitness': global_best.fitness,
                'best_formula': to_formula_string(global_best.node),
                'current_best_fitness': current_best.fitness,
                'current_best_formula': to_formula_string(current_best.node),
                'mean_fitness': mean_fit,
                'diversity': self._diversity(pop),
            })

            # ---- 生成下一代 ----
            # 精英：当代按适应度降序取前 n_elite，直接进入下一代（深拷贝）
            pop_sorted = sorted(pop, key=lambda x: x.fitness, reverse=True)
            next_pop: list[_Individual] = [
                _Individual(copy.deepcopy(e.node), e.fitness)
                for e in pop_sorted[: self.n_elite]
            ]

            # 其余个体由选择 + 交叉/变异产生
            while len(next_pop) < self.pop_size:
                p1 = self._tournament(pop)
                r = self.rng.random()
                if r < self.crossover_rate:
                    p2 = self._tournament(pop)
                    child_node = self._make_valid_offspring(
                        lambda: self._crossover(p1.node, p2.node))
                elif r < self.crossover_rate + self.mutation_rate:
                    child_node = self._make_valid_offspring(
                        lambda: self._mutate(p1.node))
                else:
                    # 直接复制（繁殖），保留父代
                    child_node = copy.deepcopy(p1.node)
                next_pop.append(_Individual(node=child_node))

            # 截断到 pop_size（精英+子代可能刚好，稳妥起见）
            next_pop = next_pop[: self.pop_size]

            # ---- 评估新一代（精英已带缓存 fitness，仍统一重算以防 fitness_fn 有状态）----
            for ind in next_pop:
                ind.fitness = self._safe_fitness(ind.node)
            pop = next_pop

        # ---- 末代收尾：再记一代统计，并刷新全局最优 ----
        fits = np.array([ind.fitness for ind in pop], dtype=float)
        finite = fits[np.isfinite(fits)]
        cur_best = float(np.max(fits)) if fits.size else float('-inf')
        current_best = max(pop, key=lambda x: x.fitness)
        if cur_best > global_best.fitness:
            global_best = _Individual(copy.deepcopy(current_best.node), current_best.fitness)
        trajectory.append({
            'gen': self.n_gen,
            'best_fitness': global_best.fitness,
            'best_formula': to_formula_string(global_best.node),
            'current_best_fitness': current_best.fitness,
            'current_best_formula': to_formula_string(current_best.node),
            'mean_fitness': float(np.mean(finite)) if finite.size else float('-inf'),
            'diversity': self._diversity(pop),
        })

        # ---- top_k：末代 + 全局最优合并去重后取前 k ----
        pool = list(pop) + [global_best]
        pool_sorted = sorted(pool, key=lambda x: x.fitness, reverse=True)
        top_k: list[tuple] = []
        seen: set[str] = set()
        for ind in pool_sorted:
            key = to_formula_string(ind.node)
            if key in seen:
                continue
            seen.add(key)
            top_k.append((ind.node, ind.fitness))
            if len(top_k) >= 10:
                break

        return {
            'best': global_best.node,
            'best_fitness': global_best.fitness,
            'top_k': top_k,
            'trajectory': trajectory,
        }
