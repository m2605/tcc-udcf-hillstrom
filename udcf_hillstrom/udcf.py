"""
Unified Discriminative Causal Forest (UDCF)
Implementacao do modelo de estimacao de CATE proposto em:

  Ai, M., Li, B., Gong, H., Yu, Q., Xue, S., Zhang, Y., Zhang, Y., Jiang, P. (2022).
  "LBCF: A Large-Scale Budget-Constrained Causal Forest Algorithm."
  Proceedings of the ACM Web Conference 2022 (WWW '22), Secao 4.2.1.
  Repositorio dos autores: https://github.com/www2022paper/WWW-2022-PAPER-SUPPLEMENTARY-MATERIALS

Esta e uma reimplementacao independente, escrita do zero em Python/NumPy a partir
da descricao do Algoritmo 1 e das Equacoes 3 e 4 do artigo (nao e uma copia do
codigo C++/GRF dos autores, que nao foi utilizado aqui).

Etapa coberta: apenas a primeira etapa do LBCF (estimacao de CATE a nivel de
usuario via UDCF). A segunda etapa do artigo (otimizacao com orcamento via DGB)
nao faz parte deste script.
"""

from __future__ import annotations

import numpy as np


RIDGE = 1e-6  # regularizacao numerica para evitar matrizes singulares


def _local_ols(T: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regressao linear local (Algoritmo 1, linhas 2-8).

    Centraliza T e Y pela media do no e resolve a normal equation.
    Retorna theta_P (K,), residuo R (n,) e Q = Ttil @ inv(A_P) (n, K).
    """
    n, k = T.shape
    Tbar = T.mean(axis=0)
    Ybar = Y.mean()
    Ttil = T - Tbar
    Ytil = Y - Ybar
    A = Ttil.T @ Ttil + RIDGE * np.eye(k)
    Ainv = np.linalg.inv(A)
    theta = Ainv @ (Ttil.T @ Ytil)
    R = Ytil - Ttil @ theta
    Q = Ttil @ Ainv  # Ainv e simetrica -> Ainv.T == Ainv
    return theta, R, Q


class _Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "leaf_id")

    def __init__(self):
        self.is_leaf = True
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.leaf_id = None


class UDCFTree:
    """Uma arvore UDCF: estrutura unificada, com split Inter + Intra (Secao 4.2.1)."""

    def __init__(self, max_depth=5, min_leaf_per_arm=30, n_candidates=19, m_top=5,
                 mtry=None, random_state=None):
        self.max_depth = max_depth
        self.min_leaf_per_arm = min_leaf_per_arm
        self.n_candidates = n_candidates
        self.m_top = m_top
        self.mtry = mtry
        self.rng = np.random.default_rng(random_state)
        self.root: _Node | None = None
        self.n_leaves = 0

    # ---------- construcao (usa somente a amostra de split, honestidade) ----------
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        idx = np.arange(X.shape[0])
        self.root = self._build(X, T, Y, idx, depth=0)
        return self

    def _valid_split(self, T_left, T_right):
        k = T_left.shape[1]
        # cada braco (controle + K tratamentos) precisa de amostra minima nos dois filhos
        for T_child in (T_left, T_right):
            n_control = (T_child.sum(axis=1) == 0).sum()
            if n_control < self.min_leaf_per_arm:
                return False
            for j in range(k):
                if (T_child[:, j] == 1).sum() < self.min_leaf_per_arm:
                    return False
        return True

    def _candidate_thresholds(self, col: np.ndarray) -> np.ndarray:
        uniq = np.unique(col)
        if uniq.size <= 1:
            return np.array([])
        if uniq.size <= self.n_candidates + 1:
            return (uniq[:-1] + uniq[1:]) / 2.0
        qs = np.linspace(0, 1, self.n_candidates + 2)[1:-1]
        return np.quantile(col, qs)

    def _build(self, X, T, Y, idx, depth) -> _Node:
        node = _Node()
        n = idx.size
        k = T.shape[1]

        enough = n >= 2 * self.min_leaf_per_arm * (k + 1)
        if depth >= self.max_depth or not enough:
            node.leaf_id = self.n_leaves
            self.n_leaves += 1
            return node

        Xn, Tn, Yn = X[idx], T[idx], Y[idx]
        theta_P, R, Q = _local_ols(Tn, Yn)
        rho = R[:, None] * Q  # (n, K), Algoritmo 1 linha 10

        n_features = X.shape[1]
        feat_pool = np.arange(n_features)
        if self.mtry is not None and self.mtry < n_features:
            feat_pool = self.rng.choice(n_features, size=self.mtry, replace=False)

        candidates = []  # (inter_score, feature, threshold, left_mask)
        for f in feat_pool:
            col = Xn[:, f]
            thresholds = self._candidate_thresholds(col)
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                nL, nR = left_mask.sum(), right_mask.sum()
                if nL == 0 or nR == 0:
                    continue
                if not self._valid_split(Tn[left_mask], Tn[right_mask]):
                    continue
                S_left = rho[left_mask].sum(axis=0)
                S_right = rho[right_mask].sum(axis=0)
                inter = (S_left @ S_left) / nL + (S_right @ S_right) / nR
                candidates.append((inter, f, thr, left_mask))

        if not candidates:
            node.leaf_id = self.n_leaves
            self.n_leaves += 1
            return node

        candidates.sort(key=lambda c: c[0], reverse=True)
        top = candidates[: self.m_top]

        best_intra = -np.inf
        best = None
        for _, f, thr, left_mask in top:
            right_mask = ~left_mask
            theta_L, _, _ = _local_ols(Tn[left_mask], Yn[left_mask])
            theta_R, _, _ = _local_ols(Tn[right_mask], Yn[right_mask])
            tbar_L, tbar_R = theta_L.mean(), theta_R.mean()
            intra = np.sum((theta_L - tbar_L) ** 2) + np.sum((theta_R - tbar_R) ** 2)
            if intra > best_intra:
                best_intra = intra
                best = (f, thr, left_mask)

        f, thr, left_mask = best
        node.is_leaf = False
        node.feature = f
        node.threshold = thr
        node.left = self._build(X, T, Y, idx[left_mask], depth + 1)
        node.right = self._build(X, T, Y, idx[~left_mask], depth + 1)
        return node

    # ---------- roteamento (usado tanto pela amostra "leaf" quanto na predicao) ----------
    def apply(self, X: np.ndarray) -> np.ndarray:
        leaf_ids = np.empty(X.shape[0], dtype=np.int64)
        self._apply_rec(self.root, X, np.arange(X.shape[0]), leaf_ids)
        return leaf_ids

    def _apply_rec(self, node: _Node, X, idx, out):
        if node.is_leaf:
            out[idx] = node.leaf_id
            return
        col = X[idx, node.feature]
        left = col <= node.threshold
        if left.any():
            self._apply_rec(node.left, X, idx[left], out)
        if (~left).any():
            self._apply_rec(node.right, X, idx[~left], out)


class UDCFForest:
    """Floresta de arvores UDCF com estimacao honesta (split/leaf sample) e
    predicao via regressao ponderada estilo Generalized Random Forest (GRF),
    conforme descrito ao final da Secao 4.2.1 do artigo."""

    def __init__(self, n_trees=200, subsample_frac=0.5, honesty_frac=0.5,
                 max_depth=5, min_leaf_per_arm=30, n_candidates=19, m_top=5,
                 mtry=None, random_state=0):
        self.n_trees = n_trees
        self.subsample_frac = subsample_frac
        self.honesty_frac = honesty_frac
        self.max_depth = max_depth
        self.min_leaf_per_arm = min_leaf_per_arm
        self.n_candidates = n_candidates
        self.m_top = m_top
        self.mtry = mtry
        self.random_state = random_state
        self.trees: list[UDCFTree] = []
        self._leaf_M: list[np.ndarray] = []  # por arvore: (n_leaves, K+1, K+1)
        self._leaf_v: list[np.ndarray] = []  # por arvore: (n_leaves, K+1)
        self._leaf_n: list[np.ndarray] = []  # por arvore: (n_leaves,)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        n, k = X.shape[0], T.shape[1]
        rng = np.random.default_rng(self.random_state)
        Z = np.hstack([np.ones((n, 1)), T])  # (n, K+1) para a regressao final com intercepto

        for b in range(self.n_trees):
            sub_size = int(self.subsample_frac * n)
            sub = rng.choice(n, size=sub_size, replace=False)
            rng.shuffle(sub)
            cut = int(self.honesty_frac * sub_size)
            split_idx, leaf_idx = sub[:cut], sub[cut:]

            tree = UDCFTree(max_depth=self.max_depth, min_leaf_per_arm=self.min_leaf_per_arm,
                             n_candidates=self.n_candidates, m_top=self.m_top,
                             mtry=self.mtry, random_state=rng.integers(1 << 31))
            tree.fit(X[split_idx], T[split_idx], Y[split_idx])

            leaf_ids = tree.apply(X[leaf_idx])
            n_leaves = tree.n_leaves
            M = np.zeros((n_leaves, k + 1, k + 1))
            v = np.zeros((n_leaves, k + 1))
            cnt = np.zeros(n_leaves)
            Zl = Z[leaf_idx]
            Yl = Y[leaf_idx]
            for lid in np.unique(leaf_ids):
                mask = leaf_ids == lid
                Zm = Zl[mask]
                M[lid] = Zm.T @ Zm
                v[lid] = Zm.T @ Yl[mask]
                cnt[lid] = mask.sum()

            self.trees.append(tree)
            self._leaf_M.append(M)
            self._leaf_v.append(v)
            self._leaf_n.append(cnt)
        return self

    def predict(self, X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
        n = X.shape[0]
        k1 = self._leaf_v[0].shape[1]
        total_M = np.zeros((n, k1, k1))
        total_v = np.zeros((n, k1))
        used_trees = 0
        for tree, M, v, cnt in zip(self.trees, self._leaf_M, self._leaf_v, self._leaf_n):
            leaf_ids = tree.apply(X)
            valid = cnt[leaf_ids] > 0
            if not valid.any():
                continue
            Mn = M[leaf_ids] / np.maximum(cnt[leaf_ids], 1)[:, None, None]
            vn = v[leaf_ids] / np.maximum(cnt[leaf_ids], 1)[:, None]
            total_M[valid] += Mn[valid]
            total_v[valid] += vn[valid]
            used_trees += 1

        total_M /= used_trees
        total_v /= used_trees
        total_M += ridge * np.eye(k1)[None, :, :]
        theta = np.linalg.solve(total_M, total_v[..., None])[..., 0]  # (n, K+1) -> [intercepto, CATE_1..K]
        return theta[:, 1:]
