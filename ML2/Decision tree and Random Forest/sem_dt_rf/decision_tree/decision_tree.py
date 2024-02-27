from typing import Optional

import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion, GiniCriterion, EntropyCriterion, MSECriterion
from sem_dt_rf.decision_tree.tree_node import TreeNode


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_leaf_size: int = 5, min_improvement: Optional[float] = None):
        self.criterion = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size #FIXME: use min_samples_leaf
        self.min_improvement = min_improvement

    def _build_nodes(self, x: np.ndarray, y: np.ndarray, criterion: Criterion, indices: np.ndarray, node: TreeNode):
        """
        Builds tree recursively

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        """
        if self.max_depth is not None and self.max_depth <= node.depth:
            return 
        if self.min_leaf_size is not None and self.min_leaf_size >= len(indices):
            return 
        if np.unique(y[indices]).shape[0] <= 1:
            return 
        if self.min_improvement is not None and node.improvement < self.min_improvement:
            return 

        q, threshold, feature = node.get_best_split(x[indices], y[indices], criterion)
        mask, left_child, right_child = node.split(
            x[indices], y[indices], criterion,
            feature=feature,
            threshold=threshold,
            improvement=q
        )
        self._build_nodes(x, y, criterion, indices[~mask], left_child)
        self._build_nodes(x, y, criterion, indices[mask], right_child)

    def _get_nodes_predictions(self, x: np.ndarray, predictions: np.ndarray, indices: np.ndarray, node: TreeNode):
        if node.is_terminal:
            predictions[indices] = node.predict_val
            return
        mask = node.get_best_split_mask(x[indices])
        self._get_nodes_predictions(x, predictions, indices[~mask], node.child_left)
        self._get_nodes_predictions(x, predictions, indices[mask], node.child_right)


class ClassificationDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "gini", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["gini", "entropy"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_classes = 0
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_classes = np.unique(y).shape[0]
        self.n_features = x.shape[1]
        if self.criterion == "gini":
            criterion = GiniCriterion(self.n_classes)
        elif self.criterion == "entropy":
            criterion = EntropyCriterion(self.n_classes)
        self.root = TreeNode(
            depth=0,
            predict_val=criterion.get_predict_val(y),
            impurity=criterion.score(y)
        )
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
 
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred = np.zeros((x.shape[0], self.n_classes))
        self._get_nodes_predictions(x, pred, np.arange(x.shape[0]), self.root)
        return pred

    @property
    def feature_importance_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        """
        feature_importances = np.zeros(self.n_features)

        def _get_feature_importance(node: TreeNode):
            if node.is_terminal: return
            feature_importances[node.feature] += node.improvement
            _get_feature_importance(node.child_left)
            _get_feature_importance(node.child_right)

        _get_feature_importance(self.root)
        return feature_importances


class RegressionDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "mse", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["mse"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        if x.ndim == 1: x = x.reshape(-1, 1)
        self.n_features = x.shape[1]
        if self.criterion == "mse":
            criterion = MSECriterion()
        self.root = TreeNode(
            depth=0,
            predict_val=criterion.get_predict_val(y),
            impurity=criterion.score(y)
        )
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)
        return self
 
    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1: x = x.reshape(-1, 1)
        pred = np.zeros(x.shape[0])
        self._get_nodes_predictions(x, pred, np.arange(x.shape[0]), self.root)
        return pred

    @property
    def feature_importance_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        """
        feature_importances = np.zeros(self.n_features)

        def _get_feature_importance(node: TreeNode):
            if node.is_terminal: return
            feature_importances[node.feature] += node.improvement
            _get_feature_importance(node.child_left)
            _get_feature_importance(node.child_right)

        _get_feature_importance(self.root)
        return feature_importances
        
