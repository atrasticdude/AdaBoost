
import numpy as np
class Node:
    def __init__(self, feature=None, value=None, left=None, right=None):
        self.feature = feature   # index of feature used to split
        self.value = value       # label if leaf
        self.left = left         # left child Node
        self.right = right 
       


class DecisionTree:

    def compute_entropy(self, node, label):
        if len(node) == 0:
            return 0 
        p1 = np.sum(np.array(node) == label) / len(node)
        if p1 == 0 or p1 == 1:
            return 0
        return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    def information_gain(self, root_node, left_node, right_node, label):
        root_entropy = self.compute_entropy(root_node, label)
        left_entropy = self.compute_entropy(left_node, label)
        right_entropy = self.compute_entropy(right_node, label)

        left_weight = len(left_node) / len(root_node)
        right_weight = len(right_node) / len(root_node)

        gain = root_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        return gain

    def split_dataset(self, X, y, feature):
        left_X, left_y = [], []
        right_X, right_y = [], []
        for x_i, y_i in zip(X, y):
            if x_i[feature] == 1:
                left_X.append(x_i)
                left_y.append(y_i)
            else:
                right_X.append(x_i)
                right_y.append(y_i)
        return left_X, left_y, right_X, right_y

    def best_split(self, X_matrix, y, label):
        num_features = X_matrix.shape[1]
        best_feature = -1
        max_info_gain = 0

        for feature in range(num_features):
            _, left_y, _, right_y = self.split_dataset(X_matrix, y, feature)
            info_gain = self.information_gain(y, left_y, right_y, label)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return best_feature
    def majority_class(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]
    
    def build_tree(self,X,y,label,depth):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        if depth == 0:
            return Node(value=self.majority_class(y))

        best_feature = self.best_split(X, y, label)
        if best_feature == -1:
            return Node(value=self.majority_class(y))

        left_X, left_y, right_X, right_y = self.split_dataset(X, y, best_feature)

        left_child = self.build_tree(left_X, left_y, label, depth - 1)
        right_child = self.build_tree(right_X, right_y, label, depth - 1)

        return Node(feature=best_feature, left=left_child, right=right_child)


