import numpy as np

class Information_gain:
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

    def split_dataset(self, X, y, threshold):
        left_X, left_y = [], []
        right_X, right_y = [], []
        for x_i, y_i in zip(X, y):
            if x_i > threshold:
                left_X.append(x_i)
                left_y.append(y_i)
            else:
                right_X.append(x_i)
                right_y.append(y_i)
        return left_X, left_y, right_X, right_y

    def best_split(self, X, y, label):
        thresholds = find_medians(X)
        max_info_gain = 0
        best_threshold = -1
        for t in thresholds:
            _, left_y, _, right_y = self.split_dataset(X, y, t)
            info_gain = self.information_gain(y, left_y, right_y, label)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_threshold = t
        return best_threshold