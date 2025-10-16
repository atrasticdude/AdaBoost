from InformationGain import Information_gain
import numpy as np


class StumpGenerator:
    def __init__(self):
        self.stump = {}
        self.gini = None
        self.feature = None

    def split(self, x, y):
        _, n_feature = x.shape
        stump_correctness = {}
        for f in range(n_feature):
            types = np.unique(x[:, f])
            if set(types).issubset({"Yes", "No"}):
                # left_indices = np.where(x[:,f] == "Yes")[0]
                # right_indices = np.where(x[:,f] == "No")[0]
                l_correct_indices = np.where((x[:, f] == "Yes") & (y == "Yes"))[0]
                r_correct_indices = np.where((x[:, f] == "No") & (y == "No"))[0]
                l_incorrect_indices = np.where((x[:, f] == "Yes") & (y == "No"))[0]
                r_incorrect_indices = np.where((x[:, f] == "No") & (y == "Yes"))[0]
                stump_correctness[f] = (len(l_correct_indices), len(l_incorrect_indices), len(r_correct_indices),
                                        len(r_incorrect_indices))
                self.stump[f] = {"feature": f,
                                 "type": "categorical",
                                 "catergory": "Yes",
                                 "correct": np.concatenate((l_correct_indices, r_correct_indices)),
                                 "incorrect": np.concatenate((l_incorrect_indices, r_incorrect_indices))}
            elif all(isinstance(v, (int, float)) for v in x[:, f]):
                info_obj = Information_gain()
                best_threshold = info_obj.best_split(x[:, f], y, "Yes")
                if best_threshold is None:
                    continue
                # left_indices = np.where(x[:,f] > best_threshold)[0]
                # right_indices = np.where(x[:,f] <= best_threshold)[0]
                l_correct_indices = np.where((x[:, f] > best_threshold) & (y == "Yes"))[0]
                r_correct_indices = np.where((x[:, f] <= best_threshold) & (y == "No"))[0]
                l_incorrect_indices = np.where((x[:, f] > best_threshold) & (y == "No"))[0]
                r_incorrect_indices = np.where((x[:, f] <= best_threshold) & (y == "Yes"))[0]
                stump_correctness[f] = (len(l_correct_indices), len(l_incorrect_indices), len(r_correct_indices),
                                        len(r_incorrect_indices))
                self.stump[f] = {"feature": f,
                                 "type": "numerical",
                                 "thereshold": best_threshold,
                                 "correct": np.concatenate((l_correct_indices, r_correct_indices)),
                                 "incorrect": np.concatenate((l_incorrect_indices, r_incorrect_indices))}
        return stump_correctness

    def gini_score(self, x, y):
        stumps = self.split(x, y)
        gini_scores = {}
        for feature, classified in stumps.items():
            total_left = classified[0] + classified[1]
            total_right = classified[2] + classified[3]
            total_root = total_right + total_left
            if total_root == 0:
                gini_scores[feature] = 0
                continue
            gini_left = 1 - (
                        (classified[0] / total_left) ** 2 + (classified[1] / total_left) ** 2) if total_left > 0 else 0
            gini_right = 1 - ((classified[2] / total_right) ** 2 + (
                        classified[3] / total_right) ** 2) if total_right > 0 else 0
            gini_scores[feature] = (total_left / total_root) * gini_left + (total_right / total_root) * gini_right
        return gini_scores

    def find_best_gini(self, x, y):
        gini_scores = self.gini_score(x, y)
        min_score = float('inf')
        feature = None
        for f, score in gini_scores.items():
            if score < min_score:
                min_score = score
                feature = f
        self.feature = feature
        self.gini = min_score
        return self.stump[feature]