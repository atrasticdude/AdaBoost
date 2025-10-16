import numpy as np

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  



def find_medians(a):
    arr = np.unique(np.sort(a))
    medians = []
    for i in range(len(arr) - 1):
        mid = (arr[i] + arr[i + 1]) / 2
        medians.append(mid)
    return medians


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
            _,left_y,_,right_y = self.split_dataset(X,y,t)
            info_gain = self.information_gain(y,left_y,right_y,label)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_threshold = t
        return best_threshold
        

# class Node:
#     def __init__(self, feature = None, root = None,  left_node = None, right_node = None,):
#         self.feature = feature
#         self.root = root
#         self.left_node = left_node
#         self.right_node = right_node


class StumpGenerator:
    def __init__(self):
        self.stump = {}
        self.gini = None
        self.feature = None
        
    def split(self,x,y):
        _, n_feature = x.shape
        stump_correctness = {}
        for f in range(n_feature):
            types = np.unique(x[:,f])
            if set(types).issubset({"Yes","No"}):
                # left_indices = np.where(x[:,f] == "Yes")[0]
                # right_indices = np.where(x[:,f] == "No")[0]
                l_correct_indices = np.where((x[:,f] == "Yes") & (y == "Yes"))[0]
                r_correct_indices = np.where((x[:,f] == "No") & (y == "No"))[0]
                l_incorrect_indices = np.where((x[:,f] == "Yes") & (y == "No"))[0]
                r_incorrect_indices = np.where((x[:,f] == "No") & (y == "Yes"))[0]
                stump_correctness[f] = (len(l_correct_indices),len(l_incorrect_indices),len(r_correct_indices),len(r_incorrect_indices))
                self.stump[f] = {"feature" : f,
                                 "type" : "categorical",
                                 "catergory" : "Yes",
                                 "correct" :  np.concatenate((l_correct_indices, r_correct_indices)),
                                 "incorrect" :  np.concatenate((l_incorrect_indices, r_incorrect_indices))}
            elif all(isinstance(v, (int, float)) for v in x[:,f]):
                info_obj = Information_gain()
                best_threshold = info_obj.best_split(x[:,f], y, "Yes")
                if best_threshold is None:
                    continue
                # left_indices = np.where(x[:,f] > best_threshold)[0]
                # right_indices = np.where(x[:,f] <= best_threshold)[0]
                l_correct_indices = np.where((x[:,f] > best_threshold) & (y == "Yes"))[0]
                r_correct_indices = np.where((x[:,f] <= best_threshold) & (y == "No"))[0]
                l_incorrect_indices = np.where((x[:,f] > best_threshold) & (y == "No"))[0]
                r_incorrect_indices = np.where((x[:,f] <= best_threshold) & (y == "Yes"))[0]
                stump_correctness[f] = (len(l_correct_indices),len(l_incorrect_indices),len(r_correct_indices),len(r_incorrect_indices))
                self.stump[f] = {"feature" : f,
                                 "type" : "numerical",
                                 "thereshold" : best_threshold,
                                 "correct" :  np.concatenate((l_correct_indices, r_correct_indices)),
                                 "incorrect" :  np.concatenate((l_incorrect_indices, r_incorrect_indices))}   
        return stump_correctness
    
    def gini_score(self,x,y):
        stumps = self.split(x,y)
        gini_scores = {}
        for feature, classified in stumps.items(): 
            total_left = classified[0] + classified[1]
            total_right = classified[2] + classified[3]
            total_root = total_right + total_left
            if total_root == 0:
                gini_scores[feature] = 0
                continue
            gini_left = 1 - ((classified[0]/total_left)**2 + (classified[1]/total_left)**2) if total_left > 0 else 0
            gini_right = 1 - ((classified[2]/total_right)**2 + (classified[3]/total_right)**2) if total_right > 0 else 0
            gini_scores[feature] = (total_left / total_root)* gini_left + (total_right / total_root) * gini_right
        return gini_scores
    
    def find_best_gini(self,x,y):
        gini_scores = self.gini_score(x,y)
        min_score = float('inf')
        feature = None
        for f, score in gini_scores.items():
            if score < min_score:
                min_score = score
                feature = f
        self.feature = feature
        self.gini = min_score
        return self.stump[feature]


# def amount_of_say_calc(total_error):
#     epsilon = 1e-10
#     return 1/2 * np.log2((1 - (total_error  +epsilon)) / (total_error  +epsilon))


def amount_of_say_calc(total_error):
    epsilon = 1e-10
    return 0.5 * np.log2((1 - total_error + epsilon) / (total_error + epsilon))

# def amount_of_say_calc(total_error):
#     epsilon = 1e-10
#     total_error = min(max(total_error, epsilon), 1 - epsilon)  # ensure in (0,1)
#     return 0.5 * np.log2((1 - total_error) / total_error)

def prob_array(arr):
    for i in range(1,len(arr)):
       arr[i] = arr[i] + arr[i - 1]
    return np.array(arr)

        

    
    



class AdaBoostBackend(StumpGenerator):
    def __init__(self):
        super().__init__()
        self.weigths = None
        self.model = []


    def set_parameter(self,x,y):
        self.find_best_gini(x,y)
        capacity = x.shape[0]
        self.weigths = np.ones(capacity) / capacity

    def amount_of_say(self):
        stump = self.stump[self.feature]
        error = stump["incorrect"]
        total_error = np.sum(self.weigths[error])
        return amount_of_say_calc(total_error)
    
    def adjust_weights(self):
        a = self.amount_of_say()
        stump = self.stump[self.feature]
        correct_poses = stump['correct']
        incorrect_poses = stump['incorrect']
        for i in correct_poses:
            self.weigths[i] *= np.exp(-a)
        for j in incorrect_poses:
            self.weigths[j] *= np.exp(a)
        total = np.sum(self.weigths)
        self.weigths /= total
        return self.weigths

    def fit(self,x,y, n_estimator = 2):
        m,n = x.shape
        if self.weigths is None:
            self.weigths = np.ones(m) / m
        temp_x = x
        temp_y = y
        for count in range(n_estimator):
            self.set_parameter(temp_x,temp_y)
            self.model.append((self.feature,self.stump[self.feature], self.amount_of_say()))
            self.adjust_weights()
            arr = prob_array(self.weigths)
            new_x = np.zeros((m,n),dtype= object)
            new_y = np.zeros(m,dtype=object)
            if count > 0: 
                for pos in range(m):
                    num = np.random.uniform(0, 1)
                    idx = np.searchsorted(arr,num)   
                    new_x[pos,:] = x[idx,:]
                    new_y[pos] = y[idx]
                temp_x = new_x
                temp_y = new_y 
        return self.model
    
    def predict_one(self,x):
        pos = []
        neg = []
        for feature,value in enumerate(x):
            for stumps in self.model:
                if feature == stumps[0]:
                    if stumps[1]['type'] == "categorical":
                        if value == "Yes":
                            pos.append(stumps[2])
                        else:
                            neg.append(stumps[2])
                    elif stumps[1]['type'] == "numerical":
                        if value > stumps[1]['thereshold']:
                            pos.append(stumps[2])
                        else:
                            neg.append(stumps[2])
        if np.sum(pos) > np.sum(neg):
            return "positive"
        elif np.sum(pos) < np.sum(neg):
            return "negative"
        return "undefined"
    
    def predict_one(self, x):
        pos, neg = [], []
        for feature, stump, alpha in self.model:
            if stump["type"] == "categorical":
                category = stump.get("catergory", "Yes")
                if x[feature] == category:
                    pos.append(alpha)
                else:
                    neg.append(alpha)
            elif stump["type"] == "numerical":
                threshold = stump["thereshold"]
                if float(x[feature]) > threshold:
                    pos.append(alpha)
                else:
                    neg.append(alpha)

        if sum(pos) > sum(neg):
            return "positive"
        elif sum(pos) < sum(neg):
            return "negative"
        return "undefined"


    # def predict_one(self, x):
    #     total_vote = 0  # sum of weighted votes

    #     for feature_idx, stump_info, alpha in self.model:
    #         # Categorical feature
    #         if stump_info['type'] == "categorical":
    #             if x[feature_idx] == stump_info.get('catergory', 'Yes'):
    #                 vote = 1
    #             else:
    #                 vote = -1
    #         # Numerical feature
    #         elif stump_info['type'] == "numerical":
    #             if float(x[feature_idx]) > stump_info['thereshold']:
    #                 vote = 1
    #             else:
    #                 vote = -1
    #         # Add weighted vote
    #         total_vote += alpha * vote

    #     # Final prediction
    #     return "positive" if total_vote > 0 else "negative"


    def predict(self,x):
        m,_ = x.shape
        predictions = np.zeros(m,dtype=object)
        for i in range(m):
            predict = self.predict_one(x[i,:])
            predictions[i] = predict


        return predictions


    



           










