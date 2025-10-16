import numpy as np
from StumpGenerator import StumpGenerator
from utils.helperfunctions import amount_of_say_calc, prob_array


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



    def predict(self,x):
        m,_ = x.shape
        predictions = np.zeros(m,dtype=object)
        for i in range(m):
            predict = self.predict_one(x[i,:])
            predictions[i] = predict


        return predictions
