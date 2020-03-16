#@Peimou Sun

import numpy as np
from collections import Counter


class Probtable():
    def __init__(self, N, name, prob):
        '''
        Parameters:
        -----------
        N: int, total number of observations
        name: string, the name of the feature
        prob: dict, the probability of each value in the specific feature.

        '''
        self.N = N
        self.name = name
        self.prob = prob

    def __repr__(self):
        return f"Name :{self.name}, Number of Values :{self.N}, ProbTable: {self.prob}"


class CProbtable():
    def __init__(self, Cname, Fname):
        self.Cname = Cname
        self.feature_list = Fname
        self.feature_prob = {con:{} for con in Cname}


    def __update(self, probt:Probtable, nclass):
        if probt.name not in self.feature_list:
            raise ValueError(f"No such feature named as: {probt.name}")
        if nclass not in self.Cname:
            raise ValueError(f"No such class named as: {nclass}")
        self.feature_prob[nclass][probt.name] = probt
        return


    def update(self, probtlist, nclass):
        for con in probtlist:
            self.__update(con, nclass)


class Naive_Bayes():
    def __init__(self, laplace_smooth = True):
        self.ls = laplace_smooth

    @classmethod
    def gen_probtable(cls, feature, name):
        N = len(feature)
        f = Counter(feature)
        prob = {con[0]:con[1]/N for con in f.items()}
        pt = Probtable(N, name, prob)
        return pt


    @classmethod
    def calc_feature_prob(cls, feature, probt:Probtable, laplace_smooth:bool):
        try:
            return probt.prob[feature]
        except:
            return 1/(probt.N+1) if laplace_smooth else 0


    def fit(self, X:np.array, y:np.array, feature_name = []):
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if len(feature_name) == 0:
            feature_name = np.arange(0, X.shape[1])
        elif len(feature_name) != X.shape[1]:
            raise ValueError("The columns in feature_name unmatch the columns in X ")
        self.fname = feature_name

        cname = np.unique(y)
        self.cpt = CProbtable(cname, feature_name)
        Nf = len(feature_name)
        Ny = len(y)

        for con in cname:
            xdata = X[y == con]
            probtlist = [self.gen_probtable(xdata[:, i], feature_name[i])
                         for i in range(Nf)]
            self.cpt.update(probtlist, con)

        cy = Counter(y)
        self.py = {con[0]:con[1]/Ny for con in cy.items()}


    def predict(self, x):
        if len(x) != len(self.fname):
            raise ValueError("Unmatched sample")
        cpt = self.cpt.feature_prob
        pey = {}
        for con in self.py.keys():
            pr = np.array([self.calc_feature_prob(x[i], cpt[con][self.fname[i]], self.ls)
                  for i in range(len(self.fname))])
            pey[con] = np.prod(pr) * self.py[con]
        return max(pey, key = pey.get)


if __name__ == "__main__":
    import pandas as pd

    df_train = pd.read_csv(r"G:\COMS4771_DATA\propublicaTrain.csv", header = 0)
    df_train_x = df_train[df_train.columns[1:]]
    df_train_y = df_train[df_train.columns[0]]
    data = df_train_x.values
    label = df_train_y.values
    fname = df_train_x.columns
    nb = Naive_Bayes() #laplace smooth is true
    nb.fit(data, label, fname)
    res = [nb.predict(data[i]) for i in range(len(data))]
    #66% accuracy

    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(data, label)
    res_sk = clf.predict(data)
    # 66.45% accuracy

    #You can print nb.cpt.feature_prob to see the prob of each feature in each class
    #e.g
    '''
    {0: {'sex': Name :sex, Number of Values :2267, ProbTable: {1: 0.7834142037935597, 0: 0.21658579620644022},
  'age': Name :age, Number of Values :2267, ProbTable: {64: 0.004411116012351125, 28: 0.032201146890163214, 32: 0.029554477282752536, 43: 0.01940891045434495, 20: 0.015880017644464048, 
  44: 0.0176444640494045, 34: 0.029113365681517425, 24: 0.04808116453462726, 22: 0.03925893250992501, 40: 0.020291133656815175, 41: 0.0176444640494045, 37: 0.029554477282752536, 26: 0.0388178209086899, 
  58: 0.010145566828407587, 29: 0.037935597706219674, 35: 0.030877812086457873, 33: 0.0282311424790472, 31: 0.033083370092633436, 25: 0.0423467137185708, 47: 0.016321129245699163, 23: 0.043228936921041024,
   72: 0.000882223202470225, 27: 0.035288928098809, 55: 0.0105866784296427, 62: 0.004852227613586237, 48: 0.015880017644464048, 21: 0.03617115130127922, 53: 0.018526687251874726, 49: 0.014997794441993825, 
   70: 0.0013233348037053375, 46: 0.015438906043228937, 30: 0.03440670489633877, 36: 0.02337891486546096, 74: 0.000882223202470225, 57: 0.012792236435818262, 59: 0.008381120423467137, 
   38: 0.025584472871636524, 42: 0.0176444640494045, 39: 0.019850022055580063, 66: 0.0035288928098809, 71: 0.0013233348037053375, 51: 0.013233348037053375, 67: 0.0035288928098809, 52: 0.015880017644464048, 
   50: 0.0141155712395236, 45: 0.0176444640494045, 83: 0.0004411116012351125, 56: 0.014997794441993825, 54: 0.010145566828407587, 61: 0.007498897220996913, 65: 0.002646669607410675, 63: 0.0022055580061755625, 
   60: 0.0066166740185266875, 68: 0.0013233348037053375, 69: 0.00176444640494045, 79: 0.0004411116012351125, 73: 0.0004411116012351125, 77: 0.000882223202470225, 19: 0.0004411116012351125},
  'race': Name :race, Number of Values :2267, ProbTable: {0: 0.6140273489192766, 1: 0.38597265108072343},
  'juv_fel_count': Name :juv_fel_count, Number of Values :2267, ProbTable: {0: 0.9823555359505955, 2: 0.002646669607410675, 1: 0.013233348037053375, 4: 0.000882223202470225, 3: 0.000882223202470225},
  'juv_misd_count': Name :juv_misd_count, Number of Values :2267, ProbTable: {0: 0.9704455227172475, 1: 0.02337891486546096, 6: 0.0004411116012351125, 3: 0.0022055580061755625, 2: 0.002646669607410675, 4: 0.0004411116012351125, 5: 0.0004411116012351125},
  'juv_other_count': Name :juv_other_count, Number of Values :2267, ProbTable: {0: 0.9594177326863697, 2: 0.011027790030877812, 1: 0.026025584472871635, 3: 0.00176444640494045, 4: 0.000882223202470225, 7: 0.0004411116012351125, 5: 0.0004411116012351125},
  'priors_count': Name :priors_count, Number of Values :2267, ProbTable: {13: 0.004411116012351125, 1: 0.20732245258050286, 8: 0.014556682840758712, 0: 0.4406704896338774, 2: 0.11071901191001324, 3: 0.06440229378032643, 9: 0.008381120423467137, 
  7: 0.015880017644464048, 5: 0.032201146890163214, 4: 0.03925893250992501, 12: 0.00529333921482135, 10: 0.00529333921482135, 20: 0.002646669607410675, 21: 0.000882223202470225, 6: 0.022055580061755623, 14: 0.004411116012351125, 15: 0.004411116012351125, 
  11: 0.008381120423467137, 27: 0.000882223202470225, 17: 0.0013233348037053375, 18: 0.000882223202470225, 23: 0.0004411116012351125, 22: 0.0004411116012351125, 16: 0.00176444640494045, 24: 0.000882223202470225, 19: 0.00176444640494045, 25: 0.0004411116012351125},
  'c_charge_degree_F': Name :c_charge_degree_F, Number of Values :2267, ProbTable: {0: 0.3992059991177768, 1: 0.6007940008822232},
  'c_charge_degree_M': Name :c_charge_degree_M, Number of Values :2267, ProbTable: {1: 0.3992059991177768, 0: 0.6007940008822232}},
 1: {'sex': Name :sex, Number of Values :1900, ProbTable: {1: 0.85, 0: 0.15},
  'age': Name :age, Number of Values :1900, ProbTable: {20: 0.034210526315789476, 22: 0.057368421052631575, 23: 0.05421052631578947, 24: 0.05789473684210526, 28: 0.04473684210526316, 41: 0.014210526315789474, 48: 0.011578947368421053, 45: 0.01, 33: 0.030526315789473683, 
  54: 0.006842105263157895, 42: 0.01368421052631579, 47: 0.014210526315789474, 26: 0.057368421052631575, 27: 0.05631578947368421, 49: 0.009473684210526316, 29: 0.04473684210526316, 40: 0.008421052631578947, 21: 0.05210526315789474, 37: 0.024736842105263158, 25: 0.05105263157894737, 
  46: 0.010526315789473684, 51: 0.012105263157894737, 50: 0.010526315789473684, 31: 0.04, 32: 0.028421052631578948, 30: 0.045789473684210526, 36: 0.018947368421052633, 34: 0.026842105263157896, 52: 0.009473684210526316, 58: 0.005789473684210527, 55: 0.00631578947368421, 44: 0.013157894736842105, 
  39: 0.015263157894736841, 35: 0.021052631578947368, 57: 0.004210526315789474, 38: 0.017894736842105262, 43: 0.017368421052631578, 56: 0.005789473684210527, 53: 0.005263157894736842, 19: 0.008947368421052631, 59: 0.007368421052631579, 61: 0.003157894736842105, 60: 0.003157894736842105, 
  66: 0.002105263157894737, 18: 0.0005263157894736842, 69: 0.0005263157894736842, 67: 0.0015789473684210526, 65: 0.0010526315789473684, 63: 0.0005263157894736842, 64: 0.0005263157894736842, 62: 0.0015789473684210526, 78: 0.0005263157894736842},
  'race': Name :race, Number of Values :1900, ProbTable: {0: 0.7047368421052631, 1: 0.29526315789473684},
  'juv_fel_count': Name :juv_fel_count, Number of Values :1900, ProbTable: {0: 0.9463157894736842, 5: 0.0015789473684210526, 1: 0.035789473684210524, 2: 0.008947368421052631, 4: 0.002631578947368421, 8: 0.0010526315789473684, 3: 0.003157894736842105, 6: 0.0005263157894736842},
  'juv_misd_count': Name :juv_misd_count, Number of Values :1900, ProbTable: {1: 0.056842105263157895, 0: 0.9126315789473685, 2: 0.019473684210526317, 13: 0.0005263157894736842, 8: 0.0010526315789473684, 3: 0.005789473684210527, 6: 0.0010526315789473684, 5: 0.0010526315789473684, 4: 0.0015789473684210526},
  'juv_other_count': Name :juv_other_count, Number of Values :1900, ProbTable: {1: 0.08368421052631579, 0: 0.8847368421052632, 3: 0.007368421052631579, 2: 0.018421052631578946, 5: 0.0015789473684210526, 4: 0.0036842105263157894, 6: 0.0005263157894736842},
  'priors_count': Name :priors_count, Number of Values :1900, ProbTable: {2: 0.10789473684210527, 5: 0.05894736842105263, 3: 0.09157894736842105, 1: 0.15368421052631578, 10: 0.03105263157894737, 0: 0.21105263157894738, 4: 0.06842105263157895, 15: 0.010526315789473684, 6: 0.041578947368421056, 
  7: 0.038421052631578946, 30: 0.0005263157894736842, 12: 0.01263157894736842, 14: 0.01263157894736842, 16: 0.008421052631578947, 8: 0.03315789473684211, 13: 0.02, 23: 0.005789473684210527, 19: 0.006842105263157895, 21: 0.004210526315789474, 18: 0.004736842105263158, 17: 0.009473684210526316, 11: 0.017894736842105262, 
  9: 0.02894736842105263, 20: 0.005263157894736842, 22: 0.005263157894736842, 25: 0.002105263157894737, 31: 0.0005263157894736842, 38: 0.0005263157894736842, 33: 0.0005263157894736842, 26: 0.0010526315789473684, 24: 0.003157894736842105, 37: 0.0005263157894736842, 28: 0.0010526315789473684, 29: 0.0010526315789473684, 27: 0.0005263157894736842},
  'c_charge_degree_F': Name :c_charge_degree_F, Number of Values :1900, ProbTable: {1: 0.6984210526315789, 0: 0.30157894736842106},
  'c_charge_degree_M': Name :c_charge_degree_M, Number of Values :1900, ProbTable: {0: 0.6984210526315789, 1: 0.30157894736842106}}}
    '''










