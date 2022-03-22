import pandas as pd
import numpy as np

np.random.seed(0)


from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

from sklearn import preprocessing


import shap
import random

random.seed(10)

#we assume that the user(DS) verify that the dataset that is send to this system is distribute similar like the train set
#the user can choose to send the dataset that he want to receive the  represented sample of all the data
#the user should send the model and the data that he want to received the sample
class CreateSample:

    def __init__(self, dataset, model):
        random.seed(10)
        #print('INIT CreateSample')

        # Prepare the dataset
        self.dataset = dataset
        self.norm_df = preprocessing.normalize(self.dataset.iloc[:, :-1])

        #find best k for Kmean clustering with elbow score
        self.best_k = self.find_best_k()
        #model that fit the data
        self.model = model


    def find_best_k(self):
        # find best_k()
        elbow_scores = {}
        for k in range(2, 10):
            # print(k)
            kmeans_model = KMeans(n_clusters=k, n_jobs=-1)
            kmeans_model.fit(self.norm_df)
            # Kmean_Pred = kmeans_model.labels_
            elbow_scores.update({k: kmeans_model.inertia_})

        score_list = list(elbow_scores.items())
        diff_score = [(var[0], (var[1] / score_list[i - 1][1])) for i, var in enumerate(score_list)]
        gap = [(var[0], (var[1] / score_list[i - 1][1])) for i, var in enumerate(diff_score)]

        def getKey(item):
            return item[1]

        best_k = sorted(gap, key=getKey)[0][0]
        #print('best_k!!!!!!!!!!!!!!!!!!!!!!!!!!!!', best_k)
        return best_k


    def Kmeans_sample(self,p):
       #print('in KMEANS sample')
       df = self.dataset.iloc[:, :-1]
       #best_k = self.find_best_k()
       kmeans = KMeans(n_clusters=self.best_k, random_state=0).fit(self.norm_df)
       # print(Counter(kmeans.labels_))
       df['cluster_label'] = kmeans.labels_
       df_samples = []
       for label, sub_df in df.groupby('cluster_label'):
           n = len(sub_df)
           sub_df = sub_df.drop(columns=['cluster_label'], errors='ignore')
           # can add the agrala between int and float
           sample_size = int(round(n * p))
           df_samples.append(sub_df.sample(sample_size))

       sample = pd.concat(df_samples)
       sample = sample.drop(columns=['cluster_label'], errors='ignore')
       return sample

    def shift_sample(self,p):
        df = self.dataset.iloc[:, :-1]
        #clustering = MeanShift(bandwidth=4).fit(self.norm_df)
        clustering = MeanShift().fit(self.norm_df)
        ##print(clustering.labels_)
        df['cluster_label'] = clustering.labels_
        df_shift = []
        for label, sub_df in df.groupby('cluster_label'):
            n = len(sub_df)
            sample_size = int(round(n * p))
            df_shift.append(sub_df.sample(sample_size))

        shift_samples = pd.concat(df_shift)
        shift_samples = shift_samples.drop(columns=['cluster_label'], errors='ignore')
        return shift_samples


    def create_sample(self,method,pourcent):
        #print('create sample')

        if method == 'Kmeans':
            sample = self.Kmeans_sample(pourcent)

        if method == 'uniform':
            prc = int(round(self.dataset.shape[0] * pourcent))
            rnd_sample = self.dataset.iloc[:, :-1].sample(prc)
            sample = rnd_sample

        if method == 'shift':
            sample = self.shift_sample(pourcent)

        return sample



    def compute_shap_value(self,sample):
        #print('compute shap value')
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(sample)
        return shap_values

    def comparison(self,shap_values , shap_values_sample):
        #ממוצע של ערכים מוחלטיפ  - מערך
        shap_values_means = np.abs(shap_values[:, :].values).mean(0)
        #חשיבות היחסית של כל פיטשר
        shap_value_means_norm = shap_values_means/sum(shap_values_means)

        shap_values_sample_means = np.abs(shap_values_sample[:, :].values).mean(0)
        shap_value_sample_means_norm = shap_values_sample_means / sum(shap_values_sample_means)


        # maximum absolut difference
        #שגיאה מקסימלית
        MAD = np.max(np.abs(shap_value_means_norm - shap_value_sample_means_norm))
        #print(MAD)
        # the most important feature over all the data
        feature_maximali = np.max(shap_value_means_norm)
        #print('max feature',feature_maximali)
        #שגיאה מקסימלית יחסית לפיטשר החשוב ביותר
        relative_mad = MAD / feature_maximali
        #print('relative mad',relative_mad)
        return relative_mad












