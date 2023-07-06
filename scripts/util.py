import numpy as np

from sklearn.preprocessing import normalize

from sklearn.neighbors import LocalOutlierFactor

import dice_ml

import pandas as pd

import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class Dataset:

    def __init__(self, path, ds_name, algo):
        self.path = path
        self.ds_name = ds_name
        self.algo = algo


    def load_german(self):

        df_train = pd.read_csv (f'{self.path}train.csv').iloc[: , 1:]
        df_test = pd.read_csv (f'{self.path}test.csv').iloc[: , 1:]

        # Extract feature names and target
        names = list(df_train.columns)
        feature_names = names[:-1]
        target = names[-1]

        print(feature_names)

        # Convert to numpy for later use
        x_train, y_train = df_train[feature_names].to_numpy(), df_train[target].to_numpy()
        y_train = to_categorical(y_train)

        x_test, y_test = df_test[feature_names].to_numpy(), df_test[target].to_numpy()
        y_test = to_categorical(y_test)
        
        if self.algo == "dice":
            d = dice_ml.Data(dataframe=df_train, continuous_features=feature_names, outcome_name=target)
            return x_train, y_train, x_test, y_test, d, feature_names
        else:
            return x_train, y_train, x_test, y_test 

    @staticmethod
    def min_max_scale(df, continuous, min_vals=None, max_vals=None):
        df_copy = copy.copy(df)
        for i, name in enumerate(continuous):
            if min_vals is None:
                min_val = np.min(df_copy[name])
            else:
                min_val = min_vals[i]
            if max_vals is None:
                max_val = np.max(df_copy[name])
            else:
                max_val = max_vals[i]
            df_copy[name] = (df_copy[name] - min_val) / (max_val - min_val)
        return df_copy

    def load_data(self):
        df = pd.read_csv(f'{self.path}{self.ds_name}.csv')
        df = df.dropna()

        if self.ds_name == "no2":
            df = df.replace(to_replace={'N': 0, 'P': 1})

        ordinal_features = {}
        discrete_features = {}
        continuous_features = list(df.columns)[:-1]

        print(continuous_features)

        target = "Outcome"

        # min max scale
        min_vals = np.min(df[continuous_features], axis=0)
        max_vals = np.max(df[continuous_features], axis=0)
        df_mm = self.min_max_scale(df, continuous_features, min_vals, max_vals)
        columns = list(df_mm.columns)
        # get X, y
        X, y = df_mm.drop(columns=['Outcome']), pd.DataFrame(df_mm['Outcome'])

        SPLIT = .2
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=SPLIT, shuffle=True,
                                                            random_state=0)
        
        if self.algo == "dice":
            d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name=target)

            x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
            y_train = to_categorical(y_train)

            x_test, y_test = x_test.to_numpy(), y_test.to_numpy()
            y_test = to_categorical(y_test)

            return x_train, y_train, x_test, y_test, d, continuous_features
        else:
            x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
            y_train = to_categorical(y_train)

            x_test, y_test = x_test.to_numpy(), y_test.to_numpy()

            y_test = to_categorical(y_test)

            return x_train, y_train, x_test, y_test 


class Explanation:

    def __init__(self, original_label):
        self.orig_class = original_label
        self.cf = {}
    


class Stats:
    def __init__(self, inputs, models):
        self.inputs = inputs
        self.models = models 
        self.ranges = {'min': None, 'max': None}
        self._compute_ranges()
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.lof.fit(self.inputs)
        self.results = {}

    def _compute_ranges(self):
        self.ranges['min'] = self.inputs.min(axis=0)
        self.ranges['max'] = self.inputs.max(axis=0)


    def _compute_normalised_l1(self, x, cfx):
        return np.sum(np.abs(x - cfx)) / (cfx.shape[0])

    def _check_validity(self, x, cfx, model):
        # reshape otherwise model complains
        # x = np.squeeze(x)
        # cfx = np.squeeze(cfx['X'])
        return not(np.argmax(model.predict(x, verbose=0), axis=1)[0] == np.argmax(model.predict(cfx, verbose=0), axis=1)[0])

    def _check_validity_all(self, x, cfx):
        # Check if cfx is valid for all models
        result = True
        for k,v in self.models.items():
            if not self._check_validity(x,cfx, v):
                result = False
                break
        return result

    def _compute_lof(self, cfx):
        return self.lof.predict(cfx)

    def _evaluate_explanation(self, x, cfx):
        # Extract explanation from dictionary
        # Evaluate differrent metrics
        valid = self._check_validity_all(x, cfx)
        dist = self._compute_normalised_l1(x, cfx)
        lof = self._compute_lof(cfx)[0]

        return valid, dist, lof

    def set_key(self, inp_num, model):
        self.current_input = inp_num
        self.current_model = model

    def evaluate_explanations(self, x, cfx):

        # Store results in a dictionary where keys are tuples (input_number, model_number)
        if cfx is None:
            self.results[(self.current_input, self.current_model)] = None
        else:
            cfx = cfx['X']
            
            # Evaluate explanation for factual input
            valid_org, dist_org, lof_org = self._evaluate_explanation(x, cfx)
                       
            if valid_org:
                self.results[(self.current_input, self.current_model)] = {'dist_from_input': dist_org, 'lof_org': lof_org}    
            else:
                self.results[(self.current_input, self.current_model)] = None

            print(f"Validity: {valid_org}. Distance: {dist_org}. LOF: {lof_org}")

    
    def get_stats_summary(self, algo):

        total_exps = len(self.results.values())

        dist_inp = [v['dist_from_input'] for v in self.results.values() if v is not None]
        avg_dist_inp = np.average(dist_inp)

        lofs = [v['lof_org'] for v in self.results.values() if v is not None]
        avg_lofs = np.average(lofs)

        return f"Algorithm: {algo}. Number of models: {len(self.models)}. Number of valid cfx: {len(dist_inp)}/{total_exps}. Avg l1 distance from input: {avg_dist_inp}. Average LOF score: {avg_lofs}.\n"

   
        

        

                





