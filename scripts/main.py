import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel(100) # suppress deprecation messages
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alibi.explainers import CounterfactualProto, Counterfactual
import dice_ml

import pandas as pd

import argparse

from util import Explanation, Stats, Dataset

np.random.seed(42)
# tf.random.set_seed(42)
 

def nn_model(input_shape):

    x_in = Input(shape=(input_shape,))
    x = Dense(10, activation='relu')(x_in)
    x = Dense(10, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn

def train_model(x_train, y_train):

    nn = nn_model(x_train.shape[1])
    nn.summary()
    nn.fit(x_train, y_train, batch_size=8, epochs=5, verbose=1)

    return nn

def main(args):

    ds_name = args.dataset_name
    ds = Dataset(args.data_path, ds_name, args.cf_algo)

    if ds_name == "german":
        if args.cf_algo == "dice":
            x_train, y_train, x_test, y_test, d, feature_names = ds.load_german() 
        else:
            tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
            x_train, y_train, x_test, y_test = ds.load_german()
    else:

        if args.cf_algo == "dice":
            x_train, y_train, x_test, y_test, d, feature_names = ds.load_data() 
        else:
            tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
            x_train, y_train, x_test, y_test = ds.load_data()



    # Train/load and evaluate models
    models = {}

    if args.train:

        for i in range(args.nmods):
            tf.random.set_seed(i)
            model = train_model(x_train, y_train)
            models[i] = model

            model.save(f'{args.model_path}nn_{ds_name}_seed_{i}.h5', save_format='h5')
            tf.saved_model.save(model, f'{args.model_path}nn_{ds_name}_seed_{i}_saved_model.h5') # used for onnx conversion

    else:
        for i in range(args.nmods):
            models[i] = tf.keras.models.load_model(f'{args.model_path}nn_{ds_name}_seed_{i}.h5')

    for k,v in models.items():
        score = v.evaluate(x_test, y_test, verbose=0)
        print(f'Test accuracy of model {k}: ', score[1])


    # Instantiate stats object
    # Compute some stats on dataset
    # lower and upper bounds of features
    stats = Stats(x_train, models)

    # Generate counterfactuals
    X = x_test[:args.nexps].reshape((args.nexps,) + x_test[1].shape)
    shape = X.shape
    
    # For each model-input pair, generate a counterfactual. 
    
    for i in range(shape[0]):
        for k, model in models.items():
            #set key to store result for current explanation/model
            stats.set_key(i,k)

            # Process one CFX at a time
            X_i = X[i]
            X_i = X_i.reshape(1,shape[1])

            
            if args.cf_algo == "wachter":

                target_proba = 1.0
                tol = 0.01 # want counterfactuals with p(class)>0.99
                target_class = 'other' 
                max_iter = 1000
                lam_init = 1e-1
                max_lam_steps = 10  
                learning_rate_init = 0.1
                feature_range = (x_train.min(),x_train.max())

                cf = Counterfactual(model, shape=X_i.shape, target_proba=target_proba, tol=tol,
                                target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                feature_range=feature_range)

                # Compute explanation for original input and noisy counterpart
                explanation = cf.explain(X_i)

            elif args.cf_algo == "proto":
                
                # initialize explainer, fit and generate counterfactual
                cf = CounterfactualProto(model, X_i.shape, use_kdtree=True, theta=10., max_iterations=1000,
                                    feature_range=(x_train.min(axis=0), x_train.max(axis=0)),
                                    c_init=1., c_steps=10)

                cf.fit(x_train)

                # Compute explanation for original input and noisy counterpart
                explanation = cf.explain(X_i)


            else:
                
                # DICE ML wrapper for model
                m = dice_ml.Model(model=model, backend="TF2")

                # Using method=random for generating CFs
                cf = dice_ml.dice.Dice(d, m, method="gradient")
                        
                # generate counterfactual

                # First convert input back into dataframe     
                X_df = pd.DataFrame(X_i, columns=feature_names)
                diverse_cfx = cf.generate_counterfactuals(X_df, total_CFs=1, desired_class="opposite").cf_examples_list

                # Wrapping explanations into our Explanation class
                ## TODO: test lines below when multiple CFs are returned
                explanations = []

                for item in diverse_cfx:
                    e = Explanation(int(1 - item.new_outcome))
                    e.cf['class'] = item.new_outcome
                    e.cf['X'] = item.final_cfs_df.iloc[:, :-1].to_numpy()
                    explanations.append(e)
        
                # Taking the first one just for the sake of printing something
                explanation = explanations[0]

            # Evaluate explanation
            stats.evaluate_explanations(X_i, explanation.cf)

    # Print result summary to file
    ppresult = stats.get_stats_summary(args.cf_algo)

    with open(args.log_path, 'a') as f:
        f.write(ppresult)




if __name__ == "__main__":

    choices_algo = ["wachter", "proto", "dice"]

    parser = argparse.ArgumentParser(description='CFX generation script.')
    parser.add_argument('dataset_name', metavar='ds', default=None, help='Dataset name.')
    parser.add_argument('data_path', metavar='dp', default=None, help='Path to dataset.')
    parser.add_argument('model_path', metavar='mp', default=None, help='Path where model should be loaded/saved.')
    parser.add_argument('log_path', metavar='lp', default=None, help='Path where logs should be loaded/saved.')
    parser.add_argument('cf_algo', metavar='a', default=None, help='Algorithm used to generate counterfactuals.', choices=choices_algo)
    parser.add_argument('-train', action="store_true", help='Controls whether model is trained anew or loaded. Default: False.')
    parser.add_argument('-nmods', type=int, default=2, help='Number of models to be considered. Default: 2.')
    parser.add_argument('-nexps', type=int, default=1, help='Number of cfx to be generated. Default: 1.')

    args = parser.parse_args()

    main(args)