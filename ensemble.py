# Copyright Varun Shenoy 2017-2018

from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model, Sequential
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.applications import VGG16
import numpy as np
import os, sys
from wound_utils import deepAccuracy, single_class_stats

class Ensemble():
    models = []
    def __init__(self, model_files):
        for m in model_files:
            vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            add_model = Sequential()
            add_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
            add_model.add(Dense(1024, activation='relu'))
            add_model.add(Dropout(0.5))
            add_model.add(Dense(1024, activation='relu'))
            add_model.add(Dropout(0.5))
            add_model.add(Dense(9, activation='sigmoid'))

            model = Model(inputs=vgg16.input, outputs=add_model(vgg16.output))
            current_dir = os.getcwd()
            MODEL_PATH = current_dir + '/weights/'
            model.load_weights(MODEL_PATH + m)
            self.models.append(model)


    def evaluate_model(self, model, x_test, y_test):
        columns_to_select = ['Wound', 'Infected', 'Granulation Tissue','Fibrinous Exudate','Open','Drainage','Steri Strips','Staples','Sutures']
        preds = model.predict(x_test)

        # round predictions to clean 1s and 0s
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        model_stats = [deepAccuracy(y_test, preds), accuracy_score(y_test, preds)]
        full_stats = []
        for (index, col) in enumerate(columns_to_select):
            acc, f1, prec, rec, per1, per0, case1, case0 = single_class_stats(index, y_test, preds)
            full_stats.append([acc, f1, prec, rec, per1, per0, case1, case0])
        return full_stats, model_stats

    def predict(self, x, thresh=0.5, expand=True, probas=False):
        scores = []
        final_scores = []
        for model in self.models:
            sc = []

            if expand is True:
                sc = model.predict(np.expand_dims(x, axis=0))
            else:
                sc = model.predict(x)

            if not probas:
                sc[sc>=thresh] = 1
                sc[sc<thresh] = 0
            scores.append(sc)

        scores = np.sum(scores, axis=0)
        final_scores = np.divide(scores, len(self.models))
        if not probas:
            final_scores[final_scores>=0.5] = 1
            final_scores[final_scores<0.5] = 0
        return final_scores[0]

    def multi_pred(self, xs, index=None, expand=True, probas=False):
        res = []
        if index is None:
            for x in xs:
                res.append(self.predict(x, expand, probas))
        else:
            for x in xs:
                res.append(self.predict(x, expand, probas)[index])
        return res

    def generate_scores(self, val_X, val_Y):
        print("Evaluating model...")
        print("----------------------------------")
        full_stats = []
        model_stats = []
        for model in self.models:
            fstats, mstats = self.evaluate_model(model, val_X, val_Y)
            full_stats.append(fstats)
            model_stats.append(mstats)

        full_stats = np.sum(full_stats, axis=0)
        full_stats = np.divide(full_stats, len(self.models))
        model_stats = np.sum(model_stats, axis=0)
        model_stats = np.divide(model_stats, len(self.models))

        print("Exact Match: " + str(model_stats[0]))
        print("Accuracy (array must match): " + str(model_stats[1]))
        print("\n")
        columns_to_select = ['Wound', 'Infected', 'Granulation Tissue','Fibrinous Exudate','Open','Drainage','Steri Strips','Staples','Sutures']

        for (ind, stat_arr) in enumerate(full_stats):
            print(columns_to_select[ind])
            print("    Percent of 0s: {}".format(stat_arr[5]))
            print("    Percent of 1s: {}".format(stat_arr[4]))
            print("    Cases with 0s: {}".format(stat_arr[7]))
            print("    Cases with 1s: {}".format(stat_arr[6]))
            print("    Accuracy: {}".format(stat_arr[0]))
            print("    Precision: {}".format(stat_arr[2]))
            print("    Recall: {}".format(stat_arr[3]))
            print("    F1-score: {}".format(stat_arr[1]))
            print("----------------------------------")
