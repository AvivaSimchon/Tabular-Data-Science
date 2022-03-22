"""Class ShapModelExplainer

                        Description: Model Explainer based on shap

                        Shapely values are based on the cooperative game theory. There is a trade off with
                        machine learning model complexity vs interpretability. Simple models are easier to
                        understand but they are often not as accurate at predicting the target variable.
                        More complicated models have a higher accuracy, but they are notorious of being
                        'black boxes' which makes understanding the outcome difficult. Python SHAP library
                        is an easy to use visual library that facilitates our understanding about feature
                        importance and impact direction (positive/negative) to our target variable both
                        globally and for an individual observation.

                        This class is a generic class for making SHAP explanations and it can be used in
                        several different operation modes:
                        1. Use a ShapModelExplainer object as a module in the pipeline. In this case, the
                            set to explain is the test set. Namely, we employ a train set to train the model
                            and a test set to evaluate its performance. Afterwords, we use the test set to
                            compute the shap values in order to explain the scores of the test examples.
                        2. Use a ShapModelExplainer object as an independent module to explain the scores of
                            a given set of instances in a batch mode. In this case, we use the batch set to
                            compute the shap values in order to explain their scores.
                        3. Use a ShapModelExplainer object as an independent module to explain the scores of a
                            given instance in online mode. In this case we have to first collect enough online
                            instances to compute the shap values in order to explain the scores of future
                            online instances.

                        How They Work?
                        SHAP values interpret the impact of having a certain value for a given feature in
                        comparison to the prediction we'd make if that feature took some baseline value.

"""

import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import os
import pathlib
import time


import utils
from utils_str import *
import datetime



# ----------------------------------------------------------------------------------------------------------------------
# Model Explainer Settings
# ----------------------------------------------------------------------------------------------------------------------
model_explainer_settings = dict()
model_explainer_settings['enable'] = True
model_explainer_settings['explainer_type'] = 'tree_explainer' #'explainer'#, 'tree_explainer'
model_explainer_settings['do_create_summary_plot'] = True
model_explainer_settings['do_create_local_plot'] = True

enable_shap_explanatory = model_explainer_settings['enable']
explainer_type = model_explainer_settings['explainer_type']
do_create_summary_plot = model_explainer_settings['do_create_summary_plot']
do_create_local_plot = model_explainer_settings['do_create_local_plot']


# ======================================================================================================================
# ShapModelExplainer
# Given a trained classification model, explain its predictions.
# ======================================================================================================================
class ShapModelExplainer:
    # ------------------------------------------------------------------------------------------------------------------
    # Constructor
    #model : model that we want to explain
    #explain_set : the dataset that we want to explain (Shap explain the data lefi the model that we give)
    #list_of_local: list of index that we want to give the local explanation plot, index must be unique and that will be the file name
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self , model , explain_set , list_for_local_plot = None):
        #define model_explanatory_dir in def_privates
        dt = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        self.shap_dir = 'SHAP'

        self.force_plot_dir_name = 'ForcePlot' + dt
        self.summary_plot_dir_name = 'SummaryPlot' + dt
        self.waterfall_plot_dir_name = 'WaterfallPlot' + dt
        self.bar_plot_dir_name = 'BarLocalPlot' + dt

        pathlib.Path(os.path.join(self.shap_dir, self.summary_plot_dir_name)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.shap_dir, self.force_plot_dir_name)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.shap_dir, self.waterfall_plot_dir_name)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.shap_dir, self.bar_plot_dir_name)).mkdir(parents=True, exist_ok=True)
        # General

        if not enable_shap_explanatory:
            print('Explanatory disabled by settings')
            return

        if model is None:
            print('model was not sent to Shap explainer.Exit the explanatory process!')
            return
        self.m_model = model
        # Create the explainer (for global explaination) object for the calculation of the shap values
        if explainer_type == 'explainer':
            self.m_explainer = shap.Explainer(self.m_model)
        elif explainer_type == 'tree_explainer':
            self.m_explainer = shap.TreeExplainer(self.m_model)
        else:
            print('Unknown type of explainer. Exit the explanatory process!')
            return

        if explain_set is None:
            print('Data to explain was not sent to Shap explainer. Exit the explanatory process!')
            return
        if not explain_set.index.is_unique:
            print('Index of data to explain is not unique. Exit the explanatory process!')
            return
        self.explain_set = explain_set

        #list of index that we want to plot local explanation plot
        if list_for_local_plot is not None:
            self.list_for_local_plot = list_for_local_plot
        else:
            self.list_for_local_plot = self.explain_set.head(10).index

        # set shap value for type of model
        # --- Calculate shap values --- #
        # The shap_values For SKLEARN model only object above is a list with two arrays:
        # - The first array is the SHAP values for a negative outcome (NEG)
        # - The second array is the list of SHAP values for the positive outcome (POS).
        # for other model type(catboost, xgboost - we don't have outcome
        # We typically think about predictions in terms of the prediction of a positive outcome, so we'll
        # pull out SHAP values for positive outcomes (pulling out shap_values[1]). To get a complete
        # view, one should pull also SHAP values for negative outcomes (pulling out shap_values[0]).
        get_expected_values_sklearn = lambda: self.m_explainer.expected_value[self.outcome]
        get_expected_values = lambda: self.m_explainer.expected_value

        get_shap_value_sklearn = lambda: self.m_shap_values.values[:, :, self.outcome]
        get_shap_value = lambda: self.m_shap_values.values[:, :]

        get_local_shap_value_sklearn = lambda: self.m_shap_values[:, :, self.outcome]
        get_local_shap_value = lambda: self.m_shap_values[:, :]

        # For each type model , set dir for plot ,set treshold(superpredictor)
        # expected value and shap value
        if type(self.m_model) is XGBClassifier:
            print('xgboost')
            model_dir = explanatory_dir_model_xg
            self.get_expected_values = get_expected_values
            self.m_shap_value = get_shap_value
            self.m_local_shap_value = get_local_shap_value
            self.link = 'logit'
        elif type(self.m_model) is CatBoostClassifier:
            model_dir = explanatory_dir_model_cb
            self.get_expected_values = get_expected_values
            self.m_shap_value = get_shap_value
            self.m_local_shap_value = get_local_shap_value
            self.link = 'identity'
        else:
            model_dir = explanatory_dir_model_rf
            self.get_expected_values = get_expected_values_sklearn
            self.m_shap_value = get_shap_value_sklearn
            self.m_local_shap_value = get_local_shap_value_sklearn
            self.link = 'identity'

        #Explain positive class
        self.outcome = 1
        self.outcome_str = 'pos'
        if self.outcome != 1:
            self.outcome_str = 'neg'


    # ------------------------------------------------------------------------------------------------------------------
    # explain()
    # Supply SHAP explanations
    # ------------------------------------------------------------------------------------------------------------------
    def explain(self, list_for_local_plot = None):
        print('ShapModelExplainer::explain')

        start_time = time.time()

        self.m_shap_values = self.m_explainer(self.explain_set)

        # -- Produce summary plot (global explanation) --- #
        if do_create_summary_plot:
             self.create_summary_plot()

        # # --- Produce local explanation --- #
        if do_create_local_plot:
             self.create_force_plot()
             self.create_waterfall_plot()
             self.create_bar_plot()

        print('time', time.time() - start_time)


    # create_summary_plot
    # This method supplies a GLOBAL explanation of the model.
    # ------------------------------------------------------------------------------------------------------------------
    def create_summary_plot(self):
        """
        :param to_explain_set: A data frame of the values of the variables to compute the SHAP values. generally it
         will be the same data frame or matrix that was passed to the model for prediction (not training).
        :param outcome: POS/NEG global explanation
        :return:
        """
        print('ShapModelExplainer::create_summary_plot')
        fig_name = 'summary_plot'

        # Define the set of features to display to be the complete feature set.
        max_features_to_display = 20 #len(to_explain_set.columns)
        size = (30, 25)

        # --- Bar plot --- #
        plt.figure(1, figsize=size)
        shap.summary_plot(self.m_shap_value(),
                          self.explain_set,
                          plot_size=size,
                          plot_type='bar',
                          max_display=max_features_to_display,
                          show=False)
        file_name = fig_name + '_bar_' + self.outcome_str
        plt.savefig(os.path.join(os.path.join(self.shap_dir, self.summary_plot_dir_name), file_name + '.png'), bbox_inches='tight')

        # --- Scattered plot --- #
        plt.figure(2, figsize=size)
        shap.summary_plot(self.m_shap_value(),
                          self.explain_set,
                          plot_size=size,
                          max_display=max_features_to_display,
                          show=False)
        file_name = fig_name + '_scattered'
        plt.savefig(os.path.join(os.path.join(self.shap_dir, self.summary_plot_dir_name), file_name + '.png'), bbox_inches= 'tight')

        # Clear the memory
        plt.close('all')

        # Here we compute and return the results of the summary plot for future use.
        shap_mean = np.abs(self.m_shap_value()).mean(0)  # The columns mean of the absolute shap values
        features = self.explain_set.columns.tolist()
        feature_importance_str = 'feature_importance_' + self.outcome_str
        feature_importance = pd.DataFrame(list(zip(features, shap_mean)), columns=['feature', feature_importance_str])
        feature_importance.sort_values(by=[feature_importance_str], ascending=False, inplace=True)
        feature_importance.reset_index(inplace=True, drop=True)
        file_name_csv = feature_importance_str + '.csv'
        feature_importance.to_csv(full_path(full_path(self.shap_dir, self.summary_plot_dir_name), file_name_csv),index=False)
        return feature_importance

    # ------------------------------------------------------------------------------------------------------------------
    # create_force_plot
    # This method supplies a Local explanation of the model. Given a set instances to explain, the method
    # supplies an explanation to each instance.
    #Aviva 17/11/21 Not for use
    # ------------------------------------------------------------------------------------------------------------------
    def create_force_plot(self):
        """
        :param to_explain_set: the set of examples to explain locally. Recall, this set may be the test set, a batch
        of instances to predict or a single instance in an online mode.
        :param outcome: POS/NEG global explanation
        :return:
        """
        print('ShapModelExplainer::create_force_plot')

        fig_name = 'force_plot'


        size = (30, 12)
        plt.figure(3, figsize=size)


        # Create a force plot for each instance. Note, the expected_value array is a member of the explainer that
        # contains the base value that the feature contributions start from. The value expected_value[0] regards the
        # base value for the negative outcome (NEG). Similarly, the value expected_value[1] regards the base value
        # for the positive outcome (POS).
        for item, instance in self.explain_set.loc[self.list_for_local_plot].iterrows():
            i = self.explain_set.index.get_loc(item)
            shap.force_plot(self.get_expected_values(),
                            self.m_shap_value()[i],
                            instance,
                            feature_names=self.explain_set.columns.tolist(),
                            matplotlib=True,
                            text_rotation=30,
                            figsize=(25, 4),
                            contribution_threshold=0.06,
                            show=False)

            file_name = str(item) + '_' + fig_name + '_' + self.outcome_str
            plt.savefig(full_path(full_path(self.shap_dir, self.force_plot_dir_name), file_name + '.png'),bbox_inches='tight')
            plt.savefig(full_path(full_path(self.shap_dir, self.force_plot_dir_name), file_name + '.pdf'),bbox_inches='tight')

        # Clear the memory
        plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # create_waterfall_plot
    # This method supplies a Local explanation of the model. Given a set instances to explain, the method
    # supplies an explanation to each instance.
    # ------------------------------------------------------------------------------------------------------------------
    def create_waterfall_plot(self):
        """
        :param to_explain_set: the set of examples to explain locally. Recall, this set may be the test set, a batch
        of instances to predict or a single instance in an online mode.
        create the plot only for the observation in the list (good prediction)
        :return:
        """
        print('ShapModelExplainer::create_waterfall_plot')
        fig_name = 'waterfall_plot'

        for item, _ in self.explain_set.loc[self.list_for_local_plot].iterrows():#give the index(tz)
            i = self.explain_set.index.get_loc(item)  # give the location in df
            plt.figure()
            shap.plots.waterfall(self.m_local_shap_value()[i], max_display=20, show=False)

            file_name = str(item) + '_' + fig_name + '_' + self.outcome_str
            plt.savefig(full_path(full_path(self.shap_dir, self.waterfall_plot_dir_name), file_name + '.png'),
                        bbox_inches='tight')

        # Clear the memory
        plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # create_bar_plot
    # This method supplies a Local explanation of the model. Given a set instances to explain, the method
    # supplies an explanation to each instance.
    # ------------------------------------------------------------------------------------------------------------------
    def create_bar_plot(self):
        """
        :param to_explain_set: the set of examples to explain locally. Recall, this set may be the test set, a batch
        of instances to predict or a single instance in an online mode.
        create the plot only for the observation in the list (good prediction)
        :return:
        """
        print('ShapModelExplainer::create_bar_plot')
        fig_name = 'bar_plot'

        for item, _ in self.explain_set.loc[self.list_for_local_plot].iterrows():  # give the index(tz)
            i = self.explain_set.index.get_loc(item)  # give the location in df

            plt.figure()
            shap.plots.bar(self.m_local_shap_value()[i], max_display=20, show=False)

            file_name = str(item) + '_' + fig_name + '_' + self.outcome_str
            plt.savefig(full_path(full_path(self.shap_dir, self.bar_plot_dir_name),file_name + '.png'),bbox_inches='tight')

        # Clear the memory
        plt.close('all')