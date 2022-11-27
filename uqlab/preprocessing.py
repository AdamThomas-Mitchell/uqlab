"""
Methods of importing data and preparing for ML models
"""
import math
import numpy as np
import pandas as pd
import torch
import pkg_resources
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataLoader:
    def __init__(self, scale_input=True, scale_output=True, as_tensor=True, random_state=None):
        self.scale_input = scale_input
        self.scale_output = scale_output
        self.as_tensor = as_tensor
        self.random_state = random_state
        self.scaler_dict = {}
        self.std_dict = {}
        self.data_dict = None

    @staticmethod
    def scale_features(X_train, X_cal, X_test):
        """
        Scale only the non-cyclic features between ±pi
        NOTE: Features are not in same order after this function is called
        X is of form [X_noncyclic, X_cyclic]

        :param X_train:
        :param X_cal:
        :param X_test:
        :return:
        """
        n_dim = X_train.shape[1]

        # list of indices for non-cyclic and cyclic features
        noncyclic_dim_idx = [d - 1 for d in range(1, n_dim + 1) if not (d > 3 and d % 3 == 0)]
        cyclic_dim_idx = [d - 1 for d in range(1, n_dim + 1) if (d > 3 and d % 3 == 0)]

        # decompose X matrices into cyclic and non-cyclic features
        X_train_cyclic = X_train[:, cyclic_dim_idx]
        X_train_noncyc = X_train[:, noncyclic_dim_idx]

        X_cal_cyclic = X_cal[:, cyclic_dim_idx]
        X_cal_noncyc = X_cal[:, noncyclic_dim_idx]

        X_test_cyclic = X_test[:, cyclic_dim_idx]
        X_test_noncyc = X_test[:, noncyclic_dim_idx]

        # scale non-cyclic features to lie between ±pi
        scaler = MinMaxScaler(feature_range=(-math.pi, math.pi))
        scaler.fit(X_train_noncyc)
        X_train_noncyc_sc = scaler.transform(X_train_noncyc)
        X_cal_noncyc_sc = scaler.transform(X_cal_noncyc)
        X_test_noncyc_sc = scaler.transform(X_test_noncyc)

        # join scaled non-cyclic features and unscaled cyclic features
        X_train_adj = np.hstack((X_train_noncyc_sc, X_train_cyclic))
        X_cal_adj = np.hstack((X_cal_noncyc_sc, X_cal_cyclic))
        X_test_adj = np.hstack((X_test_noncyc_sc, X_test_cyclic))

        return X_train_adj, X_cal_adj, X_test_adj

    @staticmethod
    def scale_targets(y_train, y_cal, y_test):
        """
        Standardise y values for easier comparison
        :param y_train:
        :param y_cal:
        :param y_test:
        :return:
        """
        # get info for rescaling first
        std = np.std(y_train.flatten())

        # now define scaler and transform arrays
        scaler = StandardScaler()
        scaler.fit(y_train)
        y_train_sc = scaler.transform(y_train)
        y_cal_sc = scaler.transform(y_cal)
        y_test_sc = scaler.transform(y_test)

        return y_train_sc, y_cal_sc, y_test_sc, scaler, std

    @staticmethod
    def prepare_tensors(X_train, y_train, X_cal, y_cal, X_test, y_test):
        """
        Convert np arrays to torch tensor for model processing
        Place tensors on GPU if available
        :param X_train:
        :param y_train:
        :param X_cal:
        :param y_cal:
        :param X_test:
        :param y_test:
        :return:
        """
        X_train_torch, X_cal_torch, X_test_torch = map(
            torch.tensor,
            (X_train, X_cal, X_test)
        )
        y_train_torch, y_cal_torch, y_test_torch = map(
            torch.tensor,
            (y_train.flatten(), y_cal.flatten(), y_test.flatten())
        )

        if torch.cuda.is_available():
            X_train_torch = X_train_torch.cuda()
            y_train_torch = y_train_torch.cuda()
            X_cal_torch = X_cal_torch.cuda()
            y_cal_torch = y_cal_torch.cuda()
            X_test_torch = X_test_torch.cuda()
            y_test_torch = y_test_torch.cuda()

        return X_train_torch, y_train_torch, X_cal_torch, y_cal_torch, X_test_torch, y_test_torch

    def unscale_targets(self, y_test, y_pred_mean, y_pred_std, atom_label):
        """
        inverse transformation on arrays to return true values, predicted mean, predicted std in original scale
        :param y_test:  true y values, np.array (n_samples,)
        :param y_pred_mean:  predicted mean, np.array (n_samples,)
        :param y_pred_std:  predicted std, np.array (n_samples,)r
        :param atom_label:
        :return:
        """
        scaler = self.scaler_dict[atom_label]
        std = self.std_dict[atom_label]

        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_mean_unscaled = scaler.inverse_transform(y_pred_mean.reshape(-1, 1))
        y_pred_std_unscaled = y_pred_std * std

        return y_test_unscaled.flatten(), y_pred_mean_unscaled.flatten(), y_pred_std_unscaled


class GlycineLoader(DataLoader):
    def __init__(self, train_prop, scale_input=True, scale_output=True, as_tensor=True, random_state=None):

        super(GlycineLoader, self).__init__(scale_input, scale_output, as_tensor, random_state)
        self.train_prop = train_prop
        self.data_dict = self.prepare_glycine_system_data()

    @staticmethod
    def get_glycine_data():
        # this is extremely dumb - could iterate through data files
        # import train datasets
        stream_C1_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_C1_TRAINING_SET.csv')
        stream_C4_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_C4_TRAINING_SET.csv')
        stream_C6_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_C6_TRAINING_SET.csv')
        stream_C9_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_C9_TRAINING_SET.csv')
        stream_C11_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_C11_TRAINING_SET.csv')
        stream_H3_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H3_TRAINING_SET.csv')
        stream_H5_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H5_TRAINING_SET.csv')
        stream_H12_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H12_TRAINING_SET.csv')
        stream_H13_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H13_TRAINING_SET.csv')
        stream_H14_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H14_TRAINING_SET.csv')
        stream_H15_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H15_TRAINING_SET.csv')
        stream_H16_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H16_TRAINING_SET.csv')
        stream_H17_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H17_TRAINING_SET.csv')
        stream_H18_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H18_TRAINING_SET.csv')
        stream_H19_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_H19_TRAINING_SET.csv')
        stream_N2_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_N2_TRAINING_SET.csv')
        stream_N8_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_N8_TRAINING_SET.csv')
        stream_O7_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_O7_TRAINING_SET.csv')
        stream_O10_train = pkg_resources.resource_stream(__name__, 'data/glycine/train/GLYCINE_O10_TRAINING_SET.csv')

        C1_train = pd.read_csv(stream_C1_train)
        C4_train = pd.read_csv(stream_C4_train)
        C6_train = pd.read_csv(stream_C6_train)
        C9_train = pd.read_csv(stream_C9_train)
        C11_train = pd.read_csv(stream_C11_train)
        H3_train = pd.read_csv(stream_H3_train)
        H5_train = pd.read_csv(stream_H5_train)
        H12_train = pd.read_csv(stream_H12_train)
        H13_train = pd.read_csv(stream_H13_train)
        H14_train = pd.read_csv(stream_H14_train)
        H15_train = pd.read_csv(stream_H15_train)
        H16_train = pd.read_csv(stream_H16_train)
        H17_train = pd.read_csv(stream_H17_train)
        H18_train = pd.read_csv(stream_H18_train)
        H19_train = pd.read_csv(stream_H19_train)
        N2_train = pd.read_csv(stream_N2_train)
        N8_train = pd.read_csv(stream_N8_train)
        O7_train = pd.read_csv(stream_O7_train)
        O10_train = pd.read_csv(stream_O10_train)

        # import test datasets
        stream_C1_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/C1_features_with_properties.csv')
        stream_C4_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/C4_features_with_properties.csv')
        stream_C6_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/C6_features_with_properties.csv')
        stream_C9_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/C9_features_with_properties.csv')
        stream_C11_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/C11_features_with_properties.csv')
        stream_H3_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H3_features_with_properties.csv')
        stream_H5_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H5_features_with_properties.csv')
        stream_H12_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H12_features_with_properties.csv')
        stream_H13_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H13_features_with_properties.csv')
        stream_H14_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H14_features_with_properties.csv')
        stream_H15_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H15_features_with_properties.csv')
        stream_H16_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H16_features_with_properties.csv')
        stream_H17_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H17_features_with_properties.csv')
        stream_H18_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H18_features_with_properties.csv')
        stream_H19_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/H19_features_with_properties.csv')
        stream_N2_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/N2_features_with_properties.csv')
        stream_N8_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/N8_features_with_properties.csv')
        stream_O7_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/O7_features_with_properties.csv')
        stream_O10_test = pkg_resources.resource_stream(__name__, 'data/glycine/test/O10_features_with_properties.csv')

        C1_test = pd.read_csv(stream_C1_test)
        C4_test = pd.read_csv(stream_C4_test)
        C6_test = pd.read_csv(stream_C6_test)
        C9_test = pd.read_csv(stream_C9_test)
        C11_test = pd.read_csv(stream_C11_test)
        H3_test = pd.read_csv(stream_H3_test)
        H5_test = pd.read_csv(stream_H5_test)
        H12_test = pd.read_csv(stream_H12_test)
        H13_test = pd.read_csv(stream_H13_test)
        H14_test = pd.read_csv(stream_H14_test)
        H15_test = pd.read_csv(stream_H15_test)
        H16_test = pd.read_csv(stream_H16_test)
        H17_test = pd.read_csv(stream_H17_test)
        H18_test = pd.read_csv(stream_H18_test)
        H19_test = pd.read_csv(stream_H19_test)
        N2_test = pd.read_csv(stream_N2_test)
        N8_test = pd.read_csv(stream_N8_test)
        O7_test = pd.read_csv(stream_O7_test)
        O10_test = pd.read_csv(stream_O10_test)

        # return as dict
        glycine_dict = {
            'C1': (C1_train, C1_test),
            'C4': (C4_train, C4_test),
            'C6': (C6_train, C6_test),
            'C9': (C9_train, C9_test),
            'C11': (C11_train, C11_test),
            'H3': (H3_train, H3_test),
            'H5': (H5_train, H5_test),
            'H12': (H12_train, H12_test),
            'H13': (H13_train, H13_test),
            'H14': (H14_train, H14_test),
            'H15': (H15_train, H15_test),
            'H16': (H16_train, H16_test),
            'H17': (H17_train, H17_test),
            'H18': (H18_train, H18_test),
            'H19': (H19_train, H19_test),
            'N2': (N2_train, N2_test),
            'N8': (N8_train, N8_test),
            'O7': (O7_train, O7_test),
            'O10': (O10_train, O10_test)
        }

        return glycine_dict

    def prepare_glycine_atom(self, atom_tuple, atom_label):
        """

        :param atom_label:
        :param atom_tuple: (train_and_cal_set, test_set)
        :return:
        """
        # separate tuple
        atom_train_and_cal, atom_test = atom_tuple

        # separate X, y values
        atom_train_and_cal.drop(columns=['Unnamed: 0'], inplace=True)
        X_train_cal = atom_train_and_cal.iloc[:, :51]
        y_train_cal = atom_train_and_cal['iqa']
        X_test = atom_test.iloc[:, :51]
        y_test = atom_test['iqa']

        # restructure data
        X_train_cal = X_train_cal.to_numpy()  # (num_samples, num_dim)
        y_train_cal = y_train_cal.to_numpy().reshape(-1, 1)  # (num_samples, 1)

        X_test = X_test.to_numpy()  # (num_samples, num_dim)
        y_test = y_test.to_numpy().reshape(-1, 1)  # (num_samples, 1)

        # split train and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_cal,
            y_train_cal,
            train_size=self.train_prop,
            random_state=self.random_state
        )

        # scale features and targets (optional)
        if self.scale_input:
            X_train, X_cal, X_test = self.scale_features(X_train, X_cal, X_test)
        if self.scale_output:
            y_train, y_cal, y_test, scaler, std = self.scale_targets(y_train, y_cal, y_test)
            self.scaler_dict[atom_label] = scaler  # save this to revert scaling later
            self.std_dict[atom_label] = std

        # prepare as tensors (optional)
        if self.as_tensor:
            X_train, y_train, X_cal, y_cal, X_test, y_test = self.prepare_tensors(
                X_train, y_train, X_cal, y_cal, X_test, y_test
            )

        # format as dictionary
        atom_data_dict = {
            'train': (X_train, y_train),
            'cal': (X_cal, y_cal),
            'test': (X_test, y_test)
        }

        return atom_data_dict

    def prepare_glycine_system_data(self):
        """
        Return dictionary of preprocessed data for each atom in water dimer system
        Returned dict is of form;

        dict = {'H1': {'train': (X_train, y_train),
                       'cal': (X_cal, y_cal),
                       'test': (X_test, y_test)},
                'H2': ...}

        :return:
        """
        glycine_dict = self.get_glycine_data()

        processed_glycine_dict = {
            key: self.prepare_glycine_atom(
                value,
                key
            )
            for key, value in glycine_dict.items()
        }

        return processed_glycine_dict


class WaterDimerLoader(DataLoader):
    def __init__(self, n_train, n_cal, n_test, scale_input=True, scale_output=True, as_tensor=True, random_state=None):
        super(WaterDimerLoader, self).__init__(scale_input, scale_output, as_tensor, random_state)
        self.n_train = n_train
        self.n_cal = n_cal
        self.n_test = n_test
        self.data_dict = self.prepare_water_dimer_system_data()

    @staticmethod
    def train_cal_test_split(X, y, train_size, cal_size, test_size, random_state=None):
        """
        Splits X and y data to train, calibration, and test sets
        :param X:
        :param y:
        :param train_size:
        :param cal_size:
        :param test_size:
        :param random_state:
        :return:
        """

        # separate test set
        X_train_and_cal, X_test, y_train_and_cal, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        # separate train and cal sets
        train_prop = train_size / (train_size + cal_size)
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_and_cal, y_train_and_cal,
            train_size=train_prop,
            random_state=random_state
        )

        return X_train, y_train, X_cal, y_cal, X_test, y_test

    @staticmethod
    def get_water_dimer_data():
        """
        Import water dimer data and return in dictionary form
        """
        stream_H2 = pkg_resources.resource_stream(__name__, 'data/waterDimer/H2_training_set_22k_random.csv')
        stream_H3 = pkg_resources.resource_stream(__name__, 'data/waterDimer/H3_training_set_22k_random.csv')
        stream_H5 = pkg_resources.resource_stream(__name__, 'data/waterDimer/H5_training_set_22k_random.csv')
        stream_H6 = pkg_resources.resource_stream(__name__, 'data/waterDimer/H6_training_set_22k_random.csv')
        stream_O1 = pkg_resources.resource_stream(__name__, 'data/waterDimer/O1_training_set_22k_random.csv')
        stream_O4 = pkg_resources.resource_stream(__name__, 'data/waterDimer/O4_training_set_22k_random.csv')

        H2_reduced_set = pd.read_csv(stream_H2)
        H3_reduced_set = pd.read_csv(stream_H3)
        H5_reduced_set = pd.read_csv(stream_H5)
        H6_reduced_set = pd.read_csv(stream_H6)
        O1_reduced_set = pd.read_csv(stream_O1)
        O4_reduced_set = pd.read_csv(stream_O4)

        waterDimer_dict = {
            'H2': H2_reduced_set,
            'H3': H3_reduced_set,
            'H5': H5_reduced_set,
            'H6': H6_reduced_set,
            'O1': O1_reduced_set,
            'O4': O4_reduced_set
        }

        return waterDimer_dict

    def prepare_water_dimer_atom(self, atom_dataset, atom_label):
        """
        return preprocessed train, calibration, and test data for given water dimer atom
        :param atom_dataset:
        :param atom_label:
        :return:
        """
        # select subset of sample points from full atom dataset
        n_total_pts = self.n_train + self.n_cal + self.n_test
        atom_subset = atom_dataset.sample(n=n_total_pts, random_state=self.random_state)

        # separate X and y values
        X = atom_subset.iloc[:, 0:12]
        y = atom_subset['iqa']

        # restructure data
        X = X.to_numpy()  # (num_samples, num_dim)
        y = y.to_numpy().reshape(-1, 1)  # (num_samples, 1)

        # split into train/calibration/test sets
        train_prop = self.n_train / n_total_pts
        cal_prop = self.n_cal / n_total_pts
        test_prop = 1.0 - (train_prop + cal_prop)
        X_train, y_train, X_cal, y_cal, X_test, y_test = self.train_cal_test_split(
            X, y,
            train_size=train_prop, cal_size=cal_prop, test_size=test_prop,
            random_state=self.random_state
        )

        # scale features and targets (optional)
        if self.scale_input:
            X_train, X_cal, X_test = self.scale_features(X_train, X_cal, X_test)
        if self.scale_output:
            y_train, y_cal, y_test, scaler, std = self.scale_targets(y_train, y_cal, y_test)
            self.scaler_dict[atom_label] = scaler  # save this to revert scaling later
            self.std_dict[atom_label] = std

        # prepare as tensors (optional)
        if self.as_tensor:
            X_train, y_train, X_cal, y_cal, X_test, y_test = self.prepare_tensors(
                X_train, y_train, X_cal, y_cal, X_test, y_test
            )

        atom_data_dict = {
            'train': (X_train, y_train),
            'cal': (X_cal, y_cal),
            'test': (X_test, y_test)
        }

        return atom_data_dict

    def prepare_water_dimer_system_data(self):
        """
        Return dictionary of preprocessed data for each atom in water dimer system
        Returned dict is of form;

        dict = {'H1': {'train': (X_train, y_train),
                       'cal': (X_cal, y_cal),
                       'test': (X_test, y_test)},
                'H2': ...}

        :return:
        """
        waterDimer_dict = self.get_water_dimer_data()

        processed_waterDimer_dict = {
            key: self.prepare_water_dimer_atom(
                value,
                key
            )
            for key, value in waterDimer_dict.items()
        }

        return processed_waterDimer_dict
