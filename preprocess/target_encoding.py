from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TargetEncode(BaseEstimator, TransformerMixin):
    """
    categories: The column names of the features you want to target-encode
    k(int): min number of samples to take a category average into the account
    f(int): Smoothing effect to balance the category average versus the prior probability, or the mean value relative to all the training
            examples
    noise_level: The amount of noise you want to add to the target encoding in order to avoid overfitting.
    random_state: The reproducibility seed in order to replicate the same target encoding while noise_level > 0
    """

    def __init__(self, categories='auto', k=1, f=1, noise_level=0, random_state=17):
        if type(categories) == str and categories != 'auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state

    def add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def fit(self, X, y=None):
        if type(self.categories) == 'auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
            # print(categories)
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target'].agg(['mean', 'count']))
            smoothing = (1 / (1 + np.exp(-(avg['count'] - self.k) / self.f)))
            self.encodings[variable] = dict(self.prior *
                                            (1 - smoothing) + avg['mean'] * smoothing)
        return self

    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable],
                                 inplace=True)
            unknown_value = {value: self.prior for value in
                             X[variable].unique() if value
                             not in self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                Xt[variable].replace(unknown_value, inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable],
                                              self.noise_level)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
#
#     def __init__(self, colnames, targetName,
#                  n_fold=5, verbosity=False,
#                  discardOriginal_col=True):
#         self.colnames = colnames
#         self.targetName = targetName
#         self.n_fold = n_fold
#         self.verbosity = verbosity
#         self.discardOriginal_col = discardOriginal_col
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         assert (type(self.targetName) == str)
#         assert (type(self.colnames) == str)
#         assert (self.colnames in X.columns)
#         assert (self.targetName in X.columns)
#
#         mean_of_target = X[self.targetName].mean()
#         kf = KFold(n_splits=self.n_fold,
#                    shuffle=True)
#         col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
#         X[col_mean_name] = np.nan
#
#         for tr_ind, val_ind in kf.split(X):
#             X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
#             X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
#                 X_tr.groupby(self.colnames)[self.targetName].mean())
#             X[col_mean_name].fillna(mean_of_target,
#                                     inplace=True)  # Fill in the place that has become nan with the global mean
#
#         if self.verbosity:
#             encoded_feature = X[col_mean_name].values
#             print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name, self.targetName,
#                                                                                   np.corrcoef(X[self.targetName].values,
#                                                                                               encoded_feature)[0][1]))
#         if self.discardOriginal_col:
#             X = X.drop(self.colnames, axis=1)
#         return X