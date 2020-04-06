"""

End-to-End Machine Learning Project Template


1. Look at the big picture
2. Get the data
3. Discover and visualize the data to gain insights
4. Prepare the data for Machine Learning algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor and maintain your system

"""


# Data Feching
import os
import tarfile
from six.moves import urllib

# Data Visualization
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Data Preprocessing
import numpy as np
import hashlib              # to hash train and test datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit  # split data randomly
from sklearn.impute import SimpleImputer        # takes care of missing values
from sklearn.preprocessing import LabelEncoder  # give numerical values to cathegories
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer # do both steps in one

# Custom Class Transformers
from sklearn.base import BaseEstimator, TransformerMixin # fit_transform() for free

# Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion   # merge concurrent pipelines

# Training models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Measuring and Evaluating
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV    # cartesian product for hyperparameters


# 1 Frame the Problem
#
# Start by understanding the problem and identify exatcly which
# information you want to retrieve from it
#
# - Organize it in Pipelines
# - Define what sort of method works best (supervised, reinforcement, etc)
# - Select a Performance Measure


# 2 Get Data
#
# Either Fetch the Data Using this Script or save the data in
# some directory

def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# 3 Discover the data
#
# This step should be done seperately, probably using jupyter notebook
# to plot multiple graphs, histograms, load the tables and interpret
# the data to better understand the problem and dataset

def display_multiple_tables(data, column="ocean_proximity"):
    data.head()     # displays table head
    data.info()     # shows each of columns 

    data[column].value_counts()     # shows count per cathegory
    data.describe()                 # calculates mean, min, max, count, etc


def plot_examples(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

    # just an example
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=data["population"]/100, label="population",
            c="median_house_value", cmpy=plt.get_cmap("jet"), colorbar=True)
    plt.legend()


def look_for_correlations(data, column="median_house_value"):
    corr_matrix = data.corr()
    corr_matrix[column].sort_values(ascending=False)

    # scatter cartesian product
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize(12, 8))


# 4 Prepare Data
# 4.1 Create a Test Set
#
# This step gets overlooked easily, but is quite essencial, if the traning
# data is not big enough
#
# There are multiple ways to solve this problem
# either do it manually
# by running the following function so:
# train_set, test_set = split_train_test(data, 0.2)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# You could and should hash it so no test instances get loaded at training
# accidentaly blowing the purpose of dividing it in the first place
#
# This can be done so:
# data_with_id = data.reset_index()     # adds index column
# train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "index")

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# You can use a splitter from sklearn that does this directly
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# You could and should divide your data according the the cathegories weights
# This is called Stratified Shuffle Split

def strat_split(data, 
                general_cathegory="income_cat", 
                other_cathegory="median_income"):

    data[general_cathegory] = np.ceil(data[other_cathegory] / 1.5)
    data[general_cathegory].where(data[general_cathegory] < 5, 5.0, 
                                  inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data[general_cathegory]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    return strat_train_set, strat_test_set


# 4.2 Data Cleaning
#
# If there is some missing data, 3 things can be done
# - get rid of the corresponding instances
# - get rid of the whole attribute
# - set the missing values to either zero, the mean, median, etc

def data_cleaning_example(data, cathegory="total_bedrooms"):
    data.dropna(subset=[cathegory])     # option 1
    data.drop(cathegory, axis=1)        # option 2

    median = data[cathegory].median()
    data[cathegory].fillna(median)      # option 3


# 4.3 Handle Text and Categorical Atttributes
#
# You should convert text to numbers, so the machine learning
# algorithms can better process the data
# This can be done in multiple ways, to one-hot-encode data is 
# generaly a fairly simple approach

def load_encoder(data, column="ocean_proximity"):
    encoder = LabelBinarizer()
    data_cat = data[column]
    data_cat_1hot = encoder.fit_transform(data_cat)
    return data_cat_1hot


# Or you create your own transformation class

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrroms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                    bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# 4. 4 Feature Scaling
#
# This step is also called normalization


# 4.5 Pipelining
#
# Load all above mentioned steps in a pipeline that can be run
# concurrently if using different parts of the data

def pipelining(data, num_attribs, cat_attribs="ocean_proximity"):
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
        ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ])

    return full_pipeline


# 5 Train 
#
# Use multiple models before tweaking the hyperparameters
# to test if one stands out

def traning_example(prepared_data, data_labels, model=LinearRegression):
    lin_reg = LinearRegression()
    lin_reg.fit(prepared_data, data_labels)


# 6 Measure and Fine Tune
#
# 6.1 Measure your results with a given metric
# 
# Use Cross-Validation

def lin_mse(model, prepared_data, labels):
    data_predictions = model.predict(prepared_data)
    mse = mean_squared_error(labels, data_predictions)
    return np.sqrt(mse)


# 6.2 Fine Tuning
#
# Use Grid Search to generate a cartesian product of all the hyperparameters
# you want to play with

def grid_search(model, prepared_data, data_labels):
    param_grid = [
            {
                'n_estimator': [3, 10, 30], 
                'max_features': [2, 4, 6, 8]},
            {
                'bootstrap': [False], 
                'n_estimators': [3, 10], 
                'max_features':[2, 3, 4]},
            ]

    model = model()
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")

    grid.fit(prepared_data, data_labels)
    grid.best_estimator_.features_importantes_

 
if __name__ == "__main__":
    HOUSING_PATH = "~/Work/handson-ml2/datasets/housing"
    housing = load_housing_data(HOUSING_PATH)   # fetch data

    # display_multiple_tables(housing)            # discover
    # plot_examples(housing)

    housing_prepared = pipelining(housing, list(housing)).fit_transform(housing)

    


