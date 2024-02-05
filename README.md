# Embedded Interpretable Regressor

Regression model for Machine Learning created with the goal of do intepretable regression without losing accuracy on its predictions. To do so, it uses a combination of an interpretable classifier, Dempster Shafer Classifier and any regression model that can be used to predict the target variable.

## Examples

`example.py` provides basic usages for this method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install every requirement in the `requirements.txt` file in a new environment.

`python -m venv env`
`source env/bin/activate`
`pip install -r requirements.txt`

## Usage

Define number of buckets that will be used to discretize the data.

```python
    N_BUCKETS = 3
    BUCKETING = "quantile"
```

Get data as Pandas DataFrame, set target feature and preprocess the data

```Python
    data = pd.read_csv("data/insurance.csv")
    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)

    # Data Split
    X, y = data.drop(target, axis=1).values, data[target].values
```

Import a Regression method. For example, Gradient Boosting.

```Python
    from sklearn.ensemble import GradientBoostingRegressor
    regressor = GradientBoostingRegressor
```

Define args for the regressor as a python dict.

```Python
    reg_args={"loss": "absolute_error", "n_estimators": 300}
```

Set test and training datasets and create a Embedded Interpreter Regressor, give args for the regressor method.

```Python
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33)

    eiReg = EmbeddedInterpreter(GradientBoostingRegressor,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_args=reg_args,
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True)
```

Fit the model with args for the regressor method.

```Python
    eiReg.fit(X_train, y_train,
              reg_args={},
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
```

Get prediction results with new data.

```Python
    y_pred = eiReg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    print(r2, MAE)
```

Finally you can get the rules used by the classifier to predict the data.
