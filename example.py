import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from EIRegressor.EIRegressor import EIRegressor

from sklearn.ensemble import GradientBoostingRegressor

def main():
    N_BUCKETS = 3
    BUCKETING = "quantile"
    
    # Load dataframe
    data = pd.read_csv("data/insurance.csv")
    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)

    # Data Split
    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    
    xgb_regressor = GradientBoostingRegressor

    EIgb = EIRegressor(xgb_regressor,
                        reg_args={"loss": "absolute_error", "n_estimators": 300},
                        n_buckets=N_BUCKETS, bucketing_method=BUCKETING, max_iter=4000, lossfn="MSE",
                        min_dloss=0.0001, lr=0.005, precompute_rules=True,
                        force_precompute=True, device="cuda")

    EIgb.fit(X_train, y_train,
                reg_args={},
                add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
                column_names=data.drop(target, axis=1).columns)
    y_pred = EIgb.predict(X_test)

    print("R2 for GradientBoostingEmbedded: ", r2_score(y_test, y_pred))
    print("MAE for GradientBoostingEmbedded: ",
            mean_absolute_error(y_test, y_pred))
    print("clf score: ", EIgb.evaluate_classifier(X_test, y_test))

if __name__ == "__main__":
    main()
