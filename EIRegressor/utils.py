import pandas as pd
import numpy as np

def replace_nan_median(matrix: np.ndarray, medians: np.ndarray = None) -> np.ndarray:
    """
    Replace nan values with the median of the columns in the dataset. If medians is None, it will be calculated and returned.
    If medians is not None, it will be used to replace nan values.
    :param matrix: matrix to replace nan values
    :param medians: medians to use to replace nan values
    :return: medians used to replace nan values
    """
    if medians is None:
        num_samples, num_features = matrix.shape
        medians = np.zeros(num_features)
        for i, col in enumerate(matrix.T):
            medians[i] = np.nanmedian(col)
            if np.isnan(medians[i]):
                print(f'error at feature {i}')
                medians[i] = 0.0
    for i, col in enumerate(matrix.T):
        matrix.T[i][np.isnan(col)] = medians[i]
    return medians


def bucketing(data, bins, type):
    """
    Bucketing the data into bins

    :param data: array to bucket
    :param bins: number of bins
    :param type: type of bucketing('ranged'/'quantile'/'max_score')
    :return: (array of buckets, bins)
    """
    if type == "ranged":
        return pd.cut(data, bins=bins,
                      labels=False, retbins=True, duplicates="raise")
    elif type == "quantile":
        return pd.qcut(data, q=bins,
                       labels=False, retbins=True, duplicates="raise")
    elif type == "max_score":
        sorted_array = np.sort(np.array(data))
        total = sorted_array.sum()
        jump = total/bins
        count = 0
        group_number = 0
        sorted_groups = {}
        bins = [min(data)]
        groups = np.zeros_like(data, dtype=np.int8)
        for i in range(len(sorted_array)):
            if count > jump*(group_number+1):
                group_number += 1
                bins += [sorted_array[i]]
            sorted_groups[sorted_array[i]] = group_number
            count += sorted_array[i]
        bins += [max(data)]
        for i in range(len(groups)):
            groups[i] = sorted_groups[data[i]]
        return (groups, np.array(bins))
    else:
        print("type must be 'ranged', 'quantile' or 'max_score'")
        return ([], [])
