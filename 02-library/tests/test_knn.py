import pytest
import pandas as pd
from cs506.knn import KNN
from sklearn.model_selection._split import train_test_split

@pytest.mark.parametrize('datasetPath', [
    ("tests/test_files/dataset_knn_1.csv"),
])
def test_knn_small_dataset(datasetPath):
    df = pd.read_csv(datasetPath)
    df['Points'] = df['Points'].apply(lambda x: eval(x))

    knn = KNN(df['Points'], df['Labels'], 1)
    test_point = [0, 0]
    assert 1 == knn.predict(test_point)

    knn = KNN(df['Points'], df['Labels'], 3)
    assert -1 == knn.predict(test_point)

    knn = KNN(df['Points'], df['Labels'], 5)
    assert 1 == knn.predict(test_point)

@pytest.mark.parametrize('datasetPath', [
    ("tests/test_files/dataset_iris_knn.csv"),
])
def test_knn_large_dataset(datasetPath):
    df = pd.read_csv(datasetPath)
    x = df.iloc[:, :4].values
    y = df['Species'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    knn = KNN(x_train, y_train, 7)

    predictions = []
    for i in range(30):
        predictions.append(knn.predict(x_test[i]))
    
    assert 1.0 == (sum(y_test[:30] == predictions) / len(predictions))

def test_out_of_bounds():
    try:
        knn = KNN([0,0], [1], -1)
    except ValueError as e:
        assert str(e) == "k needs to be in range"