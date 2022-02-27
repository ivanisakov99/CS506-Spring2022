import pytest
from cs506 import read, matrix
import numpy as np

@pytest.mark.parametrize('matrixPath', [
    ("tests/test_files/dataset_3_1.csv"),
])
def test_matrix_det_square_positive(matrixPath):
    """
    Test a matrix for with a positive determinant
    """
    mat = read.read_csv(matrixPath)
    
    assert round(matrix.get_determinant(mat), 9) == round(np.linalg.det(mat), 9)


@pytest.mark.parametrize('matrixPath', [
    ("tests/test_files/dataset_3_3.csv"),
])
def test_matrix_det_square_negative(matrixPath):
    """
    Test a matrix for with a negative determinant
    """
    mat = read.read_csv(matrixPath)

    assert round(matrix.get_determinant(mat),9) == round(np.linalg.det(mat), 9)

@pytest.mark.parametrize('matrixPath', [
    ("tests/test_files/dataset_3_2.csv"),
])
def test_matrix_det_not_square(matrixPath):
    """
    Test a matrix that's not square
    """
    try:
        mat = read.read_csv(matrixPath)
        matrix.get_determinant(mat)
    except ValueError as e:
        assert str(e) == 'Matrix must be square'
