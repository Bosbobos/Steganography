import unittest
import numpy as np
import Watermark as wm

class WatermarkTests(unittest.TestCase):
    def test_arnold_identity(self):
        # Test with identity matrix
        matrix = np.eye(3)
        after_arnold = wm.arnold(matrix)
        result = wm.reverse_arnold(after_arnold)
        self.assertTrue(np.array_equal(result, matrix))

    def test_arnold_2x2(self):
        # Test with 2x2 matrix
        matrix = np.array([[1, 2], [3, 4]])
        after_arnold = wm.arnold(matrix)
        result = wm.reverse_arnold(after_arnold)
        self.assertTrue(np.array_equal(result, matrix))

    def test_arnold_3x3(self):
        # Test with 3x3 matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        after_arnold = wm.arnold(matrix)
        result = wm.reverse_arnold(after_arnold)
        self.assertTrue(np.array_equal(result, matrix))

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()