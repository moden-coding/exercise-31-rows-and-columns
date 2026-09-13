#!/usr/bin/env python3

import unittest

import numpy as np

from src.rows_and_columns import get_columns, get_rows


class TestGetRows(unittest.TestCase):

    def test_row_types(self):
        a = np.random.randint(0, 10, (4, 5))
        rows = get_rows(a)
        self.assertIsInstance(
            rows, list, msg="The function get_rows should return a list!")
        self.assertEqual(
            len(rows), 4, msg="Incorrect number of elements in list")
        for row in rows:
            self.assertIsInstance(
                row, np.ndarray, msg="The list elements should be arrays!")

    def test_row_content(self):
        n = 4
        m = 5
        a = np.random.randint(0, 10, (n, m))
        rows = get_rows(a)
        self.assertEqual(
            len(rows), n, msg="Incorrect number of elements in list")
        for ri, row in enumerate(rows):
            self.assertEqual(row.shape, (m,), msg="Incorrect shape!")
            for ci in range(m):
                self.assertEqual(
                    a[ri, ci], row[ci],
                    msg="Incorrect value at (%i,%i)!" % (ri, ci))

    def test_single_row_matrix(self):
        a = np.array([[1, 2, 3]])
        rows = get_rows(a)
        self.assertEqual(
            len(rows), 1,
            msg="get_rows on a (1,3) matrix should return a list with "
                "exactly one row.")
        np.testing.assert_array_equal(
            rows[0], np.array([1, 2, 3]),
            err_msg="get_rows on [[1,2,3]] should return [[1,2,3]] as its "
                    "single row.")


class TestGetColumns(unittest.TestCase):

    def test_columns_types(self):
        a = np.random.randint(0, 10, (4, 5))
        columns = get_columns(a)
        self.assertIsInstance(
            columns, list,
            msg="The function get_columnss should return a list!")
        self.assertEqual(
            len(columns), 5, msg="Incorrect number of elements in list")
        for column in columns:
            self.assertIsInstance(
                column, np.ndarray, msg="The list elements should be arrays!")

    def test_column_content(self):
        n = 4
        m = 5
        a = np.random.randint(0, 10, (n, m))
        columns = get_columns(a)
        self.assertEqual(
            len(columns), m, msg="Incorrect number of elements in list")
        for ci, column in enumerate(columns):
            self.assertEqual(column.shape, (n,), msg="Incorrect shape!")
            for ri in range(n):
                self.assertEqual(
                    a[ri, ci], column[ri],
                    msg="Incorrect value at (%i,%i)!" % (ri, ci))

    def test_single_column_matrix(self):
        a = np.array([[1], [2], [3]])
        columns = get_columns(a)
        self.assertEqual(
            len(columns), 1,
            msg="get_columns on a (3,1) matrix should return a list with "
                "exactly one column.")
        np.testing.assert_array_equal(
            columns[0], np.array([1, 2, 3]),
            err_msg="get_columns on [[1],[2],[3]] should return [[1,2,3]] "
                    "as its single column.")


if __name__ == '__main__':
    unittest.main()
