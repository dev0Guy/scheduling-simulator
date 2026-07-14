# import pytest
# from core import fib, is_prime, sum_squares


# def test_fib_basic():
#     assert fib(0) == 0
#     assert fib(1) == 1
#     assert fib(2) == 1
#     assert fib(10) == 55


# def test_fib_negative_raises():
#     with pytest.raises(ValueError):
#         fib(-1)


# @pytest.mark.parametrize(
#     "n, expected",
#     [
#         (0, False),
#         (1, False),
#         (2, True),
#         (3, True),
#         (4, False),
#         (17, True),
#         (18, False),
#         (97, True),
#     ],
# )
# def test_is_prime(n, expected):
#     assert is_prime(n) is expected


# def test_sum_squares():
#     assert sum_squares(0) == 0
#     assert sum_squares(1) == 0
#     assert sum_squares(5) == 0 + 1 + 4 + 9 + 16
#     assert sum_squares(10) == sum(i * i for i in range(10))
