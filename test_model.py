"""
Test model.py
"""


def test_exponential_sample_weights():
    """ Test exponential_sample_weights function. """
    from sandbox import exponential_sample_weights
    from numpy.testing import assert_array_almost_equal

    weights_one = exponential_sample_weights(1)
    weights_two = exponential_sample_weights(2)
    weights_three = exponential_sample_weights(3)

    assert weights_one == [1.]
    assert_array_almost_equal(weights_two, [1., 0.367879], decimal=4)
    assert_array_almost_equal(weights_three, [1., 0.606530, 0.367879], decimal=4)
