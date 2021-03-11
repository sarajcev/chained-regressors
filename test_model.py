def test_exponential_sample_weights():
    """ Test xponential_sample_weights function. """
    from model import exponential_sample_weights
    assert exponential_sample_weights(1) == [1.]
