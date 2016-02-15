def add_logs(lst):
    """
    Adds a list of log-scale values in decimal scale and converts them back.
    :param lst:
    :return:
    """
    return reduce(np.logaddexp, lst)
