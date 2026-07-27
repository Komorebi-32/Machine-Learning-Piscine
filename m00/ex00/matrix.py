

class Matrix:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data[0]), len(data[1]))
    def __init__(self, shape):
        self.shape = shape
        self.data = [[[0] * shape[0]], [[0] * shape[1]]]