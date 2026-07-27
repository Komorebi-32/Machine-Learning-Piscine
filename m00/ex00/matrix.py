

class Matrix:
    # only one __init__ function is allowed so using type()
    # to know how to initialize the instance attributes
    def __init__(self, input):
        if type(input) is list:
            self.data = input
            self.shape = (len(input), len(input[0]))
        elif type(input) is tuple:
            self.shape = input
            self.data = [[[0] * input[0]], [[0] * input[1]]]

    def __str__(self):
        if self.shape[0] == 1 or self.shape[1] == 1:
            txt = "Vector("
        else:
            txt = "Matrix(" 
        rows = []
        for row in self.data:
            row_str = ", ".join(map(str, row))
            row_str = "[" + row_str + "]"
            rows.append(row_str)
        txt += ", \n".join(rows)
        return txt

# add : only matrices of same dimensions.
# __add__
# __radd__
# # sub : only matrices of same dimensions.
# __sub__
# __rsub__
# # div : only scalars.
# __truediv__
# __rtruediv__
# # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
# # returns a Vector if we perform Matrix * Vector mutliplication.
# __mul__
# __rmul__
# __str__
# __repr__