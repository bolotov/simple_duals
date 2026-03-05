
class dual:
    """
    A simple implementation of dual numbers for automatic differentiation.
    
    A dual number is of the form a + bε,
    where ε is an infinitesimal that satisfies ε^2 = 0.
    The real part a represents the value of the function
    """
    __slots__ = ['real', 'dl']  # attributes access is faster using __slots__

    def __init__(self, real, dl=0):
        self.real = real
        self.dl = dl

    def __add__(self, a):
        if isinstance(a, (int, float, complex)):
            return dual(self.real + a, self.dl)
        return dual(self.real + a.real, self.dl + a.dl)

    __radd__ = __add__

    def __mul__(self, a):
        if isinstance(a, (int, float, complex)):
            # (a + bε)c = ac + bcε
            return dual(self.real * a, self.dl * a)
        # (a + bε)(c + dε) = ac + (ad + bc)ε
        return dual(self.real * a.real, self.real * a.dl + self.dl * a.real)

    __rmul__ = __mul__

    def exp(self):
        e_a = math.exp(self.real)
        return dual(e_a, self.dl * e_a)

    def sin(self):
        return dual(math.sin(self.real), self.dl * math.cos(self.real))

    def cos(self):
        return dual(math.cos(self.real), -self.dl * math.sin(self.real))
