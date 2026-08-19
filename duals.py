"""
A simple implementation of dual numbers for forward automatic
differentiation using the Python standard library.
  
A dual number is of the form a + bε,
where ε is an infinitesimal that satisfies ε^2 = 0.
The real part a represents the value of the function
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypeAlias

Scalar: TypeAlias = float | int


def _coerce(value: Dual | Scalar) -> Dual:
    if isinstance(value, Dual):
        return value
    return Dual(float(value), 0.0)


@dataclass(frozen=True, slots=True)
class Dual:
    """A dual number with a real part and an infinitesimal part."""

    real: float
    dual: float = 0.0

    def __post_init__(self) -> None:
        # frozen dataclasses need  object.__setattr__ to force a type cast
        object.__setattr__(self, "real", float(self.real))
        object.__setattr__(self, "dual", float(self.dual))

    @classmethod
    def variable(cls, value: float) -> Dual:
        """Return a dual value suitable for differentiation at value."""
        return cls(float(value), 1.0)

    def reciprocal(self) -> Dual:
        """Return the reciprocal of this dual number."""
        return Dual(1.0 / self.real, -self.dual / (self.real * self.real))

    def __add__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other: Dual | Scalar) -> Dual:
        return self.__add__(other)

    def __sub__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        return Dual(other.real - self.real, other.dual - self.dual)

    def __mul__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real,
        )

    def __rmul__(self, other: Dual | Scalar) -> Dual:
        return self.__mul__(other)

    def __truediv__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        denom = other.real * other.real
        return Dual(
            self.real / other.real,
            (self.dual * other.real - self.real * other.dual) / denom,
        )

    def __rtruediv__(self, other: Dual | Scalar) -> Dual:
        other = _coerce(other)
        denom = self.real * self.real
        return Dual(
            other.real / self.real,
            (other.dual * self.real - other.real * self.dual) / denom,
        )

    def __pow__(self, exponent: Dual | Scalar) -> Dual:
        exponent = _coerce(exponent)

        if exponent.dual == 0.0:
            p = exponent.real

            if p.is_integer():
                n = int(p)
                if n == 0:
                    return Dual(1.0, 0.0)
                return Dual(self.real ** n, n * (self.real ** (n - 1)) * self.dual)

            return Dual(self.real ** p, p * (self.real ** (p - 1.0)) * self.dual)

        return exp(exponent * log(self))

    def __rpow__(self, base: Dual | Scalar) -> Dual:
        base = _coerce(base)

        if base.dual == 0.0:
            value = base.real ** self.real
            return Dual(value, value * math.log(base.real) * self.dual)

        return exp(self * log(base))

    def __neg__(self) -> Dual:
        return Dual(-self.real, -self.dual)

    def __pos__(self) -> Dual:
        return self

    def __abs__(self) -> Dual:
        if self.real == 0.0:
            return Dual(0.0, 0.0)

        sign = math.copysign(1.0, self.real)
        return Dual(abs(self.real), sign * self.dual)

    def __float__(self) -> float:
        return self.real

    def __int__(self) -> int:
        return int(self.real)

    def __bool__(self) -> bool:
        return bool(self.real or self.dual)

    def __repr__(self) -> str:
        return f"Dual({self.real!r}, {self.dual!r})"


NumberLike: TypeAlias = Dual | Scalar

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)


def _lift(
        value: NumberLike,
        f: Callable[[float], float],
        df: Callable[[float], float],
) -> Dual | float:
    if isinstance(value, Dual):
        return Dual(f(value.real), df(value.real) * value.dual)
    return f(float(value))


def lift_unary(
        f: Callable[[float], float],
        df: Callable[[float], float],
) -> Callable[[NumberLike], Dual | float]:
    """Return a function that applies f and its derivative to scalar or dual input."""

    def lifted(value: NumberLike) -> Dual | float:
        return _lift(value, f, df)

    return lifted


def _unary(
        name: str,
        f: Callable[[float], float],
        df: Callable[[float], float],
) -> Callable[[NumberLike], Dual | float]:
    """Makes small unary functions"""
    # Helper to bind a math function to its derivative so we don't
    # have to write 20 identical methods.
    lifted = lift_unary(f, df)
    lifted.__name__ = name
    lifted.__doc__ = f"Return {name}(x) for a scalar or dual argument."
    return lifted


# Just maping the functions to their derivatives
sin = _unary("sin", math.sin, math.cos)
cos = _unary("cos", math.cos, lambda x: -math.sin(x))
tan = _unary("tan", math.tan, lambda x: 1.0 + math.tan(x) * math.tan(x))

exp = _unary("exp", math.exp, math.exp)
expm1 = _unary("expm1", math.expm1, math.exp)

log = _unary("log", math.log, lambda x: 1.0 / x)
log1p = _unary("log1p", math.log1p, lambda x: 1.0 / (1.0 + x))
log2 = _unary("log2", math.log2, lambda x: 1.0 / (x * _LN2))
log10 = _unary("log10", math.log10, lambda x: 1.0 / (x * _LN10))

sinh = _unary("sinh", math.sinh, math.cosh)
cosh = _unary("cosh", math.cosh, math.sinh)
tanh = _unary("tanh", math.tanh, lambda x: 1.0 - math.tanh(x) * math.tanh(x))

asin = _unary("asin", math.asin, lambda x: 1.0 / math.sqrt(1.0 - x * x))
acos = _unary("acos", math.acos, lambda x: -1.0 / math.sqrt(1.0 - x * x))
atan = _unary("atan", math.atan, lambda x: 1.0 / (1.0 + x * x))


def sqrt(value: NumberLike) -> Dual | float:
    """Return the square root of x for a scalar or dual argument."""
    if isinstance(value, Dual):
        root = math.sqrt(value.real)

        if root == 0.0:
            if value.dual == 0.0:
                return Dual(0.0, 0.0)
            return Dual(0.0, math.copysign(math.inf, value.dual))

        return Dual(root, value.dual / (2.0 * root))

    return math.sqrt(float(value))


def differentiate(function: Callable[[Dual], Dual | float], point: float) -> float:
    """Return the derivative of function at point."""
    result = function(Dual.variable(point))

    if isinstance(result, Dual):
        return result.dual

    return 0.0


def value_and_derivative(
        function: Callable[[Dual], Dual | float],
        point: float,
) -> tuple[float, float]:
    """Return both the value and derivative of function at point."""
    result = function(Dual.variable(point))

    if isinstance(result, Dual):
        return result.real, result.dual

    return float(result), 0.0


if __name__ == "__main__":
    f = lambda x: sin(x) * exp(-x) / (x + 2.0)
    print(value_and_derivative(f, 0.7))

    def polynomial(x: Dual) -> Dual:
        return x * x + 3.0 * x

    print(value_and_derivative(polynomial, 2.0))
