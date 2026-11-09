import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving quadratic interpolator for IFR point data.

    Each node interval is represented by a quadratic Bezier curve.  The
    unconstrained control point is inferred from local, PCHIP-style slope
    estimates, then projected onto the range that makes the interval both
    monotone between its endpoints and convex (non-negative second
    derivative).  Projection to the midpoint gives a straight line when the
    neighbouring data are incompatible with positive curvature.

    Only C0 continuity is required by the assignment.  Relaxing derivative
    continuity at a node is what lets the interpolator retain exact node fit,
    interval monotonicity, and convexity for arbitrary finite point data.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        if self.extrap_method_ != ExtrapMethod.FLAT:
            raise ValueError("Monotone-convex interpolation requires FLAT extrapolation")

        self.axis1_ = np.asarray(self.axis1_, dtype=float)
        self.values_ = np.asarray(self.values_, dtype=float)
        if self.length_ == 0:
            raise ValueError("At least one interpolation node is required")
        if not np.all(np.isfinite(self.axis1_)) or not np.all(np.isfinite(self.values_)):
            raise ValueError("Interpolation nodes and values must be finite")
        if self.length_ > 1 and not np.all(np.diff(self.axis1_) > 0.0):
            raise ValueError("Interpolation nodes must be strictly increasing")

        # The one-node funding curve is constant, so it needs no interval data.
        self.interval_widths_ = np.diff(self.axis1_)
        self.control_points_ = np.empty(0, dtype=float)
        if self.length_ == 1:
            return

        secants = np.diff(self.values_) / self.interval_widths_
        node_slopes = self._shape_preserving_slopes(self.interval_widths_, secants)

        # A quadratic Bezier interval has one control ordinate.  Blend the
        # controls implied by the left and right slope estimates, then project
        # it into [min(y0, y1), (y0 + y1) / 2].  This interval is exactly the
        # set that gives monotonicity and a non-negative second derivative.
        y0 = self.values_[:-1]
        y1 = self.values_[1:]
        left_controls = y0 + 0.5 * self.interval_widths_ * node_slopes[:-1]
        right_controls = y1 - 0.5 * self.interval_widths_ * node_slopes[1:]
        proposed = 0.5 * (left_controls + right_controls)
        lower = np.minimum(y0, y1)
        midpoint = 0.5 * (y0 + y1)
        self.control_points_ = np.clip(proposed, lower, midpoint)

    @staticmethod
    def _shape_preserving_slopes(widths: np.ndarray, secants: np.ndarray) -> np.ndarray:
        """Return local PCHIP-style node slopes used to shape the quadratics."""
        interval_count = len(secants)
        slopes = np.zeros(interval_count + 1, dtype=float)
        if interval_count == 1:
            slopes[:] = secants[0]
            return slopes

        for i in range(1, interval_count):
            left, right = secants[i - 1], secants[i]
            if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
                slopes[i] = 0.0
            else:
                w1 = 2.0 * widths[i] + widths[i - 1]
                w2 = widths[i] + 2.0 * widths[i - 1]
                slopes[i] = (w1 + w2) / (w1 / left + w2 / right)

        slopes[0] = (
            (2.0 * widths[0] + widths[1]) * secants[0]
            - widths[0] * secants[1]
        ) / (widths[0] + widths[1])
        slopes[-1] = (
            (2.0 * widths[-1] + widths[-2]) * secants[-1]
            - widths[-1] * secants[-2]
        ) / (widths[-1] + widths[-2])

        # Standard PCHIP endpoint limiter: keep the endpoint derivative in the
        # first interval's direction and prevent an excessive overshoot.
        for endpoint, secant in ((0, secants[0]), (-1, secants[-1])):
            if secant == 0.0 or np.sign(slopes[endpoint]) != np.sign(secant):
                slopes[endpoint] = 0.0
            elif abs(slopes[endpoint]) > 3.0 * abs(secant):
                slopes[endpoint] = 3.0 * secant
        return slopes

    @staticmethod
    def _bezier_value(y0: float, control: float, y1: float, t: float) -> float:
        one_minus_t = 1.0 - t
        return (
            one_minus_t * one_minus_t * y0
            + 2.0 * one_minus_t * t * control
            + t * t * y1
        )

    @staticmethod
    def _bezier_primitive(y0: float, control: float, y1: float, t: float) -> float:
        """Integral from normalized coordinate 0 to t (before width scaling)."""
        return (
            y0 * t
            + (control - y0) * t * t
            + (y0 - 2.0 * control + y1) * t * t * t / 3.0
        )

    def interpolate(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("Interpolation coordinate must be finite")
        if self.length_ == 1 or x <= self.axis1_[0]:
            return float(self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.values_[-1])

        interval = int(np.searchsorted(self.axis1_, x, side="right") - 1)
        t = (x - self.axis1_[interval]) / self.interval_widths_[interval]
        return float(self._bezier_value(
            self.values_[interval],
            self.control_points_[interval],
            self.values_[interval + 1],
            t,
        ))

    def integrate(self, start_x: float, end_x: float) -> float:
        start_x, end_x = float(start_x), float(end_x)
        if not np.isfinite(start_x) or not np.isfinite(end_x):
            raise ValueError("Integration bounds must be finite")
        if start_x == end_x:
            return 0.0
        if start_x > end_x:
            return -self.integrate(end_x, start_x)
        if self.length_ == 1:
            return float(self.values_[0] * (end_x - start_x))

        total = 0.0
        left_node, right_node = self.axis1_[0], self.axis1_[-1]

        # Flat extrapolation on either side of the supplied node range.
        if start_x < left_node:
            left_end = min(end_x, left_node)
            total += self.values_[0] * (left_end - start_x)
        if end_x > right_node:
            right_start = max(start_x, right_node)
            total += self.values_[-1] * (end_x - right_start)

        inside_start = max(start_x, left_node)
        inside_end = min(end_x, right_node)
        if inside_start >= inside_end:
            return float(total)

        first_interval = min(
            int(np.searchsorted(self.axis1_, inside_start, side="right") - 1),
            self.length_ - 2,
        )
        last_interval = min(
            int(np.searchsorted(self.axis1_, inside_end, side="left")),
            self.length_ - 2,
        )

        for interval in range(first_interval, last_interval + 1):
            lo = max(inside_start, self.axis1_[interval])
            hi = min(inside_end, self.axis1_[interval + 1])
            if hi <= lo:
                continue
            width = self.interval_widths_[interval]
            t0 = (lo - self.axis1_[interval]) / width
            t1 = (hi - self.axis1_[interval]) / width
            primitive_0 = self._bezier_primitive(
                self.values_[interval], self.control_points_[interval],
                self.values_[interval + 1], t0,
            )
            primitive_1 = self._bezier_primitive(
                self.values_[interval], self.control_points_[interval],
                self.values_[interval + 1], t1,
            )
            total += width * (primitive_1 - primitive_0)
        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
