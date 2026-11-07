"""Data-space drawing on top of a Manim NumberPlane.

`scene_weekly.py` imports this as `ch` and never touches scene coordinates directly.
That is the whole point of the module: a Manim scene mixes two coordinate systems -- the
data's (weekdays, shares of a week) and the screen's (units of frame height) -- and every
bug that cost real time in this animation came from a number crossing between them
unnoticed. A
bar's height is a *share*; its height on screen is a consequence of the plane it is
drawn in. Here the conversion happens in one place.

The convention that matters
---------------------------
**A bar's `x` is its left edge, not its centre.** Bar `h` of a weekly profile spans
`[h/7, (h+1)/7]`, so the tick that labels it sits at `(h + 0.5)/7`. Centre-anchoring
would push the first bar half off the axis and put every tick on a boundary between two
bars rather than under one.

Reading a height back
---------------------
`Chart.dy(1.0)` is the scene height of one data unit, so `bar.height / chart.dy(1.0)`
recovers the value a bar stands for. `scene.py` relies on that to drive live numbers
from bar heights, which is why `stack` below measures its inputs the same way rather
than carrying values alongside them: there is then only one source of truth for what a
bar means, and it is the bar.
"""

import numpy as np
from manim import DL, VGroup, Dot, Line, Rectangle

# Manim refuses a zero-height Rectangle, and a bar of exactly zero is normal in this
# data (a weekday with no sales). Draw it as a hairline instead of crashing the render.
MIN_HEIGHT = 1e-4


class Chart:
    """A NumberPlane plus the data->scene conversions used to draw on it.

    The plane is held live rather than copied: `scene.py` moves and rescales it, and
    everything drawn through this object has to follow. `baseline` is the data value
    bars grow from, which is 0 everywhere in this animation but is not assumed to be.
    """

    def __init__(self, plane, baseline: float = 0.0):
        self.plane = plane
        self.baseline = baseline

    # --- coordinates ---------------------------------------------------------------

    def pt(self, x: float, y: float) -> np.ndarray:
        """Scene point of the data point (x, y)."""
        return self.plane.c2p(x, y)

    def vec(self, dx: float, dy: float) -> np.ndarray:
        """Scene displacement of a data displacement.

        Not `pt(dx, dy)`: a displacement has no origin, so the plane's own origin has to
        be subtracted. `scene.py` fans the stacked days out with this, and using `pt`
        there would offset every day by the plane's position as well.
        """
        return self.plane.c2p(dx, dy) - self.plane.c2p(0, 0)

    def dx(self, width: float) -> float:
        """Scene width of a data width."""
        return abs(self.vec(width, 0)[0])

    def dy(self, height: float) -> float:
        """Scene height of a data height. `dy(1.0)` is the unit every height is read in."""
        return abs(self.vec(0, height)[1])

    # --- bars ----------------------------------------------------------------------

    def _rect(self, x: float, low: float, high: float, width: float, **style) -> Rectangle:
        """A rectangle spanning [low, high] in y and [x, x + width] in x, all in data units."""
        rectangle = Rectangle(
            width=self.dx(width),
            height=max(self.dy(abs(high - low)), MIN_HEIGHT),
            **style,
        )
        # The lower-left corner is the anchor, so the bar sits on the baseline and
        # starts at its own x -- see the module docstring on left-edge anchoring.
        rectangle.move_to(self.pt(x, min(low, high)), aligned_edge=DL)
        return rectangle

    def bar(self, x: float, value: float, width: float, **style) -> Rectangle:
        """One bar from the baseline up to `value`."""
        return self._rect(x, self.baseline, value, width, **style)

    def bars(self, xs, values, width: float, **style) -> VGroup:
        """One bar per (x, value) pair, in the order given.

        The order is the contract: `scene.py` ties entry *i* of a vector to `bars[i]`,
        so these must stay index-aligned with the values they were built from.
        """
        return VGroup(*[self.bar(x, value, width, **style)
                        for x, value in zip(xs, values)])

    @staticmethod
    def _style_of(bar) -> dict:
        """The fill and stroke of an existing bar, so a restacked copy still looks like it."""
        return {
            "fill_color": bar.get_fill_color(),
            "fill_opacity": bar.get_fill_opacity(),
            "stroke_color": bar.get_stroke_color(),
            "stroke_width": bar.get_stroke_width(),
            "stroke_opacity": bar.get_stroke_opacity(),
        }

    def stack(self, bars, x: float, width: float) -> tuple[VGroup, float]:
        """Pile `bars` into a single column at `x`, and return it with its total.

        Each piece keeps its own height and its own colour; only its position changes.
        The heights are *measured from the bars themselves* rather than passed in
        alongside, because by the time this is called the bars have been shifted around
        the screen and a parallel list of values would be one more thing that can drift
        out of step with what is drawn.

        Returns (the pieces, the total in data units). The pieces come back in the same
        order as the input, which is what lets `scene.py` transform bar *i* into piece
        *i* individually.
        """
        unit = self.dy(1.0)
        pieces, total = VGroup(), 0.0
        for bar in bars:
            value = bar.height / unit if unit else 0.0
            piece = self._rect(x, self.baseline + total, self.baseline + total + value,
                               width, **self._style_of(bar))
            piece.set_z_index(bar.z_index)
            pieces.add(piece)
            total += value
        return pieces, total

    # --- lines and points ------------------------------------------------------------

    def polyline(self, xs, ys, **style) -> VGroup:
        """The series as individual segments, one per consecutive pair.

        Separate segments rather than one path on purpose: `scene.py` pulls a single
        day out of three weeks by selecting the segments that belong to it, which a
        single `VMobject` would not allow.
        """
        points = [self.pt(x, y) for x, y in zip(xs, ys)]
        return VGroup(*[Line(start, end, **style)
                        for start, end in zip(points[:-1], points[1:])])

    def markers(self, xs, ys, radius: float = 0.04, **style) -> VGroup:
        """A dot per point, index-aligned with `xs` and `ys`."""
        return VGroup(*[Dot(self.pt(x, y), radius=radius, **style)
                        for x, y in zip(xs, ys)])
