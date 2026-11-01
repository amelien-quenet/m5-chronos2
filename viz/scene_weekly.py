"""The weekly-shape embedding, animated.

The daily counterpart of `scene.py`, which did the same thing on hourly data. The
construction is the report's, with one substitution throughout: a **week of 7 weekdays**
replaces a **day of 24 hours**. Normalise each week to 1, pile the weeks up weekday by
weekday, normalise again, and what is left is 7 numbers that say what share of a week
each day carries -- a point in a space where shapes can be compared without volume
getting in the way.

What changed besides the arithmetic
-----------------------------------
- **The vector is shown in full.** 24 entries had to be elided to fit the frame; 7 do
  not, so every coordinate is on screen and tied live to the bar it came from. The
  ellipsis was a concession, and it is gone.
- **Every weekday gets a tick.** With 24 hours only three could be labelled.
- **The countdown runs over weeks**, and the easing, the fan-out and the camera zoom are
  derived from the number of steps rather than hard-coded for 21, so changing N_WEEKS in
  the generator does not quietly break the framing.

Coordinates versus camera (modified for m5)
-------------------------------------------
Two ways to make something look bigger -- move the mobjects or move the camera -- and
mixing them is how a scene becomes unmaintainable. The discipline here, verified across
the whole file:

- **Everything drawn is in data coordinates.** Bars, lines, dots, ticks and the
  aggregate column are placed through `chart.py`, which converts data to scene units
  against a plane. No mobject is ever positioned in raw scene units.
- **The planes fix the scale.** `x_length` / `y_length` decide how many scene units a
  data unit is worth. Changing those changes the drawing, not the view.
- **The camera is only ever translated, or scaled relative to itself.** Every scale call
  in the file is multiplicative (`scale(zoom_step)`, `scale(scale)`) or derived from its
  own current width (`frame_width` captured before the countdown and restored after,
  `new_width = 2 * width` at the split). `ZOOM_OUT` is the one absolute camera value,
  applied once at the top of `construct`.

  One near-exception, named so it is not mistaken for a second lever: `margin_base` in
  the stacking loop is a headroom in scene units, not a fraction of the frame, so a
  wider frame makes it relatively smaller and the loop zooms out less often. That is
  self-correcting -- it is an "if the column does not fit, pull back" rule, and the
  frame is restored to `frame_width` immediately afterwards either way -- so it leaves
  no trace on the framing.
- **Text is pinned to the frame, not the scene.** `screen_lock` re-derives size and
  position from the frame every single frame, as a fraction of it, so labels hold a
  constant apparent size through any camera move.

The consequence worth knowing: `ZOOM_OUT` is a pure view change. It cannot shift the
composition, because nothing downstream of it is written in absolute frame terms.

Where the data comes from
-------------------------
`scripts/make_manim_data.py` in the research repo writes the three parquets this reads.
The families are the **planted archetypes of the clustering audit's positive control**,
not invented shapes: the film shows what a real discrete structure would look like at a
separation above the measured detection floor, while the blog says what the real data
turns out to be. The example series' stored vector is the aggregate of exactly the weeks
drawn here -- the generator asserts it -- so the numbers on screen belong to the bars
on screen.

    manim -pqh scene_weekly.py WeeklyEmbedding
"""

import math
from pathlib import Path

import polars as pl
from manim import *

import chart as ch

WEEKDAYS = 7
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Total camera zoom-out and total fan spread over the whole countdown. Fixed here and
# divided by the number of weeks, so the finished pile looks the same whether 12 weeks
# or 30 are shown.
ZOOM_TOTAL = 3.9
FAN_TOTAL = 2.2

# Seconds per countdown step. The easing only chooses where in this range a step lands;
# the floor is what keeps every step renderable. One frame at the default 15 fps is
# 0.067 s, so MIN_STEP sits just above it -- a step shorter than a frame is not a fast
# step, it is a dropped one.
MIN_STEP = 0.08
MAX_STEP = 0.80
EASE_POWER = 2.5

# modified for m5: how far the camera sits back from the default framing.
#
# 1.0 reproduces manim's default frame. Raise it to see more of the scene; every
# proportion in the film is preserved, because this is the only absolute camera value
# in the script and everything downstream is relative to it.
ZOOM_OUT = 1.5


class WeeklyEmbedding(MovingCameraScene):

    BAR_W = 1 / WEEKDAYS
    X_AGG = 1 + 2 * BAR_W            # aggregate column, after a gap past Sunday
    X_MAX = X_AGG + BAR_W + 0.02     # a little air past it

    # --- camera and screen-space helpers ------------------------------------------

    def move_camera_to_x(self, plane, x, run_time=2):
        target = plane.c2p(x, 0)
        self.play(
            self.camera.frame.animate.move_to([target[0], self.camera.frame.get_y(), 0]),
            run_time=run_time,
        )

    def screen_lock(self, mob, apparent_h, corner=UL, u=0.15, v=0.2, align=LEFT):
        """Hold `mob` at `apparent_h` scene-units of *screen* height, at a frame-relative spot.

        `apparent_h` is a design constant per text role, not read from the mobject, so it
        survives the content changing underneath (Transform, become, digit swaps).
        """
        def updater(m):
            frame = self.camera.frame
            m.set(height=apparent_h * frame.width / config.frame_width)
            anchor = (frame.get_corner(corner)
                      + RIGHT * frame.width * u + DOWN * frame.height * v)
            m.move_to(anchor, aligned_edge=align)

        mob.add_updater(updater)
        updater(mob)
        return mob

    @staticmethod
    def bar_style(opacity):
        return dict(stroke_width=1, fill_color=GOLD_E, fill_opacity=opacity,
                    stroke_opacity=min(opacity + 0.2, 1))

    @staticmethod
    def _ease(i, n):
        """The shape alone: 1 at both ends of a countdown, 0 at its middle."""
        if n <= 1:
            return 1.0
        t = (i - (n - 1) / 2) / ((n - 1) / 2 + 0.5)
        return abs(t) ** EASE_POWER

    @classmethod
    def _step_seconds(cls, i, n):
        """How long step `i` of `n` lasts: the easing mapped into [MIN_STEP, MAX_STEP].

        The floor is not cosmetic. A symmetric easing is exactly 0 at the midpoint, and
        manim refuses a run_time of 0 outright. The original dodged that by accident --
        its `(10.5 - i)/11` is off-centre for `range(20)`, whose midpoint is 9.5, so the
        middle step landed on 4e-05 s rather than 0. Renderable, but far below one
        frame, which is why a third of that countdown went past in a single frame each.

        Mapping into a range fixes both faults at once: nothing is zero, and nothing is
        shorter than a frame. EASE_POWER is gentler than the original's 4.5 for the same
        reason -- at 4.5 over nineteen steps, most of them collapse onto the floor and
        the easing stops being an easing.
        """
        return MIN_STEP + (MAX_STEP - MIN_STEP) * cls._ease(i, n)

    def shown(self, values, k=None):
        """Raw values -> plane units, where 1.0 means "the whole week".

        k=1.0  : the true normalised heights, which then sum to exactly 1.0 -- the top
                 of the aggregate column.
        k=None : 'display' scale, same shape blown up so the tallest bar reaches 0.9 of
                 the axis. Used before the division is shown.

        Sums `values` itself, so each week is normalised by its own total.
        """
        total = sum(values)
        if total == 0:
            return [0.0 for _ in values]
        if k is None:
            k = 0.9 * total / max(values)
        return [v * k / total for v in values]

    # --- the normalisation beat, run twice ----------------------------------------

    def normalise_into_one(self, chart, xs, display_bars, values, explain, title):
        """The "divide by the total" beat, start to finish.

        Runs twice -- once on one week's 7 days, once on the 7 weekday aggregates --
        because the operation is identical apart from the numbers. The on-screen bars
        are replaced by their true normalised heights, copies pile into the aggregate
        column at X_AGG, a solid bar fades in over the pile, and a dashed line carries
        its top across to the 1 on the axis.

        Returns the normalised values -- the 7 numbers in [0, 1] summing to 1 that this
        whole construction is about -- rather than recomputing them later, so they
        cannot drift from the bars actually drawn.
        """
        normalised = self.shown(values, 1.0)
        bars = chart.bars(xs, normalised, self.BAR_W, **self.bar_style(1))
        self.play(ReplacementTransform(display_bars, bars), run_time=2)

        stack, _ = chart.stack(bars, self.X_AGG, self.BAR_W)
        self.play(
            *[TransformFromCopy(bars[i], stack[i]) for i in range(len(xs))],
            lag_ratio=0.06, run_time=2,
        )
        aggregate = chart.bar(self.X_AGG, 1.0, self.BAR_W, **self.bar_style(1))
        self.play(FadeIn(aggregate), run_time=0.8)
        # Leaf level: TransformFromCopy added each stack[i] individually, so the `stack`
        # VGroup was never a scene entry and remove(stack) would be a no-op.
        self.remove(*stack)                      # seams vanish under the solid bar

        one = chart.pt(0, 1.0)                   # same scene y as aggregate.get_top()
        dash = DashedLine(aggregate.get_corner(UL), one, dash_length=0.05, stroke_width=2)
        self.play(Create(dash), run_time=1)

        tick = Line(one + LEFT * 0.2, one, stroke_width=3)
        tick_text = Text("1").next_to(tick, LEFT, buff=0.15)
        self.screen_lock(explain, apparent_h=0.95, corner=UL, u=0.20, v=0.42)
        self.screen_lock(title, apparent_h=0.6, corner=UL, u=0.25, v=0.25)
        self.play(FadeOut(dash), FadeIn(tick), Write(tick_text),
                  FadeIn(explain), FadeIn(title), run_time=1)
        return normalised, bars, aggregate, tick, tick_text

    @staticmethod
    def report_vector(label, vector):
        """Print a normalised vector and cross-check that it sums to 1.

        Rendering is the side effect here: these 7 numbers are the actual output of the
        construction, so they go to stdout where a `manim render` run can be grepped.
        """
        print(f"VECTOR {label}: sum={sum(vector):.12f} n={len(vector)}")
        for day, value in zip(DAY_NAMES, vector):
            print(f"VECTOR   {day}  {value:.9f}")
        return vector

    # --- the scene ------------------------------------------------------------------

    def construct(self):
        # modified for m5: the one camera change in the scene.
        #
        # Everything is drawn in DATA coordinates through chart.py and sized in SCENE
        # units by the planes; the camera is only ever translated, or scaled by factors
        # relative to its own current width. So widening the frame once, here, before
        # anything reads it, is the only lever that reaches the whole film -- and it
        # cannot desynchronise anything, because every later camera operation is
        # relative (see the audit note below).
        #
        # Why it was needed: the week plane is 11 * X_MAX = 15.93 scene units wide,
        # against a default frame of 14.22. The chart and its aggregate column did not
        # fit in frame at all, by 12%. ZOOM_OUT = 1.5 gives 21.33 units of frame, which
        # holds the plane with about a third of its width again as margin.
        #
        # set(width=...) scales the frame about its own centre, which is the centre of
        # the screen -- so this zooms out around the middle and moves the view nowhere.
        #
        # Text does not shrink with it: screen_lock sizes every label as
        # apparent_h * frame.width / config.frame_width, so a wider frame makes labels
        # proportionally larger in scene units and identical on screen.
        self.camera.frame.set(width=config.frame_width * ZOOM_OUT)

        data = Path(__file__).resolve().parent / "data"
        series = pl.read_parquet(data / "weekly_series.parquet").sort("date")
        embeddings = pl.read_parquet(data / "archetype_embeddings.parquet")
        layout = pl.read_parquet(data / "archetype_umap.parquet")

        values = series["value"].to_list()
        dates = series["date"].to_list()
        assert len(values) % WEEKDAYS == 0, "the series is not a whole number of weeks"
        n_weeks = len(values) // WEEKDAYS
        series_id = series["series_id"][0]

        # The zoomed axis has to hold the tallest bar in the WHOLE dataset, not just this
        # series': the tours at the end visit every archetype. Derived from the data so
        # it stays correct if the parquets are regenerated.
        y_tick = math.ceil(max(float(embeddings[f"v{d}"].max())
                               for d in range(WEEKDAYS)) / 0.05) * 0.05

        # x in weeks, so one unit of the long plane is one week.
        xs = [i / WEEKDAYS for i in range(len(values))]
        last_week = n_weeks - 1
        week_indices = [i for i, x in enumerate(xs) if last_week <= x < last_week + 1]
        week_x = [xs[i] - last_week for i in week_indices]
        week_y = [values[i] for i in week_indices]

        # --- the original series --------------------------------------------------
        plane = NumberPlane(
            x_range=(0, n_weeks, 1),
            y_range=(0, max(values) if max(values) else 1, max(1, max(values) // 5)),
            x_length=n_weeks * 3,
            y_length=6,
            axis_config={"include_numbers": False},
        )
        full = ch.Chart(plane, baseline=0.0)
        # `label_constructor=Text` rather than manim's default: add_labels renders a str
        # through Tex, so these dates would be one LaTeX subprocess *each* -- twenty
        # compilations for twenty labels that contain no mathematics. Text goes through
        # Pango and needs no LaTeX at all.
        plane.x_axis.add_labels(
            {week: dates[week * WEEKDAYS].strftime("%d %b") for week in range(n_weeks)},
            font_size=24,
            label_constructor=Text,
        )

        segments = full.polyline(xs, values, color=GOLD_E, stroke_width=4)
        dots = full.markers(xs, values, color=PURPLE)
        title_main = Tex(f"{n_weeks} weeks of one series")
        self.screen_lock(title_main, apparent_h=0.6, u=0.25, v=0.1)
        self.add(title_main, plane, segments, dots)

        self.camera.frame.move_to([plane.c2p(1, 0)[0], self.camera.frame.get_y(), 0])
        self.move_camera_to_x(plane, last_week, run_time=8)

        title_next = Tex("Extracting a weekly shape")
        self.screen_lock(title_next, apparent_h=0.5, u=0.25)
        self.play(FadeOut(title_main))

        # --- isolate the last week ------------------------------------------------
        week_segments, other_segments = VGroup(), VGroup()
        for i in range(len(xs) - 1):
            target = (week_segments if last_week <= xs[i] and xs[i + 1] <= last_week + 1
                      else other_segments)
            target.add(segments[i])

        week_dots, other_dots = VGroup(), VGroup()
        for i, dot in enumerate(dots):
            (week_dots if last_week <= xs[i] <= last_week + 1 else other_dots).add(dot)

        self.play(
            FadeOut(plane.background_lines), FadeOut(plane.x_axis), FadeOut(plane.y_axis),
            FadeOut(other_segments), FadeOut(other_dots), run_time=1.5,
        )

        week_plane = NumberPlane(
            x_range=(0, self.X_MAX, 0.25),
            y_range=(0, 1.1, 0.1),                 # NORMALISED: 1.0 == the weekly total
            x_length=11 * self.X_MAX,
            y_length=6,
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            background_line_style={"stroke_opacity": 0},
        )
        week = ch.Chart(week_plane, baseline=0.0)

        # Every weekday gets a tick, centred under its own bar -- see chart.py on why a
        # bar's x is its left edge.
        day_ticks, day_labels = VGroup(), VGroup()
        for d, name in enumerate(DAY_NAMES):
            foot = week.pt((d + 0.5) / WEEKDAYS, 0.0)
            day_ticks.add(Line(foot + DOWN * 0.2, foot, stroke_width=3))
            day_labels.add(Text(name, font_size=28).next_to(day_ticks[-1], DOWN, buff=0.12))

        new_segments = week.polyline(week_x, self.shown(week_y), color=GOLD_E, stroke_width=4)
        new_dots = week.markers(week_x, self.shown(week_y), color=PURPLE)

        shift = plane.c2p(last_week + 0.5, 0) - week_plane.c2p(0.65, 0)
        for mob in (week_plane, new_segments, new_dots, day_ticks, day_labels):
            mob.shift(shift)

        self.play(
            FadeIn(week_plane),
            Transform(week_segments, new_segments), Transform(week_dots, new_dots),
            FadeIn(day_ticks), FadeIn(day_labels),
            self.camera.frame.animate.move_to(week_plane.get_center()),
            run_time=2,
        )
        self.play(FadeIn(title_next))

        # --- line -> bars, still at display scale ---------------------------------
        bar_chart = week.bars(week_x, self.shown(week_y), self.BAR_W, **self.bar_style(1))
        self.play(FadeOut(week_dots), ReplacementTransform(week_segments, bar_chart), run_time=2)
        self.play(FadeOut(title_next))

        explain = MathTex(
            r"\forall d \in \mathrm{days},\quad"
            r"\mathrm{bar}_{d}^{\prime} = \frac{\mathrm{bar}_{d}}"
            r"{\sum_{i=1}^{7} \mathrm{bar}_{i}}"
        )
        title = Tex(r"\underline{$\mathbb{L}_{1}$ normalization}")
        week_vector, norm_bars, sum_bar, tick, tick_text = self.normalise_into_one(
            week, week_x, bar_chart, week_y, explain, title)
        self.wait(2)

        # --- the countdown --------------------------------------------------------
        word = Text("Week")
        number = Text(str(n_weeks))
        self.screen_lock(word, apparent_h=0.4, corner=UL,  u=0.10, v=0.195)
        self.screen_lock(number, apparent_h=0.4, corner=UL, u=0.25, v=0.195, align=RIGHT)
        self.add(number, word)

        self.play(
            FadeOut(week_plane.x_axis), FadeOut(week_plane.y_axis),
            FadeOut(week_plane.background_lines),
            FadeIn(number), FadeIn(word),
            FadeOut(tick), FadeOut(tick_text), FadeOut(sum_bar),
            FadeOut(explain), FadeOut(title),
            FadeOut(day_ticks), FadeOut(day_labels),
        )

        # Captured BEFORE the countdown's zoom-outs, so the finished chart can be
        # brought back to exactly this view.
        frame_width, frame_centre = self.camera.frame.width, self.camera.frame.get_center()

        norm_bars.set_z_index(n_weeks + 1)
        columns = [VGroup(norm_bars[d]) for d in range(WEEKDAYS)]
        totals = list(self.shown(week_y, 1.0))

        # NOTE: every bar ends up reachable from the scene twice -- through the per-week
        # group FadeIn added, and through columns[d] that the stacking Transform added.
        # Scene.add() also dissolves groups sharing members with what is being added, so
        # removal has to happen leaf by leaf, never by container.
        remaining = n_weeks - 1
        zoom_step = ZOOM_TOTAL ** (1 / max(remaining, 1))
        fan_step = FAN_TOTAL / max(remaining, 1)
        anchor = week_plane.c2p(0.2, 0)

        for step in range(remaining):
            seconds = self._step_seconds(step, remaining)
            week_start = last_week - 1 - step
            indices = [i for i, x in enumerate(xs) if week_start <= x < week_start + 1]
            previous = self.shown([values[i] for i in indices], 1.0)

            distance = step + 1
            opacity = 0.25 + (1.0 * (1 - distance / max(remaining, 1))) ** 1.5
            bars = (week.bars([xs[i] - week_start for i in indices], previous,
                              self.BAR_W, **self.bar_style(opacity))
                    .shift(week.vec(fan_step * distance, fan_step * distance))
                    .set_z_index(n_weeks - 1 - step))
            for d in range(WEEKDAYS):
                columns[d].add(bars[d])
                totals[d] += previous[d]

            self.play(
                Transform(number, Text(str(n_weeks - 1 - step), font_size=40),
                          suspend_mobject_updating=False),
                FadeIn(bars),
                self.camera.frame.animate.scale(zoom_step, about_point=anchor),
                run_time=seconds,
            )
            self.wait(seconds)
        self.play(FadeOut(number, word))

        # --- pile each weekday's bars into one column -----------------------------
        sum_text = MathTex(
            r"\begin{aligned} &\forall d \in \mathrm{days},\\"
            r"&\quad \mathrm{bar}_{d}^{\prime} = \sum_{w=1}^{" + str(n_weeks) +
            r"} \mathrm{bar}_{w,d} \end{aligned}"
        )
        self.screen_lock(sum_text, apparent_h=1.5, corner=DR, u=-0.25, v=-0.3)
        self.add(sum_text)
        self.play(Write(sum_text))

        margin_base, camera_scale = 5.0, 1.0
        for d in range(WEEKDAYS):
            stacked, _ = week.stack(columns[d], week_x[d], self.BAR_W)
            top, frame_top = stacked.get_top()[1], self.camera.frame.get_top()[1]
            margin = margin_base * camera_scale
            scale = ((top + margin - anchor[1]) / (frame_top - anchor[1])
                     if top > frame_top - margin else 1)
            camera_scale *= scale
            self.play(
                self.camera.frame.animate.scale(scale, about_point=anchor),
                Transform(columns[d], stacked),
                run_time=self._step_seconds(d, WEEKDAYS),
            )
        self.wait()

        # --- 7 stacks -> 7 bars, refitted to the plane ----------------------------
        aggregate_bars = week.bars(week_x, self.shown(totals), self.BAR_W, **self.bar_style(1))
        self.play(
            self.camera.frame.animate.set(width=frame_width).move_to(frame_centre),
            FadeIn(week_plane.x_axis), FadeIn(week_plane.y_axis),
            FadeOut(sum_text),
            *[FadeOut(bar) for column in columns for bar in column],
            FadeIn(aggregate_bars),
            run_time=2,
        )
        self.wait()

        explain_aggregate = MathTex(
            r"\forall d \in \mathrm{days},\quad"
            r"\mathrm{bar}_{d}^{\prime\prime} = \frac{\mathrm{bar}_{d}^{\prime}}"
            r"{\sum_{i=1}^{7} \mathrm{bar}_{i}^{\prime}}"
        )
        title_aggregate = Tex(r"\underline{$\mathbb{L}_{1}$ normalization again}")
        vector, norm_aggregate, aggregate_sum, aggregate_tick, aggregate_text = (
            self.normalise_into_one(week, week_x, aggregate_bars, totals,
                                    explain_aggregate, title_aggregate))
        self.wait()

        self.report_vector(f"week {n_weeks}", week_vector)
        self.report_vector(f"{n_weeks}-week aggregate", vector)
        stored = self.series_vector(embeddings, series_id)
        error = max(abs(a - b) for a, b in zip(vector, stored))
        print(f"VECTOR agrees with archetype_embeddings to {error:.2e}")

        self.play(FadeOut(explain_aggregate), FadeOut(title_aggregate), FadeOut(aggregate_sum))

        # --- zoom the y axis ------------------------------------------------------
        zoom, zoom_bars, y_top, top_tick, top_label = self.zoom_y(
            week, week_x, vector, top=y_tick, padding=1.1, label_ref=aggregate_text)
        self.play(FadeOut(aggregate_tick), FadeOut(aggregate_text))
        title_zoom = Tex("Zooming in")
        self.screen_lock(title_zoom, apparent_h=0.6, u=0.35, v=0.2)
        self.play(ReplacementTransform(norm_aggregate, zoom_bars), Write(title_zoom),
                  FadeIn(top_tick), Write(top_label), run_time=2)
        norm_aggregate = zoom_bars                 # everything downstream reads these
        self.play(FadeOut(title_zoom))

        # --- the vector, in full --------------------------------------------------
        # 24 entries had to be elided; 7 fit, so every coordinate is named and tied to
        # its bar. The link is the INDEX: entry d is built from vector[d], the same list
        # that sized norm_aggregate[d].
        matrix, entries = self.full_vector(vector, dp=3)
        matrix.move_to(zoom.pt(0.5, 0.85 * y_top))

        self.play(
            *[norm_aggregate[d].animate.set_fill(LINK_COLORS[d]).set_stroke(LINK_COLORS[d])
              for d in range(WEEKDAYS)],
            run_time=0.8,
        )
        self.play(
            FadeIn(matrix.get_brackets()),
            *[TransformFromCopy(norm_aggregate[d], entry) for d, entry in enumerate(entries)],
            lag_ratio=0.06, run_time=2.5,
        )
        for d, entry in enumerate(entries):
            self.link_number_to_bar(entry, norm_aggregate[d], zoom)
        self.wait()

        # --- split the frame: chart upper-left, vector upper-right ----------------
        # The chart needs no new coordinates: double the frame and put its centre on the
        # lower-right corner of the x-axis, and everything up-and-left of that corner
        # lands in the upper-left quarter at half apparent size.
        width, height = self.camera.frame.width, self.camera.frame.height
        corner = week.pt(self.X_MAX, 0)
        new_width, new_height = 2 * width, 2 * height
        vector_apparent_h = 0.72
        self.play(
            self.camera.frame.animate.set(width=new_width).move_to(corner),
            matrix.animate.set(height=vector_apparent_h * new_width / config.frame_width)
            .move_to(corner + RIGHT * 0.8 * (new_width / 4) + UP * (new_height / 4)),
            run_time=2,
        )
#        self.screen_lock(matrix, apparent_h=vector_apparent_h, corner=UR,
#                         u=-0.25, v=0.25, align=ORIGIN)
        self.wait()

        projection = Tex(r"$\textbf{UMAP}$ dimension \\ reduction")
        self.screen_lock(projection, apparent_h=1.2, u=0.3, v=0.7)
        self.play(FadeIn(projection))
        self.wait()
        self.play(FadeOut(projection))

        # --- the projection, in the half the split just freed ---------------------
        # Origin at the lower-left CORNER of the cloud rather than inside it: the axes
        # carry no units and no meaningful zero, so translating costs nothing and keeps
        # every arrow in one quadrant instead of fanning across half-planes.
        pad = 0.06
        raw_x, raw_y = layout["umap_x"].to_list(), layout["umap_y"].to_list()
        span_x, span_y = max(raw_x) - min(raw_x), max(raw_y) - min(raw_y)
        origin_x, origin_y = min(raw_x) - span_x * pad, min(raw_y) - span_y * pad
        width_u, height_u = span_x * (1 + 2 * pad), span_y * (1 + 2 * pad)

        point = {code: (x - origin_x, y - origin_y)
                 for code, x, y in zip(layout["series_id"].to_list(), raw_x, raw_y)}

        # Equal scene units per projection unit on both axes: a 2-D embedding has no
        # meaningful per-axis units, so stretching one to fill the wide half would
        # misrepresent the directions. Height is the binding constraint.
        u_height = 7.0
        u_scale = u_height / height_u
        u_plane = NumberPlane(
            x_range=(0, width_u, 5), y_range=(0, height_u, 5),
            x_length=width_u * u_scale, y_length=u_height,
            axis_config={"include_numbers": False},
            background_line_style={"stroke_opacity": 0.25},
        ).move_to(corner + DOWN * (new_height / 4))
        projection_chart = ch.Chart(u_plane, baseline=0.0)

        if series_id not in point:
            raise ValueError(f"{series_id} is not in archetype_umap.parquet")
        arrow = Arrow(projection_chart.pt(0, 0), projection_chart.pt(*point[series_id]),
                      buff=0, color=RED, stroke_width=10,
                      max_tip_length_to_length_ratio=0.08)
        self.play(FadeIn(u_plane), run_time=1)
        self.play(GrowArrow(arrow), run_time=1.5)
        self.wait()

        # --- one arrow per archetype, each touring the family ---------------------
        # Nothing moves in scene space: both planes stay where the split left them.
        # Only the bar heights, the seven live numbers and the arrow tips change.
        own = layout.filter(pl.col("series_id") == series_id)["archetype"][0]
        others = sorted(a for a in layout["archetype"].unique().to_list() if a != own)
        tours = [(own, RED)] + list(zip(others, [BLUE, GREEN, YELLOW]))

        for archetype, colour in tours:
            codes = self.cluster_tour(layout, archetype, n=10,
                                      first=series_id if archetype == own else None)
            arrow, norm_aggregate = self.tour_archetype(
                codes=codes, colour=colour, embeddings=embeddings, point=point, zoom=zoom,
                xs=week_x, bars=norm_aggregate, chart=projection_chart, first_arrow=arrow)
            self.wait(0.5)
        self.wait()

    # --- data ------------------------------------------------------------------------

    @staticmethod
    def series_vector(embeddings, series_id):
        """The 7 shares of one series, straight out of archetype_embeddings.

        Not recomputed from its history: the generating script applies the same double
        normalisation this scene does and asserts they agree, so re-deriving it here
        would only be a second chance to disagree.
        """
        row = embeddings.filter(pl.col("series_id") == series_id)
        if row.height != 1:
            raise ValueError(f"{series_id} not found exactly once in archetype_embeddings")
        return [float(row[f"v{d}"][0]) for d in range(WEEKDAYS)]

    @staticmethod
    def cluster_tour(layout, archetype, n=10, first=None):
        """`n` series of one archetype, ordered as a loop around its centroid.

        Ordering by angle rather than by name makes the arrow sweep the family's extent
        instead of jittering inside it, which is what shows an archetype is a REGION and
        not a point. `first` rotates the loop to begin on a given series.
        """
        rows = layout.filter(pl.col("archetype") == archetype)
        codes = rows["series_id"].to_list()
        x, y = rows["umap_x"].to_list(), rows["umap_y"].to_list()
        cx, cy = sum(x) / len(x), sum(y) / len(y)
        order = sorted(range(len(codes)), key=lambda i: math.atan2(y[i] - cy, x[i] - cx))
        step = max(len(order) // n, 1)
        picked = [order[i * step] for i in range(min(n, len(order)))]
        if first is not None and first in codes:
            j = codes.index(first)
            picked = [j] + [i for i in picked if i != j][: n - 1]
        return [codes[i] for i in picked]

    def tour_archetype(self, *, codes, colour, embeddings, point, zoom, xs, bars,
                       chart, run_time=0.8, first_arrow=None):
        """Walk the bars, the seven numbers and ONE arrow through `codes`.

        The numbers need no animation of their own: `link_number_to_bar` ties each one to
        its bar's height, so transforming the bars drags the digits along -- and because
        Transform interpolates, they count rather than jump.

        The bars are Transformed in place rather than replaced, deliberately: those same
        updaters hold references to these mobjects, and swapping in new ones would leave
        the numbers frozen on bars no longer on screen.
        """
        arrow = None
        for i, code in enumerate(codes):
            tip = chart.pt(*point[code])
            new_bars = zoom.bars(xs, self.series_vector(embeddings, code), self.BAR_W,
                                 **self.bar_style(1))
            new_arrow = Arrow(chart.pt(0, 0), tip, buff=0, color=colour, stroke_width=10,
                              max_tip_length_to_length_ratio=0.08)
            if i == 0 and first_arrow is not None and colour == RED:
                arrow = new_arrow
                self.play(Transform(bars, new_bars), FadeIn(arrow), run_time=0.1)
                self.remove(first_arrow)
            elif arrow is None:
                arrow = new_arrow
                self.play(Transform(bars, new_bars), GrowArrow(arrow), run_time=run_time)
            else:
                # Leave a permanent dot where the arrow currently ends, so the family's
                # extent is drawn out rather than merely swept.
                self.add(Dot(arrow.get_end(), radius=0.08, color=colour))
                self.play(Transform(bars, new_bars), Transform(arrow, new_arrow),
                          run_time=run_time)
        return arrow, bars

    # --- axis and number plumbing -----------------------------------------------------

    def zoom_y(self, chart, xs, values, top=None, padding=1.1, dp=4, label_ref=None):
        """Re-scale y from 0..1 to 0..top*padding, so bars a few percent tall become readable.

        Builds a SECOND plane rather than mutating chart.plane, because `plane.y_range =`
        is a silent no-op -- c2p delegates to the NumberLine -- and mutating the
        NumberLine's range would change `chart.dy(1.0)` by an order of magnitude
        underneath everything already holding the chart, the live number updaters
        included.

        The new plane is never added to the scene. It is geometrically identical to the
        old one -- same x range, same lengths, same centre, data y=0 on the same bottom
        edge -- so the white axis already on screen is correct for it and simply
        persists. Only the MEANING of a height changes.
        """
        if top is None:
            top = math.ceil(max(values) * 10 ** dp) / 10 ** dp
        if top < max(values):
            raise ValueError(f"top={top} is below the tallest bar {max(values)}")

        zoom_plane = NumberPlane(
            x_range=chart.plane.x_axis.x_range,
            y_range=(0, top * padding, top * padding),
            x_length=chart.plane.x_axis.width,
            y_length=chart.plane.y_axis.height,
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            background_line_style={"stroke_opacity": 0},
        ).move_to(chart.plane.get_center())
        zoom = ch.Chart(zoom_plane, baseline=chart.baseline)

        bars = zoom.bars(xs, values, self.BAR_W, **self.bar_style(1))
        top_point = zoom.pt(0, top)
        tick = Line(top_point + RIGHT * 0.2, top_point, stroke_width=3)
        label = Text(f"{top:.{dp}f}".rstrip("0").rstrip("."))
        if label_ref is not None:
            label.match_height(label_ref)
        label.next_to(tick, RIGHT, buff=0.15)
        return zoom, bars, top, tick, label

    def link_number_to_bar(self, entry, bar, chart):
        """Keep `entry` showing `bar`'s height in data units, every frame.

        This is the link made live: the number stops being a snapshot taken at build time
        and becomes a readout of the bar.

        DecimalNumber.set_value costs a few milliseconds whether or not the value
        changed, so seven unguarded updaters would burn real time per frame. The guard
        compares the ROUNDED values rather than testing an epsilon: an epsilon of half a
        decimal place looks equivalent but lets the display sit behind the truth, which
        shows up as a wrong last digit on a number whose whole job is to be the vector.
        """
        dp = entry.num_decimal_places

        def updater(m):
            value = bar.height / chart.dy(1.0)
            if round(value, dp) != round(m.get_value(), dp):
                m.set_value(value)

        entry.add_updater(updater)
        return entry

    @staticmethod
    def full_vector(vector, dp=3):
        """A bracketed row of all 7 coordinates, one DecimalNumber each.

        MobjectMatrix rather than DecimalMatrix so the cells stay DecimalNumber --
        individually addressable and set_value()-able. MobjectMatrix keeps the mobjects
        it is handed, so the cells returned are the ones on screen.

        Colours are passed to the constructors rather than applied afterwards:
        set_value() regenerates the glyphs, and only a colour given at construction
        survives that.
        """
        cells = [DecimalNumber(vector[d], num_decimal_places=dp, color=LINK_COLORS[d])
                 for d in range(WEEKDAYS)]
        return MobjectMatrix([cells]), cells


# One colour per weekday, used on the bar and on its number so the correspondence needs
# no explaining. Ordered Monday to Sunday.
LINK_COLORS = [BLUE_D, BLUE_B, TEAL_C, GREEN_C, YELLOW_C, GOLD_E, RED_B]
