from __future__ import annotations

from typing import Final

import pytest
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QPointF

from labelme.shape import Shape
from labelme.widgets.canvas import Canvas
from labelme.widgets.canvas import MASK_FRAME_TIME_KEY
from labelme.widgets.canvas import MASK_FRAME_Z_KEY

_WIDTH: Final = 100
_HEIGHT: Final = 50


@pytest.fixture()
def canvas(qtbot) -> Canvas:
    canvas = Canvas()
    canvas.pixmap = QtGui.QPixmap(_WIDTH, _HEIGHT)
    qtbot.addWidget(canvas)
    return canvas


@pytest.mark.gui
@pytest.mark.parametrize(
    ("point", "is_outside"),
    [
        (QPointF(_WIDTH / 2, _HEIGHT / 2), False),
        (QPointF(0, 0), False),
        (QPointF(_WIDTH, _HEIGHT), False),
        (QPointF(_WIDTH, _HEIGHT / 2), False),
        (QPointF(_WIDTH / 2, _HEIGHT), False),
        (QPointF(_WIDTH + 0.1, _HEIGHT / 2), True),
        (QPointF(_WIDTH / 2, _HEIGHT + 0.1), True),
        (QPointF(-0.1, _HEIGHT / 2), True),
        (QPointF(_WIDTH / 2, -0.1), True),
    ],
)
def test_outOfPixmap(canvas: Canvas, point: QPointF, is_outside: bool):
    assert canvas.outOfPixmap(point) is is_outside


@pytest.mark.gui
@pytest.mark.parametrize(
    ("p1", "p2", "pt_intersection"),
    [
        (
            pt_center := QPointF(_WIDTH / 2, _HEIGHT / 2),
            QPointF(_WIDTH + 50, _HEIGHT / 2),  # to the right
            QPointF(_WIDTH, _HEIGHT / 2),  # right edge
        ),
        (
            pt_center,
            QPointF(_WIDTH / 2, -10),  # to the top
            QPointF(_WIDTH / 2, 0),  # top edge
        ),
        (
            pt_center,
            QPointF(-10, _HEIGHT / 2),  # to the left
            QPointF(0, _HEIGHT / 2),  # left edge
        ),
        (
            pt_center,
            QPointF(_WIDTH / 2, _HEIGHT + 30),  # to the bottom
            QPointF(_WIDTH / 2, _HEIGHT),  # bottom edge
        ),
    ],
)
def test_intersectionPoint(
    canvas: Canvas, p1: QPointF, p2: QPointF, pt_intersection: QPointF
):
    assert canvas.intersectionPoint(p1, p2) == pt_intersection


@pytest.mark.gui
def test_isVisible_filters_mask_by_current_frame(canvas: Canvas) -> None:
    shape = Shape(shape_type="mask")
    shape.addPoint(QPointF(0, 0))
    shape.addPoint(QPointF(2, 2))
    shape.mask = np.ones((3, 3), dtype=bool)
    shape.other_data[MASK_FRAME_TIME_KEY] = 3
    shape.other_data[MASK_FRAME_Z_KEY] = 5

    canvas.setFrameContext(time_index=3, z_index=5, enabled=True)
    assert canvas.isVisible(shape) is True

    canvas.setFrameContext(time_index=4, z_index=5, enabled=True)
    assert canvas.isVisible(shape) is False


@pytest.mark.gui
def test_isVisible_keeps_mask_visible_without_frame_metadata(canvas: Canvas) -> None:
    shape = Shape(shape_type="mask")
    shape.addPoint(QPointF(0, 0))
    shape.addPoint(QPointF(1, 1))
    shape.mask = np.ones((2, 2), dtype=bool)

    canvas.setFrameContext(time_index=10, z_index=10, enabled=True)
    assert canvas.isVisible(shape) is True


@pytest.mark.gui
def test_isVisible_filters_polygon_by_current_frame(canvas: Canvas) -> None:
    shape = Shape(shape_type="polygon")
    shape.addPoint(QPointF(1, 1))
    shape.addPoint(QPointF(5, 1))
    shape.addPoint(QPointF(3, 4))
    shape.close()
    shape.other_data[MASK_FRAME_TIME_KEY] = 2
    shape.other_data[MASK_FRAME_Z_KEY] = 1

    canvas.setFrameContext(time_index=2, z_index=1, enabled=True)
    assert canvas.isVisible(shape) is True

    canvas.setFrameContext(time_index=2, z_index=0, enabled=True)
    assert canvas.isVisible(shape) is False
