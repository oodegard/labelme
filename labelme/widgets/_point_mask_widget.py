from __future__ import annotations

import math

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from ._info_button import InfoButton


class _InfDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that treats its maximum (1e18) as infinity and displays '∞'."""

    _INF_VALUE: float = 1e18

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMaximum(self._INF_VALUE)
        self.setDecimals(1)
        self.setSingleStep(10.0)

    def textFromValue(self, v: float) -> str:
        if v >= self._INF_VALUE:
            return "∞"
        return super().textFromValue(v)

    def valueFromText(self, text: str) -> float:
        if text.strip() in ("∞", "inf", "Inf", "INF", "infinity"):
            return self._INF_VALUE
        return super().valueFromText(text)

    def validate(self, text: str, pos: int):
        if text.strip() in ("∞", "inf", "Inf", "INF", "infinity"):
            return QtGui.QValidator.Acceptable, text, pos
        return super().validate(text, pos)

    @property
    def real_value(self) -> float:
        """Return the spinbox value as a Python float, mapping the sentinel to math.inf."""
        v = self.value()
        return math.inf if v >= self._INF_VALUE else v


class PointMaskWidget(QtWidgets.QWidget):
    """Settings panel for the Point-Mask flood-fill tool.

    The clicked position defines a circular seed region. Mean and standard
    deviation are computed from that region, then flood-fill includes connected
    pixels whose intensity is within [mean - lower_sd * sd, mean + upper_sd * sd].

    Lower SD supports ∞, interpreted as "no lower bound" (floor at 0).
    """

    _radius_spin: QtWidgets.QSpinBox
    _lo_spin: _InfDoubleSpinBox
    _hi_spin: _InfDoubleSpinBox
    _body: QtWidgets.QWidget

    pointRadiusChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.setLayout(layout)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addStretch()
        label = QtWidgets.QLabel(self.tr("Point Mask"))
        header_layout.addWidget(label)
        info_button = InfoButton(
            tooltip=self.tr(
                "In 'Point-Mask' mode, click anywhere on the image to\n"
                "flood-fill from that area using 4-connectivity.\n\n"
                "1. Build a circular seed region around the click.\n"
                "2. Compute mean and SD from that region.\n"
                "3. Include connected pixels that satisfy:\n"
                "   mean − (Lower SD * SD) ≤ pixel ≤ mean + (Upper SD * SD).\n\n"
                "Set Lower SD to ∞ to allow all darker values (down to 0)."
            )
        )
        header_layout.addWidget(info_button)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self._body = body = QtWidgets.QWidget()
        body.installEventFilter(self)
        body_layout = QtWidgets.QFormLayout()
        body_layout.setContentsMargins(0, 2, 0, 0)
        body_layout.setSpacing(4)
        body.setLayout(body_layout)

        self._radius_spin = QtWidgets.QSpinBox()
        self._radius_spin.setMinimum(1)
        self._radius_spin.setMaximum(1024)
        self._radius_spin.setValue(10)
        self._radius_spin.setSingleStep(1)
        self._radius_spin.setToolTip(
            self.tr("Radius in pixels for the circular seed region")
        )
        self._radius_spin.valueChanged.connect(self.pointRadiusChanged.emit)
        body_layout.addRow(self.tr("Point Radius (px)"), self._radius_spin)

        self._lo_spin = _InfDoubleSpinBox()
        self._lo_spin.setMinimum(0.0)
        self._lo_spin.setDecimals(1)
        self._lo_spin.setSingleStep(0.5)
        self._lo_spin.setValue(_InfDoubleSpinBox._INF_VALUE)  # default ∞
        self._lo_spin.setToolTip(
            self.tr("Lower SD multiplier (∞ = no lower bound; floor is 0)")
        )
        body_layout.addRow(self.tr("Lower SD"), self._lo_spin)

        self._hi_spin = _InfDoubleSpinBox()
        self._hi_spin.setMinimum(0.0)
        self._hi_spin.setDecimals(1)
        self._hi_spin.setSingleStep(0.5)
        self._hi_spin.setValue(2.0)
        self._hi_spin.setToolTip(
            self.tr("Upper SD multiplier (default 2.0)")
        )
        body_layout.addRow(self.tr("Upper SD"), self._hi_spin)

        layout.addWidget(body)
        self.setMaximumWidth(200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def point_radius_px(self) -> int:
        """Circular seed radius in pixels."""
        return int(self._radius_spin.value())

    @property
    def lower_sd(self) -> float:
        """Lower SD multiplier (≥ 0, possibly math.inf)."""
        return self._lo_spin.real_value

    @property
    def upper_sd(self) -> float:
        """Upper SD multiplier (≥ 0, possibly math.inf)."""
        return self._hi_spin.real_value

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def setEnabled(self, a0: bool) -> None:  # type: ignore[override]
        self._body.setEnabled(a0)

    def eventFilter(self, a0: QtCore.QObject, a1: QtCore.QEvent) -> bool:
        if a0 is self._body and not self._body.isEnabled():
            if a1.type() == QtCore.QEvent.Enter:
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    self.tr("Select 'Point-Mask' mode to enable these settings"),
                    self._body,
                )
        return super().eventFilter(a0, a1)
