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

    Shows Min and Max intensity-offset spinboxes.  The flood fill starting from
    the clicked pixel includes all 4-connected pixels whose intensity lies in
    [seed_value - min, seed_value + max].  Max supports ∞.
    """

    _lo_spin: _InfDoubleSpinBox
    _hi_spin: _InfDoubleSpinBox
    _body: QtWidgets.QWidget

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
                "flood-fill from that pixel using 4-connectivity.\n\n"
                "A pixel is included when its intensity satisfies:\n"
                "  seed − Min  ≤  pixel  ≤  seed + Max\n\n"
                "Set Max to ∞ to include all brighter pixels."
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

        self._lo_spin = _InfDoubleSpinBox()
        self._lo_spin.setMinimum(0.0)
        self._lo_spin.setValue(0.0)
        self._lo_spin.setToolTip(
            self.tr("Lower offset: include pixels with value ≥ seed − Min")
        )
        body_layout.addRow(self.tr("Min"), self._lo_spin)

        self._hi_spin = _InfDoubleSpinBox()
        self._hi_spin.setMinimum(0.0)
        self._hi_spin.setValue(_InfDoubleSpinBox._INF_VALUE)  # default ∞
        self._hi_spin.setToolTip(
            self.tr("Upper offset: include pixels with value ≤ seed + Max  (∞ = no upper limit)")
        )
        body_layout.addRow(self.tr("Max"), self._hi_spin)

        layout.addWidget(body)
        self.setMaximumWidth(200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def min_value(self) -> float:
        """Lower intensity offset (≥ 0, possibly math.inf)."""
        return self._lo_spin.real_value

    @property
    def max_value(self) -> float:
        """Upper intensity offset (≥ 0, possibly math.inf)."""
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
