from __future__ import annotations

import types

import osam
from loguru import logger
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QRunnable
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThreadPool
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QProgressDialog


class _AiModelDownloadSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    progress = pyqtSignal(int, int)


class _AiModelDownloadWorker(QRunnable):
    def __init__(self, model_type, signals: _AiModelDownloadSignals):
        super().__init__()
        self.model_type = model_type
        self.signals = signals

    def run(self):
        try:
            self.model_type.pull()
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(e)


class _MicroSamDownloadWorker(QRunnable):
    def __init__(self, model_name: str, signals: _AiModelDownloadSignals):
        super().__init__()
        self.model_name = model_name
        self.signals = signals

    def run(self):
        try:
            from labelme._automation import MicroSamSession

            session = MicroSamSession(model_name=self.model_name)
            session._get_or_load_predictor_with_progress(
                progress_callback=lambda downloaded, total: self.signals.progress.emit(
                    downloaded, total
                )
            )
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(e)


def _download_microsam_model(model_name: str, parent: QtWidgets.QWidget) -> bool:
    dialog: QProgressDialog = QProgressDialog(
        "Downloading MicroSAM model weights...\n(requires internet connection)",
        None,  # type: ignore
        0,
        1000,
        parent,
    )  # type: ignore[call-overload]
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setValue(0)

    signals: _AiModelDownloadSignals = _AiModelDownloadSignals()
    worker: _MicroSamDownloadWorker = _MicroSamDownloadWorker(model_name, signals)
    pool = QThreadPool.globalInstance()

    handle_error_attrs = types.SimpleNamespace(e=None)

    def handle_error(e: Exception):
        logger.error("Exception occurred: {}", e)
        handle_error_attrs.e = e
        QtWidgets.QApplication.setOverrideCursor(Qt.ArrowCursor)
        dialog.setRange(0, 1)  # pause busy mode
        dialog.setLabelText(
            "Failed to download MicroSAM model weights.\n(check internet connection)"
        )
        dialog.setCancelButtonText("Close")

    def handle_progress(downloaded: int, total: int):
        if total <= 0:
            dialog.setRange(0, 0)
            return

        downloaded = max(0, min(downloaded, total))
        dialog.setRange(0, 1000)
        progress_value = int((downloaded / total) * 1000)
        dialog.setValue(progress_value)

        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        dialog.setLabelText(
            "Downloading MicroSAM model weights...\n"
            f"{downloaded_mb:.1f} / {total_mb:.1f} MB"
        )

    signals.finished.connect(dialog.close)
    signals.error.connect(handle_error)
    signals.progress.connect(handle_progress)

    dialog.show()
    pool.start(worker)
    dialog.exec_()

    return handle_error_attrs.e is None


def download_ai_model(model_name: str, parent: QtWidgets.QWidget) -> bool:
    if model_name.startswith("microsam:"):
        return _download_microsam_model(model_name=model_name, parent=parent)

    model_type = osam.apis.get_model_type_by_name(model_name)

    if _is_already_downloaded := model_type.get_size() is not None:
        return True

    dialog: QProgressDialog = QProgressDialog(
        "Downloading AI model...\n(requires internet connection)",
        None,  # type: ignore
        0,
        0,
        parent,
    )  # type: ignore[call-overload]
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)

    signals: _AiModelDownloadSignals = _AiModelDownloadSignals()
    worker: _AiModelDownloadWorker = _AiModelDownloadWorker(model_type, signals)
    pool = QThreadPool.globalInstance()

    handle_error_attrs = types.SimpleNamespace(e=None)

    def handle_error(e: Exception):
        logger.error("Exception occurred: {}", e)
        handle_error_attrs.e = e
        #
        QtWidgets.QApplication.setOverrideCursor(Qt.ArrowCursor)
        dialog.setRange(0, 1)  # pause busy mode
        dialog.setLabelText("Failed to download AI model.\n(check internet connection)")
        dialog.setCancelButtonText("Close")

    signals.finished.connect(dialog.close)
    signals.error.connect(handle_error)

    dialog.show()
    pool.start(worker)
    dialog.exec_()

    return handle_error_attrs.e is None
