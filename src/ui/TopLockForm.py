from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class TopLockForm(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        QApplication.instance().installEventFilter(self)

        self.focus_timer = QTimer(self)
        self.focus_timer.setInterval(100)
        self.focus_timer.timeout.connect(self.keep_focus)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        message = QLabel("화면이 잠겼습니다")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(message)

        self.setLayout(layout)

    def start_lock(self):
        self.focus_timer.start()
        self.keep_focus()

    def stop_lock(self):
        self.focus_timer.stop()

    def keep_focus(self):
        if self.isVisible():
            self.raise_()
            self.activateWindow()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def eventFilter(self, watched, event):
        if not self.isVisible():
            return False

        if event.type() == QEvent.Type.KeyPress:
            key_event = event
            if (
                isinstance(key_event, QKeyEvent)
                and key_event.key() == Qt.Key.Key_Q
                and key_event.modifiers()
                == (Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.ShiftModifier)
            ):
                self.controller.close()
            return True

        if event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        ):
            self.keep_focus()
            return True

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        event.accept()

    def mousePressEvent(self, event):
        self.keep_focus()
        event.accept()
