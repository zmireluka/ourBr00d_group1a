"""
Slim status window for the OurBr00d client (PySide6).

Activated via config.UX_EXPERIENCE = True. Changes NOTHING in the pipeline:
the proven audio loop (client.stream_audio) runs unchanged in a daemon thread;
this window only mirrors the states (connecting / listening / thinking /
speaking / ended).

Architecture:
  - Qt event loop runs on the MAIN thread (required by Qt).
  - The asyncio client runs in a daemon thread.
  - Bridge = thread-safe queue.Queue. The client pushes (state, detail)
    into it (via status hook), QTimer polls every 80ms. No direct Qt
    access from the thread → no crash.
"""

# No venv bootstrap here: gui.py is exclusively imported by client.py
# (config.UX_EXPERIENCE = True), never started directly. By this point
# client.py has already set up the .venv via _selfboot() and is running in
# the venv Python — dependencies (PySide6 etc.) are already present.

import asyncio
import math
import queue
import threading

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont


# ── Appearance ─────────────────────────────────────────────────────────────
BG     = "#16161c"
FG     = "#e8e8ee"
SUBTLE = "#7a7a88"

STATES = {
    "connecting":   ("#5a5a66", "Connecting to server…",     False),
    "listening":    ("#46c98b", "Mother is listening",        False),
    "thinking":     ("#e0a92e", "Mother is thinking…",        True),
    "speaking":     ("#4a90e2", "Mother is speaking",         True),
    "reconnecting": ("#d8743f", "Connection lost — retrying…",True),
    "ended":        ("#3a3a42", "Session ended",              False),
    "stopped":      ("#3a3a42", "Client stopped",             False),
    "error":        ("#c94646", "Error",                      False),
}

ORB_R = 54


# ── Orb widget ─────────────────────────────────────────────────────────────
class OrbWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(360, 240)
        self._color  = QColor("#5a5a66")
        self._radius = float(ORB_R)

    def set_orb(self, color_str, radius):
        self._color  = QColor(color_str)
        self._radius = radius
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))

        cx, cy = 180, 120
        r = int(self._radius)

        # Glow — semi-transparent ring
        glow_r = r + 16
        glow = QColor(self._color)
        glow.setAlpha(55)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(cx - glow_r, cy - glow_r, glow_r * 2, glow_r * 2)

        # Orb
        painter.setBrush(self._color)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        painter.end()


# ── Main window ────────────────────────────────────────────────────────────
class StatusWindow(QWidget):
    def __init__(self, status_queue):
        super().__init__()
        self.q      = status_queue
        self.state  = "connecting"
        self.detail = ""
        self.phase  = 0.0

        self.setWindowTitle("OurBr00d")
        self.setFixedSize(360, 430)
        self.setStyleSheet(f"background-color: {BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 26, 0, 4)
        layout.setSpacing(0)

        # Title
        title = QLabel("OurBr00d")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Helvetica", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {FG}; background: transparent;")
        layout.addWidget(title)

        # Orb
        self.orb_widget = OrbWidget()
        layout.addWidget(self.orb_widget)

        # Status label
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont("Helvetica", 15))
        self.label.setStyleSheet(f"color: {FG}; background: transparent;")
        layout.addWidget(self.label)

        layout.addSpacing(18)

        # Close button (centred)
        btn = QPushButton("Close")
        btn.setFixedSize(120, 34)
        btn.clicked.connect(self.close)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2a2a33;
                color: {FG};
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }}
            QPushButton:hover  {{ background-color: #3a3a44; }}
            QPushButton:pressed {{ background-color: #222228; }}
        """)
        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        btn_row_layout = QVBoxLayout(btn_row)
        btn_row_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_row_layout.addWidget(btn)
        layout.addWidget(btn_row)

        # Hint
        hint = QLabel('Say "lets kill this session" to end')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFont(QFont("Helvetica", 10))
        hint.setStyleSheet(f"color: {SUBTLE}; background: transparent;")
        layout.addWidget(hint)

        self._apply_state()

        # Queue polling
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(80)

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(40)

        # Bring to foreground
        self.raise_()
        self.activateWindow()

    def _apply_state(self):
        _, text, _ = STATES.get(self.state, STATES["connecting"])
        self.label.setText(self.detail or text)

    def _poll(self):
        try:
            while True:
                state, detail = self.q.get_nowait()
                self.state, self.detail = state, detail
                self._apply_state()
                if state in ("stopped", "error"):
                    QTimer.singleShot(2200, self.close)
        except queue.Empty:
            pass

    def _animate(self):
        color, _, pulse = STATES.get(self.state, STATES["connecting"])
        if pulse:
            self.phase += 0.18
            radius = ORB_R + 7 * (0.5 + 0.5 * math.sin(self.phase))
        else:
            radius = float(ORB_R)
        self.orb_widget.set_orb(color, radius)

    def closeEvent(self, event):
        self._poll_timer.stop()
        self._anim_timer.stop()
        event.accept()


# ── Public API ─────────────────────────────────────────────────────────────
def run(client_coro, register_hook):
    """Starts the client in a daemon thread and opens the status window.

    client_coro     — the coroutine function (client.stream_audio)
    register_hook   — client.set_status_hook, to register the queue publisher
    """
    status_queue = queue.Queue()
    register_hook(lambda state, detail="": status_queue.put((state, detail)))

    def worker():
        try:
            asyncio.run(client_coro())
        except Exception as e:
            status_queue.put(("error", str(e)))
        finally:
            status_queue.put(("stopped", ""))

    threading.Thread(target=worker, daemon=True).start()

    app = QApplication.instance() or QApplication([])
    window = StatusWindow(status_queue)
    window.show()
    app.exec()


# ── Preview mode ───────────────────────────────────────────────────────────
# Preview WITHOUT server/microphone:
#   python3 gui.py
# Cycles through all states as a demo loop.
if __name__ == "__main__":
    _hook = {}

    async def _fake_client():
        pub = _hook["fn"]
        for state, duration in [
            ("connecting", 1.2), ("listening", 2.0), ("thinking", 1.5),
            ("speaking", 3.0),   ("listening", 2.0), ("thinking", 1.2),
            ("speaking", 2.5),   ("reconnecting", 1.5), ("listening", 2.0),
            ("ended", 2.0),
        ]:
            pub(state)
            await asyncio.sleep(duration)

    run(_fake_client, lambda fn: _hook.__setitem__("fn", fn))
