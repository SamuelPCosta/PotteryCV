"""PotteryCV - torno de ceramica virtual.

    python main.py                       # webcam + interface
    python main.py --replay data/sessions/<pasta>
    python main.py --profile outro.json
    python main.py --viewer              # so o preview, recebendo UDP (teste do gerador sintetico)
"""
import argparse
import sys

from pottery.config import load_config  # antes do Qt (ver pottery/__init__.py)
from PySide6.QtWidgets import QApplication


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--profile", default=None, help="perfil de calibração (JSON)")
    ap.add_argument("--replay", default=None, help="pasta de sessão gravada")
    ap.add_argument("--viewer", action="store_true", help="só o preview, via UDP")
    ap.add_argument("--port", type=int, default=None, help="porta do viewer")
    ap.add_argument("--no-net", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.no_net:
        cfg.net.enabled = False
    app = QApplication(sys.argv)
    from pottery.gui import MainWindow
    if a.viewer:
        from pottery.net import Receiver
        port = a.port or cfg.net.viewer_port
        cfg.net.viewer_port = port
        win = MainWindow(cfg, receiver=Receiver(port, echo_to=("127.0.0.1", cfg.net.echo_port)))
    else:
        from pottery.pipeline import Pipeline
        pipe = Pipeline(cfg, replay=a.replay, profile=a.profile)
        win = MainWindow(cfg, pipeline=pipe)
        pipe.start()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
