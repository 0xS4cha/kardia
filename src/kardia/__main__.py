from __future__ import annotations
import os
import sys
os.environ.setdefault('QT_API', 'pyside6')

def main() -> int:
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('Kardia')
    window = MainWindow()
    window.show()
    return app.exec()
if __name__ == '__main__':
    raise SystemExit(main())