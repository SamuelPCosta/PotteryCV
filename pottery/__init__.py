# Python 3.15 + PySide6: se o Qt for importado antes, `concurrent.futures.ThreadPoolExecutor` fica como um
# objeto 'lazy_import' nao resolvido e o MediaPipe quebra. Forcamos a resolucao aqui (importe `pottery` antes do Qt).
import concurrent.futures as _cf
import os as _os

_cf.ThreadPoolExecutor, _cf.ProcessPoolExecutor
# Abertura da webcam via MSMF fica ~10 s mais lenta enumerando transformacoes de hardware.
_os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
