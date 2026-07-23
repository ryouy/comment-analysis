"""Yahoo!ニュース・コメント分析アプリのランタイム初期化。"""

import os

# NumPy/scikit-learnより先に設定し、Community CloudのCPU枯渇を防ぐ。
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")
