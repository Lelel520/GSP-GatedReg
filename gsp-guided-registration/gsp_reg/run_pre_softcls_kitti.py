import os
import runpy

os.environ.setdefault("KP_ARCH", "kitti_full")
os.environ.setdefault("KP_FIRST_SUBSAMPLING_DL", "0.3")
os.environ.setdefault("KP_CONV_RADIUS", "4.25")
os.environ.setdefault("KP_EXTENT", "2.0")
os.environ.setdefault("PREPROC_DIR", "data/kitti/preprocessed")
os.environ.setdefault("CORR_DIR", "data/kitti/preprocessed_corr")
os.environ.setdefault("STOPO_DIR", "data/kitti/preprocessed_stopo_abs")
os.environ.setdefault("TAU_LOW", "0.03")
os.environ.setdefault("TAU_HIGH", "0.20")

runpy.run_path("gsp_reg/run_pre_softcls.py", run_name="__main__")
