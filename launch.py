# -*- coding: utf-8 -*-
"""Windows multiprocessing-safe launcher for SuperDRTtools."""

import multiprocessing as mp


def main():
    # Keep this import inside the guarded function.  Spawned Fit All workers
    # execute launch.py while preparing, but must not import the GUI/SciPy stack.
    from pyDRTtools.app.bootstrap import launch_gui
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass

    launch_gui()


if __name__ == "__main__":
    mp.freeze_support()
    main()
