__authors__ = 'Francesco Ciucci, Adeleke Maradesa'

__date__ = '16th Jan., 2024'

from pyDRTtools.GUI import launch_gui

def main():
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass

    launch_gui()

if __name__ == "__main__":
    main()
