"""Command-line entry point for launching the desktop application."""

import click

from .bootstrap import launch_gui


@click.command()
def main():
    """Launch the GUI."""
    launch_gui()


if __name__ == '__main__':
    main()
