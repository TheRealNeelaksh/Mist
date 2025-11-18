#!/usr/bin/env python3
"""
GUI-like Text UI using Rich.

Provides a nicer interface than cli_app, but still works in terminal.
"""

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import subprocess
import shlex

console = Console()


def run(cmd):
    console.print(f"[bold green]Running:[/bold green] {cmd}")
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        console.print(line.rstrip())
    proc.wait()


def main():
    console.rule("[bold blue]Mist2 - GUI CLI[/bold blue]")

    while True:
        table = Table(title="Mist2 Options")
        table.add_column("Key", justify="center")
        table.add_column("Action")

        table.add_row("1", "Full protect (Adversarial + Watermark)")
        table.add_row("2", "Watermark only")
        table.add_row("3", "Adversarial only")
        table.add_row("4", "Exit")

        console.print(table)

        choice = Prompt.ask("Select", choices=["1", "2", "3", "4"])

        # Full pipeline
        if choice == "1":
            inp = Prompt.ask("Input file path")
            out = Prompt.ask("Output path", default="outputs/protected.jpg")
            key = Prompt.ask("Watermark key", default="mykey")

            cmd = (
                f'python protect_image.py --input "{inp}" --out "{out}" '
                f'--key "{key}" --do_adv --do_watermark'
            )
            run(cmd)

        # Watermark only
        elif choice == "2":
            inp = Prompt.ask("Input file path")
            out = Prompt.ask("Output path", default="outputs/wm.jpg")
            key = Prompt.ask("Watermark key", default="mykey")

            cmd = (
                f'python protect_image.py --input "{inp}" --out "{out}" '
                f'--key "{key}" --do_watermark'
            )
            run(cmd)

        # Adversarial only
        elif choice == "3":
            inp = Prompt.ask("Input file path")
            out = Prompt.ask("Output path", default="outputs/adv.jpg")

            cmd = (
                f'python protect_image.py --input "{inp}" --out "{out}" '
                f'--do_adv'
            )
            run(cmd)

        else:
            console.print("[bold yellow]Goodbye.[/bold yellow]")
            break


if __name__ == "__main__":
    main()
