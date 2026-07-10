"""Check the remote JAX/CUDA runtime without producing experiment artifacts."""
from __future__ import annotations

import shutil
import subprocess
import sys


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise SystemExit(f"{command[0]} failed with exit {completed.returncode}:\n{output}")
    return output


def main() -> None:
    ptxas = shutil.which("ptxas")
    if not ptxas:
        raise SystemExit("ptxas was not found on PATH")
    ptxas_version = _run([ptxas, "--version"])
    if "release 12." not in ptxas_version:
        raise SystemExit(f"expected CUDA 12 ptxas, got:\n{ptxas_version}")

    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    import jaxlib  # noqa: WPS433

    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu":
        raise SystemExit(f"expected JAX backend 'gpu', got {backend!r}; devices={devices!r}")

    value = jnp.arange(4, dtype=jnp.float32).sum()
    value.block_until_ready()

    print(f"python={sys.executable}")
    print(f"jax={jax.__version__}")
    print(f"jaxlib={jaxlib.__version__}")
    print(f"ptxas={ptxas}")
    print("ptxas_version:")
    print(ptxas_version)
    print(f"backend={backend}")
    print(f"devices={[str(device) for device in devices]}")
    print(f"compile_sum={float(value)}")


if __name__ == "__main__":
    main()
