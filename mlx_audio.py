import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _default_outputs_dir() -> Path:
    return Path.home() / ".mlx_audio" / "outputs"


def _find_latest_output(outputs_dir: Path, started_at: float) -> Path | None:
    if not outputs_dir.exists():
        return None

    candidates: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        candidates.extend(outputs_dir.glob(ext))

    candidates = [p for p in candidates if p.is_file() and p.stat().st_mtime >= started_at - 0.5]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_csm(
    *,
    text: str,
    model: str,
    speed: float,
    ref_audio: str | None,
    play: bool,
    outputs_dir: Path,
    out_path: str | None,
    verbose: bool,
) -> int:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    cmd = [sys.executable, "-m", "mlx_audio.tts.generate", "--model", model, "--text", text]
    cmd += ["--speed", str(speed)]
    if ref_audio:
        cmd += ["--ref_audio", ref_audio]
    if play:
        cmd += ["--play"]

    # IMPORTANT:
    # This repo contains a local `mlx_audio.py` (this file). If the subprocess ran from
    # this repo directory, `python -m mlx_audio...` could import this file instead of the
    # installed `mlx_audio` package. Running from HOME avoids that shadowing.
    safe_cwd = str(Path.home())

    if verbose:
        print("[csm-test] Running:")
        print(" ".join(cmd))
        print(f"[csm-test] Using outputs dir: {outputs_dir}")

    try:
        subprocess.run(cmd, check=True, cwd=safe_cwd)
    except subprocess.CalledProcessError as e:
        print("[csm-test] mlx_audio generation failed.", file=sys.stderr)
        print("[csm-test] Install MLX-Audio in this environment:", file=sys.stderr)
        print("  pip install mlx-audio", file=sys.stderr)
        return e.returncode

    generated = _find_latest_output(outputs_dir, started_at)
    if not generated:
        print(
            "[csm-test] Generation completed but no new output file was found in outputs dir.\n"
            "[csm-test] Check the mlx_audio CLI logs above for where it wrote the file.",
            file=sys.stderr,
        )
        return 1

    if out_path:
        dst = Path(out_path).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, dst)
        print(f"[csm-test] Copied output to: {dst}")
    else:
        print(f"[csm-test] Output file: {generated}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Sesame CSM (mlx-community/csm-1b) via MLX-Audio")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--model", default="mlx-community/csm-1b", help="Model repo/id")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.5-2.0)")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path (optional)")
    parser.add_argument("--out", default=None, help="Optional output path to copy to")
    parser.add_argument("--outputs-dir", default=str(_default_outputs_dir()), help="MLX-Audio outputs dir")
    parser.add_argument("--play", action="store_true", help="Pass --play to mlx_audio CLI")
    parser.add_argument("--verbose", action="store_true", help="Print the exact command")

    args = parser.parse_args()

    ref_audio = args.ref_audio
    if ref_audio:
        ref_path = Path(ref_audio).expanduser()
        if not ref_path.exists():
            print(f"[csm-test] ref audio does not exist: {ref_path}", file=sys.stderr)
            return 2
        ref_audio = str(ref_path)

    outputs_dir = Path(args.outputs_dir).expanduser()
    return run_csm(
        text=args.text,
        model=args.model,
        speed=args.speed,
        ref_audio=ref_audio,
        play=args.play,
        outputs_dir=outputs_dir,
        out_path=args.out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
