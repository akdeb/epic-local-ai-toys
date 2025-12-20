import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple MLX-Audio TTS (README-style)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--model", default="mlx-community/csm-1b", help="Model repo/id")
    parser.add_argument("--voice", default=None, help="Optional voice (model-dependent)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.5-2.0)")
    parser.add_argument("--file-prefix", default="mlx_tts", help="Output file prefix (no extension)")
    parser.add_argument("--audio-format", default="wav", help="Output format (wav/flac/mp3/m4a)")
    parser.add_argument("--join-audio", action="store_true", help="Join segments into a single file")
    parser.add_argument("--play", action="store_true", help="Play audio after generating")
    parser.add_argument("--verbose", action="store_true", help="Show mlx_audio CLI output")

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "mlx_audio.tts.generate",
        "--model",
        args.model,
        "--text",
        args.text,
        "--speed",
        str(args.speed),
        "--file_prefix",
        args.file_prefix,
        "--audio_format",
        args.audio_format,
    ]

    if args.voice:
        cmd += ["--voice", args.voice]
    if args.join_audio:
        cmd += ["--join_audio"]
    if args.play:
        cmd += ["--play"]

    # Avoid repo-local `mlx_audio.py` shadowing the real package by running from HOME.
    safe_cwd = str(Path.home())

    if args.verbose:
        return subprocess.run(cmd, cwd=safe_cwd).returncode

    p = subprocess.run(cmd, cwd=safe_cwd, text=True, capture_output=True)
    if p.returncode != 0:
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, end="", file=sys.stderr)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
