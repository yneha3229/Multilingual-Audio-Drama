"""Generate one background music track with Stable Audio 3 Small-Music.

This script runs inside the separate Stable Audio 3 environment (it needs
PyTorch), not the project's venv. background_music.py calls it as a
subprocess, so the main project stays light.

The output is a WAV in the same format as the voice clips (22050 Hz, mono,
16-bit), peak-normalised; background_music.py sets the final level.

Usage (normally called by background_music.py):
    python music_worker.py --prompt "slow ambient sitar, 70 BPM" --seconds 115 --seed 42 --out music.wav
"""

import argparse
import wave

import torchaudio
from stable_audio_3 import StableAudioModel

MODEL = "small-music"  # 433M parameters, runs on CPU, up to 120 seconds
PEAK = 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rate", type=int, default=22050)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = StableAudioModel.from_pretrained(MODEL)
    audio = model.generate(
        prompt=args.prompt,
        duration=args.seconds,
        seed=args.seed,
        steps=args.steps,
        chunked_decode=True,  # decode in pieces: lower peak memory on small laptops
    )
    # (batch, channels, samples) at 44.1 kHz stereo -> mono at the voices' rate
    mono = audio[0].float().cpu().mean(dim=0, keepdim=True)
    mono = torchaudio.functional.resample(mono, model.model.sample_rate, args.rate)
    peak = mono.abs().max().item()
    if peak > 0:
        mono = mono * (PEAK / peak)
    pcm = (mono.clamp(-1, 1) * 32767).round().short().numpy().tobytes()

    with wave.open(args.out, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(args.rate)
        out.writeframes(pcm)
    print(f"Wrote {args.out} ({mono.shape[-1] / args.rate:.1f}s)")


if __name__ == "__main__":
    main()
