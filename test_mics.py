"""Test all microphone devices to find which ones are picking up sound."""
import sounddevice as sd
import numpy as np

TEST_DURATION = 2  # seconds per device

devices = sd.query_devices()
input_devices = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]

print(f"Testing {len(input_devices)} input devices ({TEST_DURATION}s each)...")
print("Make some noise / speak while this runs!\n")
print(f"{'Idx':<5} {'RMS':<12} {'Status':<10} Device Name")
print("-" * 70)

for idx, dev in input_devices:
    name = dev["name"]
    sr = int(dev["default_samplerate"])
    try:
        recording = sd.rec(int(TEST_DURATION * sr), samplerate=sr, channels=1, dtype="float32", device=idx)
        sd.wait()
        rms = np.sqrt(np.mean(recording**2))
        status = "** ACTIVE **" if rms > 0.005 else "silent"
        print(f"[{idx:<2}]  {rms:<12.6f} {status:<12} {name}")
    except Exception as e:
        err_msg = str(e).split("\n")[0][:40]
        print(f"[{idx:<2}]  {'---':<12} {'ERROR':<12} {name} ({err_msg})")

print("\nDone. Use --device <idx> with any ACTIVE device.")
