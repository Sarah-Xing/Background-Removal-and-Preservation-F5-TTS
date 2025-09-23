import torch
import torchaudio
import os
from f5_tts.model.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader


def create_dummy_wav(path, duration=1.0, sample_rate=24000):
    waveform = torch.randn(1, int(duration * sample_rate)) * 0.05
    torchaudio.save(path, waveform, sample_rate)
    return path

def main():
    # Create dummy data
    os.makedirs("tests/dummy_audio", exist_ok=True)
    dummy_data = []
    for i in range(3):
        audio_path = create_dummy_wav(f"tests/dummy_audio/test_{i}.wav", duration=i+1)
        dummy_data.append({
            "audio_path": audio_path,
            "text": f"hello world {i}",
            "duration": (i+1),
        })
    """    
    audio_path = create_dummy_wav("tests/dummy_audio/test.wav")
    dummy_data = [
        {
            "audio_path": audio_path,
            "text": "hello world",
            "duration": 1.0,
        }
    ]
    """

    # Instantiate CustomDataset
    dataset = CustomDataset(
        custom_dataset=dummy_data,
        target_sample_rate=24000,
        hop_length=256,
        n_mel_channels=80,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Batch clean mel shape:", batch["clean_mel_spec"].shape)
        print("Batch noisy mel shape:", batch["noisy_mel_spec"].shape)
        print("Batch reverb mel shape:", batch["reverberated_mel_spec"].shape)
        print("Batch text:", batch["text"])
        break 

    # Fetch a sample and print keys and shapes
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Clean mel shape:", sample["clean_mel_spec"].shape)
    print("Noisy mel shape:", sample["noisy_mel_spec"].shape)
    print("Reverberated mel shape:", sample["reverberated_mel_spec"].shape)
    print("Text:", sample["text"])

    # Optionally, plot the mel spectrograms if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Clean Mel")
        plt.imshow(sample["clean_mel_spec"].cpu(), aspect="auto", origin="lower")
        plt.subplot(1, 3, 2)
        plt.title("Noisy Mel")
        plt.imshow(sample["noisy_mel_spec"].cpu(), aspect="auto", origin="lower")
        plt.subplot(1, 3, 3)
        plt.title("Reverb Mel")
        plt.imshow(sample["reverberated_mel_spec"].cpu(), aspect="auto", origin="lower")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()