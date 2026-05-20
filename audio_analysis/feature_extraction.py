import librosa
import numpy as np


def extract_features(file_path):
    try:
        # Remove duration limit — carrega o áudio completo
        audio, sr = librosa.load(file_path, sr=22050)

        features = []

        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfccs, axis=1).tolist())
        features.extend(np.std(mfccs, axis=1).tolist())

        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=512)
        features.extend(np.mean(chroma, axis=1).tolist())

        # 3. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=32)
        features.extend(np.mean(mel, axis=1).tolist())

        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=512, n_bands=6)
        features.extend(np.mean(contrast, axis=1).tolist())

        # 5. ZCR
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # 7. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        result = np.array(features, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None