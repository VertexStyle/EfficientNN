import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio
from torch.utils.data import Dataset
try:
    import sounddevice as sd
except OSError as e:
    print('Sounddevice could not be imported!')

class GoogleSpeechCommandsDataset(Dataset):
    def __init__(self,
                 data_dir,
                 cache_dir='./cache/data/',
                 labels=('up', 'down', 'left', 'right', 'go', 'stop', 'yes', 'no', 'on', 'off'),
                 transform=None,
                 encoder='mfcc',
                 target_length=111,
                 augment=0,
                 max_pitch=5,
                 train=True,
                 target_sample_rate=22050,
                 sample_limit=None,
                 logging=True):
        """
        Using Google Speech Commands Dataset: https://www.tensorflow.org/datasets/catalog/speech_commands
        Download Train: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
        Download Test: https://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
        """

        # Initialize the audio encoder
        self.target_sample_rate = target_sample_rate
        if isinstance(encoder, str) and encoder.lower() == 'hz':
            self.encoder = torchaudio.transforms.Spectrogram(target_sample_rate)
            self.encoder_name = '-enc_hz'
            self.encoder_id = 'hz'
        elif isinstance(encoder, str) and encoder.lower() == 'mel':
            self.encoder = torchaudio.transforms.MelSpectrogram(target_sample_rate)
            self.encoder_name = '' # TODO: empty for backwards compatibility
            self.encoder_id = 'mel'
        elif isinstance(encoder, str) and encoder.lower() == 'mfcc':
            self.encoder = torchaudio.transforms.MFCC(target_sample_rate)
            self.encoder_name = '-enc_mfcc'
            self.encoder_id = 'mfcc'
        elif callable(encoder):
            self.encoder = encoder
            self.encoder_name = f'-enc_{encoder.__class__.__name__.lower()}'
            self.encoder_id = 'custom'
        else:
            raise RuntimeError('Invalid encoder provided! Either set type using ("hz", "mel", "mfcc") or your own'
                               'callable encoder')

        self.data_dir = data_dir
        self.cache_dir = os.path.join(cache_dir, os.path.basename(data_dir))
        self.labels = labels
        self.label_indices = {lbl: i for i, lbl in enumerate(labels)}
        self.transform = transform
        self.target_length = target_length
        self.augment = augment
        self.max_pitch = max_pitch
        self.train = train
        self.logging = logging
        self.file_paths = []

        if self.logging:
            print(f'--- {"Train" if self.train else "Test"} Dataset ---')
            print(f'Path:\t{self.data_dir}')
            print(f'Cache:\t{self.cache_dir}')
            print(f'Labels:\t{self.labels}')

        # Retrieve all audiofile paths
        aug_index = 0
        self.label_count = {}
        for label in self.labels:
            lbl_paths = [os.path.join(self.data_dir, label, f) for f in os.listdir(os.path.join(self.data_dir, label))]
            self.label_count[label] = len(lbl_paths)
            for data_path in lbl_paths:
                for aug_id in range(self.augment+1):
                    data_idx, pitch_shift = self.__augment_index(aug_index, data_path=data_path)
                    pitch_str = f'_pitch_{pitch_shift:.2f}' if pitch_shift != 0 else ''
                    cache_path = os.path.join(self.cache_dir,
                                            f'{os.path.basename(data_path)}-{label}{self.encoder_name}{pitch_str}.pt')
                    self.file_paths.append((data_path, cache_path, label, pitch_shift))
                    aug_index += 1
        self.sample_limit = sample_limit if sample_limit is not None else len(self.file_paths)

        print(f'Length:\t{len(self)}')
        print()

        # Create cache directory if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return min(self.sample_limit, len(self.file_paths))

    def __augment_index(self, augmented_index, data_path=None):
        data_index = augmented_index // (self.augment+1)
        augment_number = augmented_index % (self.augment+1)
        is_original = augment_number == 0
        if data_path is None:
            try:
                data_path, _, _, _ = self.file_paths[augmented_index]
            except IndexError as e:
                print(f'Error: augmented_index = {augmented_index} len(file_paths) = {len(self.file_paths)}')
                raise e

        if is_original:
            pitch_shift = 0
        else:
            # Calculate seed from filepath and augmentation id
            filename = os.path.basename(data_path)
            hash_string = f'{filename}_{augment_number}'
            encoded_string = hash_string.encode()
            md5_hash = hashlib.md5()
            md5_hash.update(encoded_string)
            hash_digest = md5_hash.hexdigest()
            seed = int(hash_digest, 16)

            random.seed(seed)
            pitch_shift = random.gauss(0, self.max_pitch)
        return data_index, pitch_shift

    def load_audio(self, idx):
        data_idx, pitch_shift = self.__augment_index(idx)
        data_path, cache_path, label, _ = self.file_paths[data_idx]

        # Load and process audio
        waveform, sample_rate = torchaudio.load(data_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)

        # Pitch shifting
        # waveform = torchaudio.sox_effects.apply_effects_tensor(waveform, self.target_sample_rate,
        #                                                        [['pitch', str(round(pitch_shift))]])[0]
        if pitch_shift != 0:
            waveform = librosa.effects.pitch_shift(waveform.numpy().squeeze(), self.target_sample_rate,
                                                   n_steps=pitch_shift)
            waveform = torch.from_numpy(waveform).view(1, -1)

        return waveform, self.target_sample_rate, pitch_shift, data_path

    def __getitem__(self, idx):
        data_idx, pitch_shift = self.__augment_index(idx)
        data_path, cache_path, label, _ = self.file_paths[idx]

        # Check if the spectrogram is cached
        if os.path.exists(cache_path):
            audio_encoding = torch.load(cache_path)
        else:
            waveform, sample_rate, _, audio_path = self.load_audio(idx)
            audio_encoding = self.encoder(waveform)
            torch.save(audio_encoding, cache_path)

        if self.transform:
            audio_encoding = self.transform(audio_encoding)

        # Adjust the length of the spectrogram
        current_length = audio_encoding.shape[2]

        if current_length > self.target_length:
            # Truncate the spectrogram
            audio_encoding = audio_encoding[:, :, :self.target_length]
        elif current_length < self.target_length:
            # Pad the spectrogram
            padding_size = self.target_length - current_length
            padding = torch.zeros(audio_encoding.shape[0], audio_encoding.shape[1], padding_size)
            audio_encoding = torch.cat((audio_encoding, padding), dim=2)

        label_tensor = torch.zeros(len(self.labels))
        lbl_idx = self.label_indices[label]
        label_tensor[lbl_idx] = 1
        return audio_encoding, label_tensor, lbl_idx, label, idx, pitch_shift

    def precache_single(self):
        items = [i for i in range(len(self))]
        do_pre_cache = False
        for i in items:
            try:
                _, cache_path, _, _ = self.file_paths[i]
            except IndexError as e:
                print(f'Error: index = {i}, len(file_paths) = {len(self.file_paths)}, len(self) = {len(self)}')
                raise e
            if not os.path.exists(cache_path):
                do_pre_cache = True
                break

        if do_pre_cache:
            if self.logging:
                print('Caching the dataset...')
                for i in tqdm(items):
                    self.__getitem__(i)
            else:
                for i in items:
                    self.__getitem__(i)

    def precache(self):
        items = [i for i in range(len(self))]
        do_pre_cache = False

        # Check if any file needs caching
        for i in items:
            try:
                _, cache_path, _, _ = self.file_paths[i]
            except IndexError as e:
                print(f'Error: index = {i}, len(file_paths) = {len(self.file_paths)}, len(self) = {len(self)}')
                raise e
            if not os.path.exists(cache_path):
                do_pre_cache = True
                break

        def cache_item(i):
            try:
                self.__getitem__(i)
            except Exception as e:
                print(f"Error caching item {i}: {e}")

        if do_pre_cache:
            if self.logging:
                print('Caching the dataset...')
                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(cache_item, i): i for i in items}
                    with tqdm(total=len(items)) as progress:
                        for future in as_completed(futures):
                            progress.update(1)
            else:
                with ProcessPoolExecutor() as executor:
                    executor.map(cache_item, items)

    def play(self, idx, blocking=True):
        waveform, sample_rate, pitch_shift, audio_path = self.load_audio(idx)

        # Play the audio
        audio_data = waveform.numpy()
        audio_data = np.squeeze(audio_data)
        print(f'Play audio "{audio_path}" with pitch {pitch_shift:.2f}')
        sd.play(audio_data, sample_rate, blocking=blocking)
        sd.wait()

    def show(self, idx):
        encoding, label, lbl_idx, label_name, index, pitch_shift = self[idx]
        encoding = encoding.numpy()
        plt.imshow(encoding[0])
        plt.title(f'{self.encoder.__class__.__name__} Encoding')
        plt.suptitle(f'Label: {label_name}, Index: {index}, Pitch-Shift: {pitch_shift:.2f}')
        plt.show()

    def size(self):
        return self.__getitem__(0)[0].shape




if __name__ == '__main__':
    test_data = GoogleSpeechCommandsDataset('data/speech_commands_test_set_v0.02', encoder='mel')
    test_data.precache()
    train_data = GoogleSpeechCommandsDataset('data/speech_commands_v0.02', encoder='mfcc', augment=3)
    train_data.precache()

    train_data.play(0)
    train_data.play(1)
    train_data.show(0)
    train_data.show(1)
