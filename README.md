# Efficient Speech Recognition: Model Compression, Spiking Neural Networks and Conditional Computation

The aim of this project is to allow fast inference on edge devices (mobile devices, mobile robots, etc.).

## Dataset
The task is a 10-class speech command recognition using Google's 
[Speech Command Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). 
Click the following links to download the dataset:
- [Train Data](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (place the content into ./data/speech_commands_v0.02/)
- [Test Data](https://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz) (place the content into ./data/speech_commands_test_set_v0.02/)

## Methodology
The speech commands are preprocessed into equally sized mel-spectrograms with each 128 bins and 
111 time samples. A ResNet-18 architecture is used as baseline. 

## Optimizations
Several deep learning optimization techniques are implemented:
- Weight pruning
- Static quantization
- Knowledge distillation
- Spiking Neural Networks
- Conditional Computation

Also, the implementation allows multiple techniques to be combined.

## Run
1. Login into wandb
2. Execute `python main.py -c [configuration]`
3. Optionally you can add the following arguments:
   - `-c`/`--configs`: list of configurations to run
   - `-p`/`--path`: Path to run configurations
   - `-r`/`--repeats`: Number of repeats per configuration
   - `-i`/`--id`: Resume ID for logging
   - `-s`/`--checkpoint`: Resume checkpoint
   - `-e`/`--epoch`: Initial epoch to start from
   - `--multicache`: Use parallel processing when caching dataset
