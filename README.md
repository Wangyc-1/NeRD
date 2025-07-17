# Documentation for Paper Code

## 1. Environment Setup

To ensure the proper functioning of the code, please set up the following environment within the `nerd` virtual environment:

### Dependencies

- **CUDA Toolkit**: Version 11.3.1
- **PyTorch**: Version 1.10.0
- **Python**: Version 3.8

### Installation Commands

```bash
# Create and activate the virtual environment (if not already created)
conda create -n nerd python=3.8
conda activate nerd

# Install CUDA Toolkit 11.3.1
conda install -c conda-forge cudatoolkit=11.3.1

# Install PyTorch 1.10.0 with CUDA 11.3 support
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
```

## 2. Data Structure

The code expects the data to be organized in the following directory structure:

```
├── Visda-C
│   ├── training
│   ├── validation
│   ├── train_list.txt
│   ├── validation_list.txt
```

## 3.Pretrain

You can either run the provided source domain training code using the command `python pretrain.py`

## 4.Adaptation

You need to replace some image paths and output paths in the code with your own, and then run the following command.

```
python nerd.py
```

