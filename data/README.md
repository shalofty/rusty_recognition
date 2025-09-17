# MNIST Dataset Setup

This directory contains the MNIST dataset files required for training and evaluation.

## Required Files

You need to download the following MNIST dataset files and place them in the `data/` subdirectory:

```
data/
└── data/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## Download Instructions

### Option 1: Direct Download (Recommended)

Download the files directly from the official MNIST database:

```bash
# Create the data subdirectory
mkdir -p data/data

# Download training images
curl -o data/data/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

# Download training labels  
curl -o data/data/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# Download test images
curl -o data/data/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

# Download test labels
curl -o data/data/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Extract all files
cd data/data
gunzip *.gz
cd ../..
```

### Option 2: Manual Download

1. Visit [MNIST Database Website](http://yann.lecun.com/exdb/mnist/)
2. Download these four files:
   - `train-images-idx3-ubyte.gz` (Training set images)
   - `train-labels-idx1-ubyte.gz` (Training set labels)
   - `t10k-images-idx3-ubyte.gz` (Test set images)
   - `t10k-labels-idx1-ubyte.gz` (Test set labels)
3. Extract the `.gz` files to get the raw binary files
4. Place the extracted files in `data/data/`

## File Information

| File | Description | Size |
|------|-------------|------|
| `train-images-idx3-ubyte` | Training set images (60,000 images) | ~47 MB |
| `train-labels-idx1-ubyte` | Training set labels (60,000 labels) | ~60 KB |
| `t10k-images-idx3-ubyte` | Test set images (10,000 images) | ~7.8 MB |
| `t10k-labels-idx1-ubyte` | Test set labels (10,000 labels) | ~10 KB |

## Dataset Format

The MNIST dataset uses a custom IDX file format:

- **Images**: 28×28 pixel grayscale images (0-255 values)
- **Labels**: Single digit class labels (0-9)
- **Format**: Big-endian binary format with header information

The Rust code automatically handles:
- IDX format parsing
- Normalization to [0,1] float range
- Batch loading and preprocessing

## Verification

After downloading, you can verify the setup by running:

```bash
# Check if files exist and have correct sizes
ls -la data/data/

# Run a quick training test
cargo run --package mnist-runner -- sanity --batch 32 --steps 10 --lr 0.01
```

## Troubleshooting

### Files Not Found Error
```
Error: No such file or directory (os error 2)
```
- Ensure files are in `data/data/` directory (note the nested structure)
- Check that files are extracted (not still `.gz` compressed)
- Verify filenames match exactly (case-sensitive)

### Permission Errors
```bash
# Make sure files are readable
chmod 644 data/data/*
```

### Download Issues
If direct download fails, try:
```bash
# Use wget instead of curl
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

# Or download with browser and move files manually
```

## Dataset License

The MNIST database is available under Yann LeCun's terms:
- Free for research and educational purposes
- Original source: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges

---

*Once you have the dataset files in place, you're ready to start training! See the main README for training commands.*
