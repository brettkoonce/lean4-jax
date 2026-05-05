#!/usr/bin/env python3
"""Pre-process Oxford-IIIT Pets to raw binary format for the Lean loader.

Usage: python3 preprocess_pets.py <pets_dir> <output_dir>

  <pets_dir> must contain:
    images/                  JPEG photos
    annotations/trimaps/     PNG trimaps (1=foreground, 2=background, 3=boundary)
    annotations/trainval.txt 3680 training examples
    annotations/test.txt     3669 val/test examples

Writes train.bin, val.bin in the following format:
  Header: count (4 bytes, little-endian uint32)
  Per record:
    image: 224*224*3 bytes (channel-first RGB; row-major within each channel)
    mask:  224*224     bytes (per-pixel class label, 0/1/2)
  Total per record: 200,704 bytes.

Mask classes (after remapping from trimap 1/2/3 -> 0/1/2):
  0 = foreground (the pet)
  1 = background
  2 = boundary  (the buffer pixels around the pet outline)
"""
import os, sys, struct
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required. Install with: pip install Pillow numpy")
    sys.exit(1)

SIZE = 224  # both images and masks resized to 224x224


def load_split_list(annotation_path):
    """Read trainval.txt or test.txt and return list of base filenames (no extension)."""
    names = []
    with open(annotation_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: <name> <class_id> <species> <breed_id>
            name = line.split()[0]
            names.append(name)
    return names


def process_split(pets_dir, names, out_path):
    images_dir = os.path.join(pets_dir, 'images')
    masks_dir = os.path.join(pets_dir, 'annotations', 'trimaps')

    records = []
    skipped = 0

    for name in names:
        img_path = os.path.join(images_dir, name + '.jpg')
        mask_path = os.path.join(masks_dir, name + '.png')

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            skipped += 1
            continue

        try:
            # Image: RGB, resize to 224x224 with bilinear, channel-first uint8
            img = Image.open(img_path).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
            img_arr = np.asarray(img, dtype=np.uint8)            # (H, W, 3)
            img_chw = img_arr.transpose(2, 0, 1).copy()          # (3, H, W)
            img_bytes = img_chw.tobytes()                         # 3*224*224 bytes
            assert len(img_bytes) == 3 * SIZE * SIZE, len(img_bytes)

            # Mask: nearest-neighbor resize (categorical), remap 1/2/3 -> 0/1/2
            mask = Image.open(mask_path).resize((SIZE, SIZE), Image.NEAREST)
            mask_arr = np.asarray(mask, dtype=np.uint8)          # (H, W) values 1/2/3
            mask_remapped = (mask_arr - 1).astype(np.uint8)      # values 0/1/2
            assert mask_remapped.min() >= 0 and mask_remapped.max() <= 2, \
                f"unexpected mask values for {name}: {np.unique(mask_arr)}"
            mask_bytes = mask_remapped.tobytes()                  # 224*224 bytes
            assert len(mask_bytes) == SIZE * SIZE, len(mask_bytes)

            records.append(img_bytes + mask_bytes)
        except Exception as e:
            print(f"  skipping {name}: {e}", file=sys.stderr)
            skipped += 1

    print(f"  {len(records)} records ({skipped} skipped)")

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(records)))
        for r in records:
            f.write(r)
    print(f"  wrote {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    pets_dir = sys.argv[1]
    out_dir = sys.argv[2]

    train_names = load_split_list(os.path.join(pets_dir, 'annotations', 'trainval.txt'))
    val_names   = load_split_list(os.path.join(pets_dir, 'annotations', 'test.txt'))
    print(f"trainval: {len(train_names)} | test: {len(val_names)}")

    print("Processing training split...")
    process_split(pets_dir, train_names, os.path.join(out_dir, 'train.bin'))

    print("Processing validation split...")
    process_split(pets_dir, val_names, os.path.join(out_dir, 'val.bin'))

    print("Done.")


if __name__ == '__main__':
    main()
