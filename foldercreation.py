import os
import shutil
import re

def reorganize_dataset(dataset_root):
    """Handles actual observed file structure with missing sequences"""
    for root, dirs, files in os.walk(dataset_root):
        if 'DDoS' in root or 'Normal' in root:
            print(f"\nProcessing {root}...")
            files = [f for f in files if f.endswith('.png')]
            
            # Extract sequence numbers from actual filenames
            seq_map = {}
            for f in files:
                # Match patterns like ddos-228-0.png or norm-1000-1.png
                match = re.match(r'(ddos|norm)-(\d+)-(\d+)\.png', f)
                if match:
                    prefix, seq_num, frame_num = match.groups()
                    seq_num = int(seq_num)
                    if seq_num not in seq_map:
                        seq_map[seq_num] = []
                    seq_map[seq_num].append((int(frame_num), f))
            
            # Create sequence folders
            for seq_num in sorted(seq_map.keys()):
                frames = sorted(seq_map[seq_num])
                if len(frames) == 8:  # Only complete sequences
                    seq_dir = os.path.join(root, f'seq_{seq_num}')
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    for frame_num, f in frames:
                        src = os.path.join(root, f)
                        dst = os.path.join(seq_dir, f'frame_{frame_num}.png')
                        shutil.move(src, dst)
                    print(f"Created {seq_dir} with {len(frames)} frames")

if __name__ == "__main__":
    dataset_root = "dataset_root"  # Confirm this is your correct path
    reorganize_dataset(dataset_root)
    print("\nReorganization complete! Verifying...")
    
    # Verification
    for split in ['train', 'val']:
        for cls in ['DDoS', 'Normal']:
            path = os.path.join(dataset_root, split, cls)
            seqs = [d for d in os.listdir(path) if d.startswith('seq_')]
            print(f"{path}: {len(seqs)} sequences")