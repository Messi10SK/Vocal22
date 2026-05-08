import numpy as np
import glob
import os

def process_dataset(dstype):

    print(f"\n🔥 Processing {dstype} data...")

    paths = sorted(
        glob.glob(f'./processed_data/{dstype}/*.npz')
    )

    # ============================================
    # FIRST PASS: COUNT TOTAL SAMPLES
    # ============================================

    total_samples = 0

    for path in paths:
        data = np.load(path)
        total_samples += data['mix_mag'].shape[0]

    print(f"\n✅ Total samples: {total_samples}")

    # ============================================
    # CREATE MEMMAP FILES
    # ============================================

    mix_mags = np.memmap(
        f'./processed_data/mix_mags_{dstype}_512x128.npy',
        dtype='float32',
        mode='w+',
        shape=(total_samples, 512, 128, 1)
    )

    mix_phases = np.memmap(
        f'./processed_data/mix_phases_{dstype}_512x128.npy',
        dtype='complex64',
        mode='w+',
        shape=(total_samples, 512, 128, 1)
    )

    vocal_mags = np.memmap(
        f'./processed_data/vocal_mags_{dstype}_512x128.npy',
        dtype='float32',
        mode='w+',
        shape=(total_samples, 512, 128, 1)
    )

    # ============================================
    # SECOND PASS: WRITE IN CHUNKS
    # ============================================

    current_idx = 0

    for i, path in enumerate(paths):

        print(f'Processing {i+1}/{len(paths)}')

        data = np.load(path)

        mix_mag = data['mix_mag'][:,:,:,np.newaxis].astype(np.float32)
        mix_phase = data['mix_phase'][:,:,:,np.newaxis].astype(np.complex64)
        vocal_mag = data['vocal_mag'][:,:,:,np.newaxis].astype(np.float32)

        batch_size = mix_mag.shape[0]

        mix_mags[current_idx:current_idx+batch_size] = mix_mag
        mix_phases[current_idx:current_idx+batch_size] = mix_phase
        vocal_mags[current_idx:current_idx+batch_size] = vocal_mag

        current_idx += batch_size

    # ============================================
    # FLUSH TO DISK
    # ============================================

    mix_mags.flush()
    mix_phases.flush()
    vocal_mags.flush()

    print(f"\n✅ {dstype} dataset saved successfully")

def main():

    process_dataset('train')
    process_dataset('test')

if __name__ == '__main__':
    main()