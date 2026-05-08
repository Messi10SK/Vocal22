import numpy as np
import glob

def main():

    for dstype in ['train', 'test']:

        print(f'\n🔥 Processing {dstype} data...')

        dstype_paths = sorted(
            glob.glob(f'./processed_data/{dstype}/*.npz')
        )

        mix_mags_dstype = []
        mix_phases_dstype = []
        vocal_mags_dstype = []

        for i, path in enumerate(dstype_paths):

            print(f'Processing {i+1}/{len(dstype_paths)}')

            data = np.load(path)

            # ============================================
            # USE SMALLER DATATYPES TO SAVE RAM
            # ============================================

            mix_mags_dstype.append(
                data['mix_mag'][:,:,:,np.newaxis].astype(np.float32)
            )

            mix_phases_dstype.append(
                data['mix_phase'][:,:,:,np.newaxis].astype(np.complex64)
            )

            vocal_mags_dstype.append(
                data['vocal_mag'][:,:,:,np.newaxis].astype(np.float32)
            )

        print('\n🔥 Concatenating arrays...')

        mix_mags_dstype = np.concatenate(mix_mags_dstype, axis=0)
        mix_phases_dstype = np.concatenate(mix_phases_dstype, axis=0)
        vocal_mags_dstype = np.concatenate(vocal_mags_dstype, axis=0)

        print('\n🔥 Saving datasets...')

        np.save(
            f'./processed_data/mix_mags_{dstype}_512x128.npy',
            mix_mags_dstype
        )

        np.save(
            f'./processed_data/mix_phases_{dstype}_512x128.npy',
            mix_phases_dstype
        )

        np.save(
            f'./processed_data/vocal_mags_{dstype}_512x128.npy',
            vocal_mags_dstype
        )

        print(f'\n✅ {dstype} dataset saved successfully')

if __name__ == '__main__':
    main()