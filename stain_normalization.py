import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--level', type=str, default='patch', choices=['patch', 'slide'])
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def normalize_all_slides(source_dir, target_dir, level='patch', num_samples=1000, seed=0):
    """
    Applies stain normalization to all slides in a directory, and saves the resulting slides in a new directory.

    Args:
        source_dir: Path to a directory that contains a collection slide directories.
        target_dir: Path to a directory where normalized slides shall be stored.
        level: Whether to normalize on patch or slide level.
            - 'patch': For each patch, an individual stain vector is computed, and transformed into the target stain
                vector.
            - 'slide': For each slide, a single stain vector is computed based on a number of sampled patches. All
                patches of this slide are then transformed based on this stain vector.
        num_samples: Number of patches to sample to compute a slide stain vector (only for level 'slide').
        seed: Seed for numpy, e.g. for sampling mechanism for level 'slide'.
    """
    # Set up data structures
    np.random.seed(seed)
    normalizer = NumpyMacenkoNormalizer()

    print(f"Applying stain normalization on {level} level for {len(os.listdir(source_dir))} slides.")

    # Normalize all slides according to the given methods
    for slide_id in tqdm(os.listdir(source_dir)):
        slide_source_path = os.path.join(source_dir, slide_id)
        if not os.path.isdir(slide_source_path) or \
                len([patch for patch in os.listdir(slide_source_path) if patch.endswith('.tif')]) == 0:
            continue
        slide_target_path = os.path.join(target_dir, slide_id)
        if level == 'slide':
            normalize_slide(slide_source_path, slide_target_path, normalizer, num_samples)
        elif level == 'patch':
            normalize_patches(slide_source_path, slide_target_path, normalizer)
        else:
            raise ValueError(f"Unknown level {level}")


def normalize_slide(source_slide_dir, target_slide_dir, normalizer, num_samples):

    # Set up data structures
    patch_files = [patch for patch in os.listdir(source_slide_dir) if patch.endswith('.jpg')]
    num_samples = len(patch_files) if num_samples == -1 else num_samples

    # Choose the samples to compute the stain vectors from
    if len(patch_files) <= num_samples:
        sample_files = patch_files
    else:
        sample_files = np.random.choice(patch_files, size=num_samples, replace=False)

    # Compute the stain vectors from the samples
    samples = [np.asarray(Image.open(os.path.join(source_slide_dir, sample_file))) for sample_file in sample_files]
    HE, maxC = normalizer.compute_stain_vectors(samples)

    # Normalize all slide patches with the same stain vectors
    os.makedirs(target_slide_dir, exist_ok=True)
    for patch_file in patch_files:
        patch = np.asarray(Image.open(os.path.join(source_slide_dir, patch_file)))
        norm_patch, _, _ = normalizer.normalize_from_stain_vectors(patch, HE, maxC, stains=False)
        norm_patch_path = os.path.join(target_slide_dir, patch_file)
        if os.path.exists(norm_patch_path):
            print(f"Normalized patch {norm_patch_path} already exists.")
        else:
            Image.fromarray(norm_patch).save(norm_patch_path)


def normalize_patches(source_slide_dir, target_slide_dir, normalizer):

    # Set up data structures
    patch_files = [patch for patch in os.listdir(source_slide_dir) if patch.endswith('.tif')]

    # Normalize all slide patches
    os.makedirs(target_slide_dir, exist_ok=True)
    for patch_file in patch_files:
        patch = np.asarray(Image.open(os.path.join(source_slide_dir, patch_file)))
        norm_patch, _, _ = normalizer.normalize(patch, stains=False)
        norm_patch_path = os.path.join(target_slide_dir, patch_file)
        if os.path.exists(norm_patch_path):
            print(f"Normalized patch {norm_patch_path} already exists.")
        else:
            Image.fromarray(norm_patch).save(norm_patch_path)


class NumpyMacenkoNormalizer:
    """
    Adaptation of https://github.com/EIDOSLAB/torchstain/blob/main/torchstain/numpy/normalizers/macenko.py
    """

    def __init__(self):
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(np.float)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Normalize staining appearence of H&E stained images
        Example use:
            see test.py
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity
        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image
        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)


        H, E = None, None

        if stains:
            # unmix hematoxylin and eosin
            H = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
            H[H > 255] = 255
            H = np.reshape(H.T, (h, w, c)).astype(np.uint8)

            E = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
            E[E > 255] = 255
            E = np.reshape(E.T, (h, w, c)).astype(np.uint8)

        return Inorm, H, E

    def compute_stain_vectors(self, samples, Io=240, alpha=1, beta=0.15):

        I = np.stack(samples).reshape((-1, 3))

        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        return HE, maxC

    def normalize_from_stain_vectors(self, I, HE, maxC, Io=240, beta=0.15, stains=True):

        h, w, c = I.shape
        I = I.reshape((-1, 3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        C = self.__find_concentration(OD, HE)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)

        H, E = None, None

        if stains:
            # unmix hematoxylin and eosin
            H = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
            H[H > 255] = 255
            H = np.reshape(H.T, (h, w, c)).astype(np.uint8)

            E = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
            E[E > 255] = 255
            E = np.reshape(E.T, (h, w, c)).astype(np.uint8)

        return Inorm, H, E


if __name__ == '__main__':
    args = get_args()
    normalize_all_slides(args.source_dir, args.target_dir, args.level, args.num_samples, args.seed)
