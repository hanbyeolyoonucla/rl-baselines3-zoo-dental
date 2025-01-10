
import numpy as np
import nibabel as nib

def bbox_3d(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # return rmin, rmax, cmin, cmax, zmin, zmax
    return -rmin+rmax, -cmin+cmax, -zmin+zmax

if __name__ == "__main__":
    tnum = 5
    scale, rz, ry, rx, tx, ty, tz = 1.0, 0, 0, 0, 0, 0, 0
    data = np.load(f'dental_env/labels_augmented/tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}.npy')
    print(np.array(bbox_3d(data==2))*0.340)
    print(np.array(bbox_3d(data))*0.340)
