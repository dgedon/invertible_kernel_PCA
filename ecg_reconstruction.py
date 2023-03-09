import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA, PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import RBFSampler

if __name__ == "__main__":
    """
    This script visualizes reconstructions for different methods on my CPSC beats dataset
    """
    ############################
    # setup
    ############################
    i_ecg, i_lead = 4, 0  # {1: [0], 4: [0], 18: [1]}
    train_ratio = 0.7

    seed = 42
    n_runs = 200  # increase for better uncertainty estimates

    eval_fun = lambda x_, x_hat_: np.mean((x_ - x_hat_) ** 2)  # MSE

    # ikPCA parameters (optimal)
    rff_components = 512
    ikpca_components = 1
    gamma_ikpca = 5e-5
    alpha_ikpca = 10

    # kPCA parameters (optimal)
    kpca_components = 1
    gamma_kpca = 10
    alpha_kpca = 15

    # PCA parameters (optimal)
    pca_components = 1

    ############################
    # do stuff
    ############################
    mse_test_pca = np.zeros(n_runs)
    mse_test_kpca = np.zeros(n_runs)
    mse_test_ikpca = np.zeros(n_runs)

    for i_runs in tqdm(range(n_runs)):
        # get data
        with h5py.File(os.path.join(os.getcwd(), 'ecg_data', 'cpsc_normal_beats.h5'), "r") as f:
            data_noisy = f['ecg{}_lead{}'.format(i_ecg, i_lead)][:].T
        n_data = data_noisy.shape[0]
        x_train_noisy, x_test_noisy = train_test_split(data_noisy, random_state=seed + i_runs,
                                                       train_size=train_ratio)
        x_test = x_test_noisy.copy()
        n_features = x_train_noisy.shape[1]
        # mean beat as true data
        x_train_mean = np.mean(x_train_noisy, axis=0)
        x_test_mean = np.mean(x_test_noisy, axis=0)
        # center data
        x_train_noisy -= x_train_mean
        x_test_noisy -= x_train_mean

        ############################
        # PCA
        ############################
        # definition
        pca = PCA(n_components=pca_components)
        # fit
        pca.fit(x_train_noisy)
        # test
        x_test_reconstructed_pca = pca.inverse_transform(pca.transform(x_test_noisy)) + x_train_mean
        # eval
        mse_test_pca[i_runs] = eval_fun(x_test_mean, x_test_reconstructed_pca)

        ############################
        # kPCA
        ############################
        # definition
        kpca = KernelPCA(n_components=kpca_components,
                         kernel='rbf',
                         gamma=gamma_kpca,
                         alpha=alpha_kpca,
                         fit_inverse_transform=True, )
        # fit
        kpca.fit(x_train_noisy)
        # test
        x_test_reconstructed_kpca = kpca.inverse_transform(kpca.transform(x_test_noisy)) + x_train_mean
        # eval
        mse_test_kpca[i_runs] = eval_fun(x_test_mean, x_test_reconstructed_kpca)

        ############################
        # ikPCA
        ############################
        # define ikPCA
        pca = PCA(n_components=ikpca_components)
        sampler = RBFSampler(n_features=n_features,
                             n_components=rff_components,
                             gamma=gamma_ikpca,
                             regularization=alpha_ikpca)
        # fit
        pca.fit(sampler.transform(x_train_noisy)[0])
        # test
        x_test_rbf, info = sampler.transform(x_test_noisy)
        x_test_reconstructed_rbf = pca.inverse_transform(pca.transform(x_test_rbf))
        x_test_reconstructed_ikpca = sampler.invert_transform(x_test_reconstructed_rbf, info) + x_train_mean
        # eval
        mse_test_ikpca[i_runs] = eval_fun(x_test_mean, x_test_reconstructed_ikpca)

    ############################
    # plot reconstructions
    ############################
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # ax1: PCA
    ax1.plot(np.mean(x_test, axis=0))
    ax1.fill_between(np.arange(n_features), np.min(x_test, axis=0), np.max(x_test, axis=0), alpha=0.5)
    ax1.plot(x_test_reconstructed_pca.T, color='r', alpha=0.3, linestyle='dashed')
    ax1.set_title(r'PCA, mse={:.2e}$\pm${:.2e}'.format(np.mean(mse_test_pca), np.std(mse_test_pca)))
    ax1.legend(['test mean', 'test (min/max)', 'reconstruction'])

    # ax2: kPCA
    ax2.plot(np.mean(x_test, axis=0))
    ax2.fill_between(np.arange(n_features), np.min(x_test, axis=0), np.max(x_test, axis=0), alpha=0.5)
    ax2.plot(x_test_reconstructed_kpca.T, color='r', alpha=0.3, linestyle='dashed')
    ax2.set_title(r'kPCA+SL, mse={:.2e}$\pm${:.2e}'.format(np.mean(mse_test_kpca), np.std(mse_test_kpca)))

    # ax3: ikPCA
    ax3.plot(np.mean(x_test, axis=0))
    ax3.fill_between(np.arange(n_features), np.min(x_test, axis=0), np.max(x_test, axis=0), alpha=0.5)
    ax3.plot(x_test_reconstructed_ikpca.T, color='r', alpha=0.3, linestyle='dashed')
    ax3.set_title(r'ikPCA, mse={:.2e}$\pm${:.2e}'.format(np.mean(mse_test_ikpca), np.std(mse_test_ikpca)))

    fig.suptitle('De-noising of ECG beats')
    fig.tight_layout()
    plt.show()
