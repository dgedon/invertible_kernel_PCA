import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA, PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import RBFSampler


if __name__ == "__main__":
    """
    This script compares the performance of iKPCA for different number of RFF components with PCA, KPCA on USPS.
    We average the results over multiple runs.
    """
    ############################
    # setup
    ############################
    seed = 42

    # data set size
    n_data_train = 2_000
    n_data_test = 2_000
    n_data = n_data_train + n_data_test
    d_data = 3

    # number of monte carlo runs
    n_runs = 1  # increase for error bars

    # fixed parameters for evaluation criterion
    eval_fun = lambda x_, x_hat_: np.mean((x_ - x_hat_) ** 2)  # MSE
    noise = 0.25
    pca_components_list = np.linspace(2, 20, 10, dtype=int)

    # ikPCA parameters
    alpha_ikpca = 1e0
    gamma_ikpca = 5e-1
    rff_ikpca_list = [50, 500, 5000]

    # kPCA parameters (optimal)
    gamma_kpca = 1e0
    alpha_kpca = 1e0

    # allocation
    mse_ikpca_list = np.zeros([n_runs, len(pca_components_list), len(rff_ikpca_list)])
    mse_kpca_list = np.zeros([n_runs, len(pca_components_list)])

    ############################
    # Loops
    ############################
    for i_run in range(n_runs):
        print(f"Run {i_run + 1}/{n_runs}.")

        x, t = make_s_curve(n_data, noise=0.0)
        n_features = d_data
        # split data
        x_train, x_test, y_train, y_test = train_test_split(x, t, random_state=seed + i_run,
                                                            train_size=n_data_train,
                                                            test_size=n_data_test)
        # add noise
        x_train_noisy = x_train + np.random.normal(0, noise, size=x_train.shape)
        x_test_noisy = x_test + np.random.normal(0, noise, size=x_test.shape)
        # center data
        x_train_noisy_mean = np.mean(x_train_noisy, axis=0)
        x_train_noisy -= x_train_noisy_mean
        x_test_noisy -= x_train_noisy_mean

        # pca_components loop
        for i_pca_components, pca_components in tqdm(enumerate(pca_components_list), total=len(pca_components_list)):
            ############################
            # kPCA
            ############################
            # definition
            kpca = KernelPCA(n_components=pca_components,
                             kernel='rbf',
                             gamma=gamma_kpca,
                             alpha=alpha_kpca,
                             fit_inverse_transform=True, )
            # fit
            kpca.fit(x_train_noisy)
            # test
            x_test_reconstructed_kpca = kpca.inverse_transform(
                kpca.transform(x_test_noisy)) + x_train_noisy_mean
            # evaluate
            mse_kpca_list[i_run, i_pca_components] = eval_fun(x_test, x_test_reconstructed_kpca)

            ############################
            # ikPCA
            ############################
            for i_rff, rff_ikpca in enumerate(rff_ikpca_list):
                # define ikPCA
                pca = PCA(n_components=min(rff_ikpca, pca_components))
                sampler = RBFSampler(n_features=n_features,
                                     n_components=rff_ikpca,
                                     gamma=gamma_ikpca,
                                     regularization=alpha_ikpca)
                # fit
                pca.fit(sampler.transform(x_train_noisy)[0])
                # test
                x_test_rbf, info = sampler.transform(x_test_noisy)
                x_test_reconstructed_rbf = pca.inverse_transform(pca.transform(x_test_rbf))
                x_test_reconstructed_ikpca = sampler.invert_transform(x_test_reconstructed_rbf,
                                                                      info) + x_train_noisy_mean
                # evaluate
                mse_ikpca_list[i_run, i_pca_components, i_rff] = eval_fun(x_test, x_test_reconstructed_ikpca)

    ############################
    # plot results
    ############################
    # plot mse
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i_rff, rff_ikpca in enumerate(rff_ikpca_list):
        # plot error bars
        ax.errorbar(pca_components_list, np.mean(mse_ikpca_list[:, :, i_rff], axis=0),
                    yerr=np.std(mse_ikpca_list[:, :, i_rff], axis=0),
                    label=r'$d_R={}$'.format(rff_ikpca))
    ax.errorbar(pca_components_list, np.mean(mse_kpca_list, axis=0),
                yerr=np.std(mse_kpca_list, axis=0),
                linestyle='dotted',
                color='k',
                label='kPCA+SL')
    ax.set_xlabel('PCA components')
    ax.set_ylabel('MSE')
    ax.set_title(r'S-curve data with noise $\sigma$={} and $n$={} samples'.format(noise, n_data_train))
    ax.set_ylim(bottom=0.045, top=0.08)
    lgd = ax.legend()
    plt.show()
