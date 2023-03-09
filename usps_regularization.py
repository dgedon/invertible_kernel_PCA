import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA, PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import RBFSampler


if __name__ == "__main__":
    """
    This script compares the performance of iKPCA for different alpha with PCA, KPCA on USPS.
    We average the results over multiple runs.
    """
    ############################
    # setup
    ############################
    seed = 42

    # data set size
    n_data_train = 1_000
    n_data_test = 400

    # number of monte carlo runs
    n_runs = 1  # increase for error bars

    # fixed parameters for evaluation criterion
    eval_fun = lambda x_, x_hat_: np.mean((x_ - x_hat_) ** 2)  # MSE
    noise = 0.5
    pca_components_list = np.linspace(5, 25, 10, dtype=int)

    # ikPCA parameters
    rff_components = 30_000
    gamma_ikpca = 1e-4
    alpha_ikpca_list = [0.1, 0.5, 2.0]

    # kPCA parameters (optimal)
    gamma_kpca = 5e-3
    alpha_kpca = 1e-2

    # allocation
    mse_ikpca_list = np.zeros([n_runs, len(pca_components_list), len(alpha_ikpca_list)])
    mse_kpca_list = np.zeros([n_runs, len(pca_components_list)])
    mse_pca_list = np.zeros([n_runs, len(pca_components_list)])

    ############################
    # Loops
    ############################
    for i_run in range(n_runs):
        print(f'Run {i_run + 1}/{n_runs}')
        ############################
        # get data
        ############################
        x, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True, parser='auto')
        x = MinMaxScaler().fit_transform(x)
        x_train, x_test, _, _ = train_test_split(x, y, stratify=y, random_state=seed + i_run,
                                                 train_size=n_data_train,
                                                 test_size=n_data_test)
        n_features = x_train.shape[1]
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
            # PCA
            ############################
            # definition
            pca = PCA(n_components=pca_components)
            # fit
            pca.fit(x_train_noisy)
            # test
            x_test_reconstructed_pca = pca.inverse_transform(pca.transform(x_test_noisy)) + x_train_noisy_mean
            # evaluate
            mse_pca_list[i_run, i_pca_components] = eval_fun(x_test, x_test_reconstructed_pca)

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
            x_test_reconstructed_kpca = kpca.inverse_transform(kpca.transform(x_test_noisy)) + x_train_noisy_mean
            # evaluate
            mse_kpca_list[i_run, i_pca_components] = eval_fun(x_test, x_test_reconstructed_kpca)

            ############################
            # ikPCA
            ############################
            for i_alpha, alpha_ikpca in enumerate(alpha_ikpca_list):
                # define ikPCA
                pca = PCA(n_components=pca_components)
                sampler = RBFSampler(n_features=n_features,
                                     n_components=rff_components,
                                     gamma=gamma_ikpca,
                                     regularization=alpha_ikpca)
                # fit
                temp = sampler.transform(x_train_noisy)[0]
                pca.fit(temp)
                # test
                x_test_rbf, info = sampler.transform(x_test_noisy)
                x_test_reconstructed_rbf = pca.inverse_transform(pca.transform(x_test_rbf))
                x_test_reconstructed_ikpca = sampler.invert_transform(x_test_reconstructed_rbf,
                                                                      info) + x_train_noisy_mean
                # evaluate
                mse_ikpca_list[i_run, i_pca_components, i_alpha] = eval_fun(x_test, x_test_reconstructed_ikpca)

    ############################
    # plot results
    ############################
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i_alpha, alpha_ikpca in enumerate(alpha_ikpca_list):
        # plot error bars
        ax.errorbar(pca_components_list, np.mean(mse_ikpca_list[:, :, i_alpha], axis=0),
                    yerr=np.std(mse_ikpca_list[:, :, i_alpha], axis=0),
                    label=r'$\alpha={}$'.format(alpha_ikpca))
    ax.errorbar(pca_components_list, np.mean(mse_kpca_list, axis=0),
                yerr=np.std(mse_kpca_list, axis=0),
                linestyle='dotted',
                color='k',
                label='kPCA+SL')
    ax.errorbar(pca_components_list, np.mean(mse_pca_list, axis=0),
                yerr=np.std(mse_pca_list, axis=0),
                linestyle='dashed',
                color='k',
                label='PCA')
    ax.set_xlabel('PCA components')
    ax.set_ylabel('MSE')
    ax.set_title(r'USPS data with noise $\sigma$= {}'.format(noise))
    ax.legend()
    ax.set_ylim(top=0.05)
    plt.show()
