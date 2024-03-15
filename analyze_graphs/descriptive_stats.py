import marimo

__generated_with = "0.3.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.style.use("seaborn-v0_8")
    return mo, np, pd, plt, stats


@app.cell
def __():
    from sklearn.preprocessing import (
        PowerTransformer,
        StandardScaler,
        RobustScaler,
    )
    from sklearn.covariance import GraphicalLasso
    from sklearn.model_selection import ValidationCurveDisplay
    from sklearn.pipeline import make_pipeline
    return (
        GraphicalLasso,
        PowerTransformer,
        RobustScaler,
        StandardScaler,
        ValidationCurveDisplay,
        make_pipeline,
    )


@app.cell
def __(pd):
    adni_low = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_low_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"])

    adni_med_low = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_med_low_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"])

    adni_med_high = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_med_high_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"])

    adni_high = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_high_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"])

    a4_low = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/a4_quartiles/a4_amy_tau_merged_cent_low_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"]).drop(columns=['quartile','CEREBELLUM_CORTEX'])

    a4_med_low = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/a4_quartiles/a4_amy_tau_merged_cent_med_low_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"]).drop(columns=['quartile','CEREBELLUM_CORTEX'])

    a4_med_high = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/a4_quartiles/a4_amy_tau_merged_cent_med_high_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"]).drop(columns=['quartile','CEREBELLUM_CORTEX'])

    a4_high = pd.read_csv(
        "../data_paths_and_cleaning/data/final_cleaned_quartiles/a4_quartiles/a4_adni_amy_tau_merged_cent_high_quartile.csv"
    ).drop(columns=["RID", "CENTILOIDS"]).drop(columns=['quartile','CEREBELLUM_CORTEX'])
    return (
        a4_high,
        a4_low,
        a4_med_high,
        a4_med_low,
        adni_high,
        adni_low,
        adni_med_high,
        adni_med_low,
    )


@app.cell
def __(a4_low, adni_low, plt, stats):
    fig, axs = plt.subplots(
        len(a4_low.columns) // 4 + 1, 4, figsize=(20, 30), sharex=True
    )

    for i, region in enumerate(adni_low.columns):

        _data = adni_low[region].sort_values().to_frame()

        # stats.probplot(PowerTransformer().fit_transform(_data).squeeze(),plot=axs.flat[i],rvalue=True)
        stats.probplot(_data.squeeze(), plot=axs.flat[i], rvalue=True)
        axs.flat[i].set_title(region)
        axs.flat[i].set_xlabel("")

    plt.tight_layout()
    plt.show()
    return axs, fig, i, region


@app.cell
def __(PowerTransformer, adni_low, plt):
    transformed = (
        PowerTransformer(method="box-cox")
        .set_output(transform="pandas")
        .fit_transform(adni_low)
    )
    transformed.hist(figsize=(16, 16))
    plt.show()
    return transformed,


@app.cell
def __(
    GraphicalLasso,
    PowerTransformer,
    a4_high,
    a4_low,
    a4_med_high,
    a4_med_low,
    adni_high,
    adni_low,
    adni_med_high,
    adni_med_low,
    make_pipeline,
    pd,
):
    data_scaled = pd.concat([adni_low, adni_med_low, adni_med_high, adni_high, a4_low, a4_med_low, a4_med_high, a4_high]).drop(columns='CEREBELLUM_CORTEX')
    # data_scaled = pd.concat([adni_low, adni_med_low, adni_med_high, adni_high]).drop(columns='CEREBELLUM_CORTEX')

    model = make_pipeline(
        PowerTransformer(),
        # RobustScaler(),
        GraphicalLasso(
            alpha=0.1,
            max_iter=1000,
            tol=1e-3,
            enet_tol=1e-7,
            mode="cd",
            verbose=False,
        ),  # , eps=1e-21, assume_centered=True)
    )
    return data_scaled, model


@app.cell
def __(data_scaled, mo, model, np):
    # generating a sequence of 128 alpha values evenly spaces between .05 and 1
    # testing out different strengths of regularization parameter alpha for the graphical lasso model
    # alphas = np.linspace(0.0106,1.,64)
    alphas = np.linspace(0.10, 0.16, 16)

    # initializing lists to store precision matrix, covariance matrix, and log-liklihoods
    # looking at each metric for model fits with different alpha values
    precision_nonzero = []
    covariance_nonzero = []
    log_likelihoods = []

    n_features = data_scaled.shape[1]
    n_samples = data_scaled.shape[0]

    bootstrap_samples = 256

    # iterate through each alpha value
    for alpha in mo.status.progress_bar(alphas):

        model.named_steps["graphicallasso"].alpha = alpha

        log_likelihoods_boot = []
        precision_nonzero_boot = []
        covariance_nonzero_boot = []

        for _i in range(bootstrap_samples): 
            model.fit(data_scaled.sample(frac=1,replace=True))

            log_likelihoods_boot.append(model.score(data_scaled))
            precision_nonzero_boot.append(np.count_nonzero(model.named_steps["graphicallasso"].precision_))
            covariance_nonzero_boot.append(np.count_nonzero(model.named_steps["graphicallasso"].covariance_))

        log_likelihoods.append(log_likelihoods_boot)
        precision_nonzero.append(precision_nonzero_boot)
        covariance_nonzero.append(covariance_nonzero_boot)

    log_likelihoods = np.array(log_likelihoods)  #
    precision_nonzero = np.array(
        precision_nonzero
    )  # number nonzero entries in precision matrix
    covariance_nonzero = np.array(
        covariance_nonzero
    )  # number nonzero entries in covariance matrix
    return (
        alpha,
        alphas,
        bootstrap_samples,
        covariance_nonzero,
        covariance_nonzero_boot,
        log_likelihoods,
        log_likelihoods_boot,
        n_features,
        n_samples,
        precision_nonzero,
        precision_nonzero_boot,
    )


@app.cell
def __(alphas, covariance_nonzero, n_features, np, plt, precision_nonzero):
    plt.figure(figsize=(3.5, 3.5))

    plt.plot(alphas, np.median(covariance_nonzero,axis=1) / n_features**2, label="Covariance",color='C0')

    plt.fill_between(alphas, 
                     np.quantile(covariance_nonzero,0.025,axis=1) / n_features**2,
                     np.quantile(covariance_nonzero,0.975,axis=1) / n_features**2,
                     color='C0',alpha=0.4)

    plt.plot(alphas, np.median(precision_nonzero,axis=1) / n_features**2, label="Precision",color='C1')

    plt.fill_between(alphas, 
                     np.quantile(precision_nonzero,0.025,axis=1) / n_features**2,
                     np.quantile(precision_nonzero,0.975,axis=1) / n_features**2,
                     color='C1',alpha=0.4)


    plt.ylabel("Fraction of nonzero entries")
    plt.axhline(
        1 / n_features, color="k", linestyle="--", label=r"$1/n_\text{regions}$", linewidth=0.5
    )
    plt.xlabel(r"$L_1$ Regularization parameter $\alpha$")
    plt.tight_layout()
    plt.xscale("linear")
    plt.yscale("linear")
    plt.legend(loc=(0.1,0.5))
    # plt.savefig('nonzero_frac_L1.pdf')
    plt.ylim(0.26,0.3)
    plt.show()
    return


@app.cell
def __():
    # ValidationCurveDisplay.from_estimator(model,
    #                                      X= data_scaled,
    #                                       y=None,
    #                                      param_name='graphicallasso__alpha',
    #                                      param_range=np.geomspace(0.2,1,8),
    #                                     cv=5,
    #                                      n_jobs=8)

    # plt.show()
    return


@app.cell
def __(
    alphas,
    log_likelihoods,
    n_features,
    n_samples,
    np,
    plt,
    precision_nonzero,
):
    bics = -2 * log_likelihoods + (
        (precision_nonzero - n_features) / 2 + n_features + n_features
    ) * np.log(n_samples)
    # bics = - 2 * log_likelihoods +  precision_nonzero * np.log(n_samples)
    aics = (
        -2 * log_likelihoods
        + ((precision_nonzero - n_features) / 2 + 2 * n_features) * 2
    )

    _fig, _axs = plt.subplots(1, 1, figsize=(3.5, 3.5))

    _axs.plot(alphas, np.median(bics,axis=1))

    _axs.fill_between(alphas, 
                      np.quantile(bics,0.025,axis=1),
                      np.quantile(bics,0.975,axis=1),
                      alpha=0.4
                     )
    # _axs[0].plot(alphas,aics,label='AIC')
    # _axs.legend()
    # _axs[1].plot(alphas, log_likelihoods, label="Log likelihood")
    _axs.set_ylabel("BIC")
    # _axs[1].set_ylabel("Log likelihood")
    # plt.axvline(alpha,color='k',linestyle='--',label=r"$\alpha="+str(alpha)+"$")
    _axs.set_xlabel(r"$L_1$ Regularization parameter $\alpha$")
    # _axs[1].set_xlabel(r"$L_1$ Regularization parameter")
    _fig.tight_layout()
    # _axs.set_xlim(left=0,right=0.3)
    # _axs.set_ylim(bottom=1700,top=3000)
    _axs.set_xscale('linear')
    plt.savefig('BIC_L1.pdf')
    plt.show()
    return aics, bics


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
