import marimo

__generated_with = "0.3.3"
app = marimo.App(width="full")


@app.cell
def __():
    import altair as alt
    return alt,


@app.cell
def __(mo):
    mo.md("# Load data")
    return


@app.cell
def __(pd):
    # Tau
    adni_high = pd.read_csv("../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_high_quartile.csv").drop(columns=["RID", "CEREBELLUM_CORTEX","CENTILOIDS"])
    adni_med_high = pd.read_csv("../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_med_high_quartile.csv").drop(columns=["RID", "CEREBELLUM_CORTEX","CENTILOIDS"])
    adni_med_low = pd.read_csv("../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_med_low_quartile.csv").drop(columns=["RID", "CEREBELLUM_CORTEX","CENTILOIDS"])
    adni_low = pd.read_csv("../data_paths_and_cleaning/data/final_cleaned_quartiles/adni_quartiles/adni_amy_tau_merged_cent_low_quartile.csv").drop(columns=["RID", "CEREBELLUM_CORTEX","CENTILOIDS"])

    adni_all =  pd.concat( [
        adni_high,
        adni_med_high,
        adni_med_low,
        adni_low
        ],ignore_index=True)

    data = adni_low

    # Demographics
    demo_a4 = pd.read_csv(
        "../data_paths_and_cleaning/data/demographic_csvs/A4/a4_filtered_demo.csv"
    )

    demo_adni = pd.read_csv(
        "../data_paths_and_cleaning/data/demographic_csvs/ADNI/adni_filtered_demo.csv"
    ).rename(columns={"AGE": "PTAGE", "PTRACCAT": "PTRACE"})

    demog = pd.concat([demo_adni, demo_a4], keys=["ADNI", "A4"]).reset_index(
        level=0, names="Dataset"
    )
    return (
        adni_all,
        adni_high,
        adni_low,
        adni_med_high,
        adni_med_low,
        data,
        demo_a4,
        demo_adni,
        demog,
    )


@app.cell
def __(data):
    data
    return


@app.cell
def __(mo):
    mo.md("# Correct tau scores for age and sex")
    return


@app.cell
def __(mo):
    mo.md("# Generate one graph")
    return


@app.cell
def __():
    params = {
        "alpha": 0.15,
        "max_iter": 1000,
        "tol": 1e-3,
        "mode": "cd",
        "eps": 1e-12,
        "enet_tol": 1e-7,
    }
    return params,


@app.cell
def __(mo):
    mo.md(
        """The partial correlation matrix has the same sparsity as the precision matrix, but is easier to interpret, it has the correlation coefficients between the residuals of each pair of variables once you regressed on all the other ones."""
    )
    return


@app.cell
def __(alt, pcorr):
    pcorr_tall = pcorr[(pcorr < 1) & (pcorr != 0)].unstack().dropna().reset_index().rename(columns={0:'Partial Correlation'})

    # _n_samp = 32
    # pcorr_std = pd.concat([partial_correlation(prec) for prec in bootstrap(adni_high, partial(compute_precision,params=params),n_samples=_n_samp)],keys=range(_n_samp)).groupby(level=1).std().sort_index(axis=1)

    # pcorr_std_tall = pcorr_std.unstack().dropna().reset_index().rename(columns={0:'Stddev'})

    # pcorr_with_std = pd.merge(pcorr_tall,pcorr_std_tall,on=['level_0','level_1'])

    # bar = alt.Chart(pcorr_with_std).mark_errorbar().encode(
    #     x=alt.X('level_0'),
    #     y=alt.Y('Partial Correlation'),
    #     yError='Stddev',
    #     tooltip=['level_1','Partial Correlation']
    # )

    # mo.ui.altair_chart(
    point = alt.Chart(pcorr_tall).mark_point().encode(
        x=alt.X('level_0').title(''),
        y=alt.Y('Partial Correlation'),
        tooltip=['level_1','Partial Correlation'],
    ).properties(width=2500)

    point
    # )
    return pcorr_tall, point


@app.cell
def __(precision, precision_to_graph):
    G = precision_to_graph(precision)

    G['POSTCENTRAL']
    return G,


@app.cell
def __(mo):
    mo.md('All the algorithms only loop over the edges that exist, so it makes sense to mask the partial correlations set to zero before converting to distances.')
    return


@app.cell
def __(adni_high, compute_precision, params, partial_correlation, px):
    precision, covariance = compute_precision(
        adni_high, params, return_covariance=True
    )

    pcorr = partial_correlation(precision)

    px.imshow(
        pcorr[pcorr <1].round(2),
        width=500,
        height=500,
        color_continuous_scale="PiYG",
        color_continuous_midpoint=0,
        title='Partial Correlation',
        text_auto=True
    )
    return covariance, pcorr, precision


@app.cell
def __(precision, precision_to_graph):
    graph = precision_to_graph(precision)
    return graph,


@app.cell
def __(mo):
    mo.md(
        "In Dyrba 2020 they use as adjacency matrix 1 - |R| where R is the partial correlation matrix"
    )
    return


@app.cell
def __(graph, nx, plt):
    nx.draw_circular(graph, with_labels=True)
    plt.show()
    return


@app.cell
def __(mo):
    mo.md("# Sparsity of partial correlation matrix with regularization parameter")
    return


@app.cell
def __(bootstrap, compute_precision, data, mo, np, partial, pd):
    alphas = np.linspace(0.05, 1, 8)

    _nz = []

    _df = data

    for alpha in mo.status.progress_bar(alphas):
        _params = {
            "alpha": alpha,
            "max_iter": 1000,
            "tol": 1e-3,
            "mode": "cd",
            "eps": 1e-12,
            "enet_tol": 1e-7,
        }

        _fun = partial(compute_precision, params=_params)

        # this is a list, one per bootstrap samples
        _nonzero_counts = np.array(
            list(
                map(np.count_nonzero, bootstrap(_df, _fun, n_samples=8))
            )  # about 60s with n = 128
        )

        _nz.append(
            {
                "alpha": alpha,
                "Median": np.median(_nonzero_counts) / len(_df.columns) ** 2,
                "CI_low": np.quantile(_nonzero_counts, 0.025)
                / len(_df.columns) ** 2,
                "CI_high": np.quantile(_nonzero_counts, 0.975)
                / len(_df.columns) ** 2,
            }
        )

    partial_corr_nz = pd.DataFrame(_nz)  # fraction of nonzero entries
    return alpha, alphas, partial_corr_nz


@app.cell
def __(partial_corr_nz, plt):
    partial_corr_nz.plot(x="alpha", y="Median")
    plt.fill_between(
        partial_corr_nz.alpha,
        partial_corr_nz["CI_low"],
        partial_corr_nz["CI_high"],
        alpha=0.3,
    )

    plt.title("Fraction of nonzero entries in the partial correlation matrix")
    return


@app.cell
def __():
    # Subsample correlation of some metric
    return


@app.cell(disabled=True)
def __(
    alphas,
    compute_metrics,
    compute_precision,
    data,
    metrics,
    mo,
    np,
    params,
    pd,
    precision_to_graph,
):
    finite_size = []

    for _alpha in mo.status.progress_bar(alphas):

        _params = {
            "alpha": _alpha,
            "max_iter": 1000,
            "tol": 1e-3,
            "mode": "cd",
            "eps": 1e-12,
            "enet_tol": 1e-7,
        }

        for _frac in np.linspace(0.2,1,16):
            for _ in range(8): # bootstrap samples
                _sample = data.sample(frac=_frac,replace=True)
                metrics_dict = compute_metrics(precision_to_graph(compute_precision(_sample,params)),metrics)
                metrics_dict['N'] = len(_sample)
                metrics_dict['alpha'] = _alpha
                finite_size.append(metrics_dict)

    finite_size = pd.DataFrame(finite_size)
    return finite_size, metrics_dict


@app.cell
def __(finite_size):
    finite_size
    return


@app.cell
def __(finite_size, sns):
    sns.lineplot(data=finite_size.groupby('N').mean(),x='N',y='Efficiency')
    # sns.lineplot(data=finite_size,x='N',y='Clustering Coefficient')
    return


@app.cell
def __():
    # add alpha selection sparsity
    return


@app.cell
def __():
    # bic
    return


@app.cell
def __():
    # make sensible function to draw graph
    # do not draw dots
    # make edges thicker for higher weights
    return


@app.cell
def __(mo):
    mo.md("By default the functions in networkx ignore edge weights. Should we threshold the edge weights? Or figure out some way of using the weights?")
    return


@app.cell
def __(compute_metrics, graph, nx):
    metrics = {
        "Efficiency": nx.global_efficiency,
        "Clustering Coefficient": nx.average_clustering,
    }

    compute_metrics(graph, metrics)
    return metrics,


@app.cell(disabled=True)
def __(boostrap_graph_metrics, metrics, params, reduced_data):
    boostrap_graph_metrics(reduced_data, params, metrics, n_samples=4)
    return


@app.cell
def __():
    # from graph to random graph
    return


@app.cell
def __():
    # put together bootstrapped results for different groups/datasets and make a single plot
    return


@app.cell
def __(mo):
    mo.md("# Utilities")
    return


@app.cell
def __(np):
    def partial_correlation(precision):
        """Compute the partial correlation from a given precision matrix"""

        diag = precision.values.diagonal()

        # The - sign is correct, but the diagonal should have 1 instead of -1,
        # so we fill it explicitly with ones. The formula on wikipedia only applies
        # to off-diagonal elements
        partial_correl = -precision.copy() / np.sqrt(diag[:, None] * diag[None, :])

        np.fill_diagonal(partial_correl.values, 1)

        return partial_correl
    return partial_correlation,


@app.cell
def __(np):
    def pcorr_to_distance(pcorr):
        """Compute the distance matrix associated to a given partial correlation
        matrix: disconnected nodes should stay disconnected, nodes with high
        correlation should be close to each other. We also drop connections associated with negative weights
        """

        return (pcorr > 0) * (1 - np.abs(pcorr))
    return pcorr_to_distance,


@app.cell
def __(
    GraphicalLasso,
    PowerTransformer,
    make_pipeline,
    partial_correlation,
    pd,
):
    def compute_precision(data, params, return_covariance=False):
        """Takes a dataframe and computes the precision matrix via sklearn.covariance.GraphicalLasso. The data is first transformed via PowerTransformer to ensure normality.

        Arguments:
        data: dataframe
        params: dictionary of paramters passed to GraphicaLasso.
        return_covariance: if True, also return the covariance matrix. By default only return precision
        """

        glasso = make_pipeline(PowerTransformer(), GraphicalLasso(**params))

        glasso.fit(data)

        if return_covariance:
            labels = glasso.feature_names_in_
            return (
                pd.DataFrame(
                    glasso.named_steps["graphicallasso"].precision_,
                    index=labels,
                    columns=labels,
                ),
                pd.DataFrame(
                    glasso.named_steps["graphicallasso"].covariance_,
                    index=labels,
                    columns=labels,
                ),
            )
        else:
            # GraphicalLasso does not support working natively with dataframes
            # we have to restore the column names by hand
            labels = glasso.feature_names_in_
            return pd.DataFrame(
                glasso.named_steps["graphicallasso"].precision_,
                index=labels,
                columns=labels,
            )


    def compute_partial_correlation(data, params):
        """Convenience function to compute the partial correlation matrix from data"""

        return partial_correlation(compute_precision(data, params))
    return compute_partial_correlation, compute_precision


@app.cell
def __(nx, partial_correlation, pcorr_to_distance):
    # from precision matrix to graph
    def precision_to_graph(precision):
        """Convert the provided precision matrix into a networkx graph.

        Arguments:
        precision: dataframe, column names will become node labels
        allow_self_connections: bool, if False zero out the diagonal"""

        # Adjacency matrix
        adj = pcorr_to_distance(partial_correlation(precision))

        graph = nx.from_pandas_adjacency(adj)

        return graph
    return precision_to_graph,


@app.cell
def __():
    def compute_metrics(graph, metrics):
        """Compute graph metrics for a networks graph.

        Arguments:
        graph: networkx graph
        metrics: dictionary of metric names and callables

        Returns:
        res: dictionary of metric names and metric values"""

        return {metric: metrics[metric](graph) for metric in metrics}
    return compute_metrics,


@app.cell
def __(
    compute_metrics,
    compute_precision,
    multiprocessing,
    partial,
    pd,
    precision_to_graph,
):
    def data_to_metrics(data, params, metrics):
        # Convenience function, wrapping all steps into one
        precision = compute_precision(data, params=params)
        graph = precision_to_graph(precision)
        return compute_metrics(graph, metrics)


    def boostrap_graph_metrics(data, params, metrics, n_samples=8):
        """Resample the datframe data, generate graph and compute metrics. Returns a dataframe with the bootstrapped metrics"""

        bootstrap_samples = [
            data.sample(frac=1, replace=True) for _ in range(n_samples)
        ]

        with multiprocessing.Pool() as pool:
            res = pool.map(
                partial(data_to_metrics, params=params, metrics=metrics),
                bootstrap_samples,
            )

        return pd.DataFrame(res)


    def bootstrap(data, func, n_samples=8):
        # Resample from data and apply fun to each sample
        # use this version for general functions rather than graph metrics

        bootstrap_samples = [
            data.sample(frac=1, replace=True) for _ in range(n_samples)
        ]

        with multiprocessing.Pool() as pool:
            res = pool.map(func, bootstrap_samples)

        return res
    return boostrap_graph_metrics, bootstrap, data_to_metrics


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import networkx as nx
    import multiprocessing
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

    pd.set_option('display.max_columns',100)
    return mo, multiprocessing, np, nx, pd, plt


@app.cell
def __():
    from sklearn.preprocessing import PowerTransformer
    from sklearn.covariance import GraphicalLasso
    from sklearn.pipeline import make_pipeline
    return GraphicalLasso, PowerTransformer, make_pipeline


@app.cell
def __():
    import plotly.express as px
    return px,


@app.cell
def __():
    from functools import partial
    return partial,


@app.cell
def __():
    import seaborn as sns
    return sns,


if __name__ == "__main__":
    app.run()
