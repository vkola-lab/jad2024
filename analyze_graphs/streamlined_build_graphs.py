import marimo

__generated_with = "0.4.7"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        """# Load data
    ## Tau SUVR
    ADNI and A4 use two slightly different normalization conventions for computing SUVR, to make them comparable we have to divide all the columns in ADNI by the values in the `cerebellum_cortex` column. We then drop that column, as it is identically 1.
    """
    )
    return


@app.cell
def __(pd):
    adni = pd.read_csv(
        "../data_paths_and_cleaning/data/intermediate_data/adni/merged_adni_at_amy_pos_bi_harm.csv",
        dtype={"RID": str},
    )

    # Normalize all regions by cerebellum cortex
    adni = pd.concat(
        (
            adni[["RID", "CENTILOIDS"]],
            adni.drop(columns=["RID", "CENTILOIDS"]).div(
                adni["CEREBELLUM_CORTEX"], axis=0
            ),
        ),
        axis=1,
    ).drop(columns="CEREBELLUM_CORTEX")

    a4 = pd.read_csv(
        "../data_paths_and_cleaning/data/intermediate_data/a4/merged_a4_at_amy_pos_bi_harm.csv",
        dtype={"RID": str},
    ).drop(columns="CEREBELLUM_CORTEX")
    return a4, adni


@app.cell(disabled=True)
def __(a4, adni):
    adni.to_csv("adni_normalized.csv", index=False)
    a4.to_csv("a4_normalized.csv", index=False)
    return


@app.cell
def __(mo):
    mo.md("## Demographics")
    return


@app.cell
def __(pd):
    demo_a4 = pd.read_csv(
        "../data_paths_and_cleaning/data/demographic_csvs/A4/a4_filtered_demo.csv",
        dtype={"RID": str},
    )

    demo_adni = pd.read_csv(
        "../data_paths_and_cleaning/data/demographic_csvs/ADNI/adni_filtered_demo.csv",
        dtype={"RID": str},
    )

    demog = pd.concat([demo_adni, demo_a4], keys=["ADNI", "A4"]).reset_index(
        level=0, names="Dataset"
    )

    demog
    return demo_a4, demo_adni, demog


@app.cell
def __(adni, demo_adni, pd):
    adni_with_demo = pd.merge(adni, demo_adni[["RID", "PTAGE"]], on="RID")
    return adni_with_demo,


@app.cell
def __(a4, demo_a4, pd):
    a4_with_demo = pd.merge(a4, demo_a4[["RID", "PTAGE"]], on="RID")
    return a4_with_demo,


@app.cell
def __(a4_with_demo):
    a4_with_demo.head(3)
    return


@app.cell
def __(adni_with_demo):
    adni_with_demo.head(3)
    return


@app.cell
def __(mo):
    mo.md(
        """# Generate one example graph
    We keep demographics (e.g. age) as a pseudoregion, so the partial correlations are controlled for age."""
    )
    return


@app.cell
def __(PowerTransformer):
    from sklearn.preprocessing import StandardScaler

    power = PowerTransformer().set_output(transform='pandas')
    return StandardScaler, power


@app.cell
def __(adni_with_demo):
    adni_with_demo.drop(columns=['RID','CENTILOIDS','PTAGE']).apply(['skew','kurtosis']).agg(['median','std',lambda x: x.quantile(0.25),lambda x:x.quantile(0.75)],axis=1)
    return


@app.cell
def __(adni_with_demo, power):
    power.fit_transform(adni_with_demo.drop(columns=['RID','CENTILOIDS','PTAGE'])).apply(['skew','kurtosis']).agg(['median','std',lambda x: x.quantile(0.25),lambda x:x.quantile(0.75)],axis=1)
    return


@app.cell
def __(a4_with_demo):
    a4_with_demo.drop(columns=['RID','CENTILOIDS','PTAGE']).apply(['skew','kurtosis']).agg(['median','std',lambda x: x.quantile(0.25),lambda x:x.quantile(0.75)],axis=1)
    return


@app.cell
def __(a4_with_demo, power):
    power.fit_transform(a4_with_demo.drop(columns=['RID','CENTILOIDS','PTAGE'])).apply(['skew','kurtosis']).agg(['median','std',lambda x: x.quantile(0.25),lambda x:x.quantile(0.75)],axis=1)
    return


@app.cell
def __(adni_with_demo):
    data = adni_with_demo[adni_with_demo["CENTILOIDS"] >= 94].drop(
        columns=["RID", "CENTILOIDS"]
    )
    return data,


@app.cell
def __(data):
    data
    return


@app.cell
def __(compute_precision, data, partial_correlation, precision_to_graph):
    _params = {
        "alpha": 0.15,
        "max_iter": 1000,
        "tol": 1e-3,
        "mode": "cd",
        "eps": 1e-12,
        "enet_tol": 1e-7,
    }

    precision, covariance = compute_precision(
        data, _params, return_covariance=True
    )

    pcorr = partial_correlation(precision)

    graph = precision_to_graph(precision)
    return covariance, graph, pcorr, precision


@app.cell
def __(mo):
    mo.md(
        """The partial correlation matrix has the same sparsity as the precision matrix, but is easier to interpret, it has the correlation coefficients between the residuals of each pair of variables once you regressed on all the other ones."""
    )
    return


@app.cell
def __(alt, mo, pcorr):
    pcorr_tall = (
        pcorr[(pcorr < 1) & (pcorr != 0)]
        .unstack()
        .dropna()
        .reset_index()
        .rename(columns={0: "Partial Correlation"})
    )

    mo.ui.altair_chart(
        alt.Chart(pcorr_tall)
        .mark_point()
        .encode(
            x=alt.X("level_0").title(""),
            y=alt.Y("Partial Correlation"),
            tooltip=["level_0", "level_1", "Partial Correlation"],
        )
    )
    return pcorr_tall,


@app.cell
def __(np, pcorr, px):
    _mask = np.triu(np.ones_like(pcorr, dtype=bool))
    _pcorr_masked = pcorr.copy()
    _pcorr_masked[_mask] = np.nan

    # _pcorr_masked = _pcorr_masked.replace(0,np.nan)

    _fig = px.imshow(
        _pcorr_masked.round(2),
        width=1000,
        height=1000,
        color_continuous_scale="PiYG",
        color_continuous_midpoint=0,
        title="Partial Correlation",
        text_auto=True,
    )


    _fig.update_layout(
        {
            "plot_bgcolor": "white",
        },
    )

    _fig.update_traces(textfont_size=7)
    return


@app.cell
def __(mo):
    mo.md("## Visualize graph")
    return


@app.cell
def __(graph, nx, plt):
    _edge_weights = [
        1.5 * (2.22 * graph[u][v]["abs(correlation)"]) ** 2
        for u, v in graph.edges()
    ]

    _fig, _ax = plt.subplots(1, 1, figsize=(20, 10))

    nx.set_edge_attributes(
        graph,
        {
            (u, v): {"plot_weight": 0.75 * (2.22 * d["abs(correlation)"]) ** 2}
            for u, v, d in graph.edges(data=True)
        },
    )

    pos = nx.spectral_layout(graph, weight="distance")
    # pos = nx.spring_layout(graph, pos=pos,weight="abs(correlation)", iterations=100)
    pos = nx.spring_layout(graph, pos=pos, weight="plot_weight", iterations=100)

    nx.draw_networkx_edges(graph, pos, width=_edge_weights, ax=_ax)

    nx.draw_networkx_labels(
        graph,
        pos,
        ax=_ax,
        font_size=8,
        bbox=dict(facecolor="white", alpha=1, edgecolor="white", pad=0),
    )

    # Draw edge labels
    # edge_labels = nx.get_edge_attributes(graph, 'correlation')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.axis("off")
    _fig.tight_layout()
    plt.show()
    return pos,


@app.cell
def __(mo):
    mo.md("# Sparsity of partial correlation matrix with regularization parameter")
    return


@app.cell(hide_code=True)
def __(bootstrap, compute_precision, data, mo, np, partial, pd):
    alphas = np.linspace(0.05, 1, 16)

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

    plt.xlabel(r"$L_1$ regularization $\alpha$")
    plt.ylabel("Partial correlation matrix nonzero fraction")
    return


@app.cell
def __(mo):
    mo.md("""# Finite Size Effects
    We can do a training curve or compute the correlation between metrics at subsample sizes
    """
    )
    return


@app.cell
def __(metrics):
    metrics
    return


@app.cell
def __(
    compute_metrics,
    compute_precision,
    data,
    metrics,
    mo,
    np,
    pd,
    precision_to_graph,
):
    _params = {
        "alpha": 0.15,
        "max_iter": 1000,
        "tol": 1e-3,
        "mode": "cd",
        "eps": 1e-12,
        "enet_tol": 1e-7,
    }

    finite_size = []

    _n_boot = 100

    for _frac in mo.status.progress_bar(np.linspace(0.4, 1, 16)):
            for _ in range(_n_boot):  # bootstrap samples
                _sample = data.sample(frac=_frac, replace=True)
                metrics_dict = compute_metrics(
                    precision_to_graph(compute_precision(_sample, _params)), metrics
                )

                metrics_dict["Sample Fraction"] = _frac
                metrics_dict["N"] = len(_sample)
                metrics_dict["alpha"] = _params['alpha']
                finite_size.append(metrics_dict)

    finite_size = pd.DataFrame(finite_size)
    return finite_size, metrics_dict


@app.cell
def __(finite_size):
    finite_size.head()
    return


@app.cell
def __(finite_size, plt, sns):
    _fig,_ax = plt.subplots(1,3,figsize=(10,3))

    sns.lineplot(data=finite_size,x='N',y='Weighted Clustering Coefficient',errorbar='sd',ax=_ax[0])
    sns.lineplot(data=finite_size,x='N',y='Weighted Avg. Shortest Path Length',errorbar='sd',ax=_ax[1])
    sns.lineplot(data=finite_size,x='N',y='Weighted Small World',errorbar='sd',ax=_ax[2])

    # sns.boxplot(data=finite_size, x="N", y="Clustering",native_scale=True)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def __():
    # finite_size.to_csv("adni_high_subsample_100boot.csv",index=False)
    return


@app.cell
def __(mo):
    mo.md("# Population level metrics")
    return


@app.cell
def __(a4, adni, demog, pd, plt, sns):
    # px.histogram(demog,x='',color='Dataset')
    data_all = (
        pd.concat([adni, a4], keys=["ADNI", "A4"])
        .reset_index(level=0)
        .rename(columns={"level_0": "Dataset"})
    ).merge(demog[["RID", "PTAGE"]], on="RID")

    sns.histplot(
        data_all,
        x="CENTILOIDS",
        hue="Dataset",
        alpha=0.3,
        common_norm=False,
        # stat="density",
        cumulative=False,
    )
    plt.xlabel("Centiloids")
    return data_all,


@app.cell
def __(mo):
    mo.md("## Amyloid Quantiles")
    return


@app.cell
def __(adni_with_demo, pd):
    n_quantiles = 3

    adni_quantile_labels, adni_amy_bins = pd.qcut(
        adni_with_demo["CENTILOIDS"], q=n_quantiles, retbins=True, labels=False
    )

    adni_amy_bins
    return adni_amy_bins, adni_quantile_labels, n_quantiles


@app.cell
def __(a4):
    a4[
        "CENTILOIDS"
    ].max()  # this should be less than the adni maximum, which it is
    return


@app.cell
def __(a4, adni_amy_bins, pd):
    # cut a4 using the adni quantiles
    a4_quantile_labels = pd.cut(a4["CENTILOIDS"], adni_amy_bins, labels=False)
    return a4_quantile_labels,


@app.cell
def __(a4_quantile_labels, adni_quantile_labels, pd):
    pd.concat(
        (a4_quantile_labels, adni_quantile_labels), keys=("A4", "ADNI")
    ).reset_index(level=0).rename(
        columns={"level_0": "Dataset", "CENTILOIDS": "Centiloid quantile"}
    ).value_counts().sort_index()
    return


@app.cell
def __(mo):
    mo.md(
        """# Graph Metrics
    Note that by default many `nx` functions do not keep into accont edge weights
    """
    )
    return


@app.cell
def __(nx, partial, small_world_coeff):
    metrics = {
        # "Efficiency": nx.global_efficiency,  # does not keep into account edge weights
        # "Clustering Coefficient": partial(nx.average_clustering, weight=None),
        # "Avg. Shortest Path Length": partial(
        # nx.average_shortest_path_length, weight=None
        # ),
        # "Small World": partial(unweighted_small_world_coeff, niter=1, nrand=10),
        "Weighted Clustering Coefficient": partial(
            nx.average_clustering, weight="correlation"
        ),
        "Weighted Avg. Shortest Path Length": partial(
            nx.average_shortest_path_length, weight="distance"
        ),
        "Weighted Small World": partial(small_world_coeff, niter=1, nrand=10),
    }
    return metrics,


@app.cell
def __():
    # average weighted degree
    # betweenness
    return


@app.cell
def __(
    a4_quantile_labels,
    a4_with_demo,
    adni_quantile_labels,
    adni_with_demo,
    boostrap_graph_metrics,
    metrics,
    mo,
    n_quantiles,
    pd,
):
    _params = {
        "alpha": 0.15,
        "max_iter": 1000,
        "tol": 1e-3,
        "mode": "cd",
        "eps": 1e-12,
        "enet_tol": 1e-7,
    }

    _n_boot = 3

    adni_boot_metrics_results = []
    a4_boot_metrics_results = []

    for quantile in mo.status.progress_bar(range(n_quantiles)):
        adni_boot_metrics_results.append(
            boostrap_graph_metrics(
                adni_with_demo[adni_quantile_labels == quantile]
                .drop(columns=["RID", "CENTILOIDS"])
                .dropna(),
                _params,
                metrics,
                n_samples=_n_boot,
                randomize_graph=False,
            )
        )

    for quantile in mo.status.progress_bar(range(n_quantiles)):
        a4_boot_metrics_results.append(
            boostrap_graph_metrics(
                a4_with_demo[a4_quantile_labels == quantile]
                .drop(columns=["RID", "CENTILOIDS"])
                .dropna(),
                _params,
                metrics,
                n_samples=_n_boot,
                randomize_graph=False,
            )
        )

    graph_metrics_by_quantile = (
        pd.concat(
            [
                pd.concat(adni_boot_metrics_results, keys=range(n_quantiles))
                .reset_index(level=0)
                .rename(columns={"level_0": "Centiloid Quantile"}),
                pd.concat(a4_boot_metrics_results, keys=range(n_quantiles))
                .reset_index(level=0)
                .rename(columns={"level_0": "Centiloid Quantile"}),
            ],
            keys=["ADNI", "A4"],
        )
        .reset_index(level=0)
        .rename(columns={"level_0": "Dataset"})
    )
    return (
        a4_boot_metrics_results,
        adni_boot_metrics_results,
        graph_metrics_by_quantile,
        quantile,
    )


@app.cell
def __():
    # graph_metrics_by_quantile.to_csv('graph_metrics_adni_a4_bootstrapped_3quant.csv',index=False)
    return


@app.cell
def __(graph_metrics_by_quantile, metrics, plt, sns):
    _fig, _ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    for _i, _metric in enumerate(metrics):
        sns.boxplot(
            graph_metrics_by_quantile,
            x="Centiloid Quantile",
            y=_metric,
            hue="Dataset",
            ax=_ax.flat[_i],
        )

    _fig.tight_layout()
    _ax[0].legend().set_visible(False)
    _ax[1].legend().set_visible(False)
    plt.show()
    return


@app.cell
def __():
    # Small world without random graph reference
    # _fig,_ax = plt.subplots(1,2,figsize=(8,3))

    # sns.boxplot(x=graph_metrics_by_quantile['Centiloid Quantile'],y=graph_metrics_by_quantile['Weighted Clustering Coefficient']/graph_metrics_by_quantile['Weighted Avg. Shortest Path Length'],ax=_ax[0])
    # _ax[0].set_ylabel("Wgt. Unref. Small World")

    # sns.boxplot(x=graph_metrics_by_quantile['Centiloid Quantile'],y=graph_metrics_by_quantile['Clustering Coefficient']/graph_metrics_by_quantile['Avg. Shortest Path Length'],ax=_ax[1])
    # _ax[1].set_ylabel("Unreferenced Small World")

    # _fig.tight_layout()
    # plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        """# Statistical tests
    Are the three groups significantly different?"""
    )
    return


@app.cell
def __(mo):
    mo.md("# Identify important nodes")
    return


@app.cell
def __(mo):
    mo.md("# Utilities")
    return


@app.cell
def __(nx):
    def small_world_coeff(G, niter=1, nrand=10):
        """Compute the small world coefficient of a weighted graph. Average over `nrand` samples of the randomized graph."""
        Crand = 0
        Lrand = 0

        for _ in range(nrand):
            G_rand = nx.random_reference(G, niter)

            Crand += nx.average_clustering(G_rand, weight="correlation")
            Lrand += nx.average_shortest_path_length(G_rand, weight="distance")

        C = nx.average_clustering(G, weight="correlation")
        L = nx.average_shortest_path_length(G, weight="distance")

        return (C / Crand) / (L / Lrand)
    return small_world_coeff,


@app.cell
def __(nx):
    def unweighted_small_world_coeff(G, niter=1, nrand=10):
        """Compute the small world coefficient of a weighted graph. Average over `nrand` samples of the randomized graph."""
        Crand = 0
        Lrand = 0

        for _ in range(nrand):
            G_rand = nx.random_reference(G, niter)

            Crand += nx.average_clustering(G_rand)
            Lrand += nx.average_shortest_path_length(G_rand)

        C = nx.average_clustering(G)
        L = nx.average_shortest_path_length(G)

        return (C / Crand) / (L / Lrand)
    return unweighted_small_world_coeff,


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

        # return (1 / np.abs(pcorr) - 1).replace({np.inf: 0, -np.inf: 0})

        # return (pcorr != 0) * (1 - np.abs(pcorr)) # same as Dyrba 2020, but removing connections set to 0 by lasso

        if pcorr <= 0:
            return 0
        else:
            return -np.arctanh(np.abs(pcorr) - 1)

        # return 1 - np.abs(pcorr) # this is what Dyrba 2020 claims to use, but it makes fully connected graphs?

        # return -np.log(np.abs(pcorr)).replace({np.inf: 0, -np.inf: 0, np.nan:0})
    return pcorr_to_distance,


@app.cell
def __(GraphicalLasso, PowerTransformer, make_pipeline, pd):
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
    return compute_precision,


@app.cell
def __(compute_precision, partial_correlation):
    def compute_partial_correlation(data, params):
        """Convenience function to compute the partial correlation matrix from data"""

        return partial_correlation(compute_precision(data, params))
    return compute_partial_correlation,


@app.cell
def __(np, nx, partial_correlation, pcorr_to_distance):
    # from precision matrix to graph
    def precision_to_graph(precision):
        """Convert the provided precision matrix into a networkx graph.

        Arguments:
        precision: dataframe, column names will become node labels
        allow_self_connections: bool, if False zero out the diagonal"""

        # Adjacency matrix
        adj = partial_correlation(precision)

        graph = nx.from_pandas_adjacency(adj)

        graph.remove_node("PTAGE")

        # remove self loops
        graph.remove_edges_from([(u, v) for u, v in graph.edges() if u == v])

        # rename weight to correlation
        nx.set_edge_attributes(
            graph,
            {
                (u, v): {"correlation": d["weight"]}
                for u, v, d in graph.edges(data=True)
            },
        )

        # convenience: absolute value of correlation
        nx.set_edge_attributes(
            graph,
            {
                (u, v): {"abs(correlation)": np.abs(d["weight"])}
                for u, v, d in graph.edges(data=True)
            },
        )

        # compute distances from partial correlations
        nx.set_edge_attributes(
            graph,
            name="distance",
            values={
                (u, v): pcorr_to_distance(weight)
                for u, v, weight in graph.edges(data="weight")
            },
        )

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
def __(compute_metrics, compute_precision, nx, precision_to_graph):
    def data_to_metrics(data, params, metrics, randomize_graph=False):
        # Convenience function, wrapping all steps into one
        precision = compute_precision(data, params=params)

        if randomize_graph:
            graph = nx.random_reference(precision_to_graph(precision))
        else:
            graph = precision_to_graph(precision)

        return compute_metrics(graph, metrics)
    return data_to_metrics,


@app.cell
def __(data_to_metrics, multiprocessing, partial, pd):
    def boostrap_graph_metrics(
        data, params, metrics, n_samples=8, randomize_graph=False
    ):
        """Resample the datframe data, generate graph and compute metrics. Returns a dataframe with the bootstrapped metrics"""

        bootstrap_samples = [
            data.sample(frac=1, replace=True) for _ in range(n_samples)
        ]

        with multiprocessing.Pool() as pool:
            res = pool.map(
                partial(
                    data_to_metrics,
                    params=params,
                    metrics=metrics,
                    randomize_graph=randomize_graph,
                ),
                bootstrap_samples,
            )

        return pd.DataFrame(res)
    return boostrap_graph_metrics,


@app.cell
def __(multiprocessing):
    def bootstrap(data, func, n_samples=8):
        # Resample from data and apply fun to each sample
        # use this version for general functions rather than graph metrics

        bootstrap_samples = [
            data.sample(frac=1, replace=True) for _ in range(n_samples)
        ]

        with multiprocessing.Pool() as pool:
            res = pool.map(func, bootstrap_samples)

        return res
    return bootstrap,


@app.cell
def __(mo):
    mo.md("## Imports")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import networkx as nx
    import multiprocessing
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import altair as alt

    from functools import partial

    import seaborn as sns

    plt.style.use("ggplot")

    pd.set_option("display.max_columns", 100)
    return alt, mo, multiprocessing, np, nx, partial, pd, plt, px, sns


@app.cell
def __():
    from sklearn.preprocessing import PowerTransformer
    from sklearn.covariance import GraphicalLasso
    from sklearn.pipeline import make_pipeline
    return GraphicalLasso, PowerTransformer, make_pipeline


if __name__ == "__main__":
    app.run()
