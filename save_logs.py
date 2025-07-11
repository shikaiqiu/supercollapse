import pickle
import pandas as pd
import wandb
import numpy as np

def average_df_series(series):
    # Initialize an accumulator DataFrame for sums
    sum_df = None
    # Number of DataFrames in the series
    n = len(series)

    for df in series:
        # If sum_df is None, initialize it with the first DataFrame
        if sum_df is None:
            sum_df = df.copy()
        else:
            # Otherwise, add the current DataFrame to the sum_df
            sum_df += df

    # Divide each value by n to get the average
    average_df = sum_df / n
    return average_df


def get_runs(filters, proj="supercollapse", tune_on=None, avg_seeds=False, history=True):
    api = wandb.Api()
    tags = filters['config.wandb_tag'].split(",")
    runs = []
    for tag in tags:
        filters_ = filters.copy()
        filters_['config.wandb_tag'] = tag
        runs.extend(api.runs(proj, filters=filters_, order="-created_at"))
    summary_list, config_list, name_list = [], [], []
    history_list = []
    for run in runs:
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        if history:
            hist = run._sampled_history(keys=["compute", "tokens", "test_loss"], x_axis="step", samples=100000)
            hist_df = pd.DataFrame(hist)
            try:
                lr_hist = run._sampled_history(keys=["lr", "tau"], x_axis="step", samples=100000)
                lr_hist_df = pd.DataFrame(lr_hist)
                hist_df = hist_df.merge(lr_hist_df, on='step', how='outer')
            except:
                pass
            try:
                trace_hist = run._sampled_history(keys=["Buvar"], x_axis="step", samples=100000)
                trace_hist_df = pd.DataFrame(trace_hist)
                hist_df = hist_df.merge(trace_hist_df, on='step', how='outer')
            except:
                history_list.append(None)
                continue
            hist_df['compute'] /= 1e15 # in units of peta flops
            hist_df = hist_df.ffill()
            # drop rows that have nans
            hist_df = hist_df.dropna()
            history_list.append(hist_df)

    runs_dict = {
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
    }
    if history:
        runs_dict["history"] = [history_list[i] for i in range(len(history_list)) if history_list[i] is not None]
        runs_dict["summary"] = [summary_list[i] for i in range(len(summary_list)) if history_list[i] is not None]
        runs_dict["config"] = [config_list[i] for i in range(len(config_list)) if history_list[i] is not None]
        runs_dict["name"] = [name_list[i] for i in range(len(name_list)) if history_list[i] is not None]
    runs_df = pd.DataFrame(runs_dict)

    runs_df = runs_df[runs_df["summary"].apply(lambda x: x != {})]
    keys = [
        "N",
        "V",
        "L",
        "D",
        "B",
        "lr",
        "T",
        "P",
        "num_params",
        "schedule",
        "decay_frac",
        "test_loss",
        "seed"
    ]
    for key in keys:
        # For other keys, just extract the value if it exists
        runs_df[key] = runs_df["config"].apply(lambda x: x[key] if key in x else -1)
    # delete name summary and config
    runs_df = runs_df.drop(columns=["summary", "config", "name"])

    # Everything else being equal, only keep the best run
    if tune_on is not None:
        if "test_loss" in tune_on:
            idx = runs_df.groupby(keys)[tune_on].idxmin()
        else:
            idx = runs_df.groupby(keys)[tune_on].idxmax()
        runs_df = runs_df.loc[idx]

    # average over seeds
    if avg_seeds:
        keys = [k for k in keys if k != "seed"]
        numeric_cols = runs_df.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {col: "mean" for col in numeric_cols}
        agg_dict.update({col: "first" for col in keys})
        if history:
            agg_dict["history"] = average_df_series
        history_cols = runs_df["history"].iloc[0].columns
        runs_df = runs_df.groupby(keys).agg(agg_dict).reset_index(drop=True)
        runs_df["history"] = runs_df["history"].apply(lambda row: pd.DataFrame(row, columns=history_cols))
    return runs_df


if __name__ == "__main__":
    # comma separated tags for experiments to include in the output file
    # e.g. for MLP lr schedules, use "mlp_schedules,mlp_schedules_across_D,mlp_schedules_across_T"
    # for mlp compute-optimal training horizon fit, use "mlp_fit"
    # for mlp linear decay, use "mlp"
    tag = "mlp_schedules,mlp_schedules_across_D,mlp_schedules_across_T"

    # output file name
    output_file = "logs/mlp_schedules.pkl"
    filters = {
        "config.wandb_tag": tag,
        "config.lr": 0.001,
        "state": "finished",
    }
    df = get_runs(filters, proj="supercollapse", tune_on="test_loss", history=True, avg_seeds=False)
    with open(output_file, "wb") as f:
        pickle.dump(df, f)