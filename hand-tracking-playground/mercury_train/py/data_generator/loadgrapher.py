import os
import psutil
import time
import pandas
import subprocess
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

sequence_length: float = 200.0


class State:
    superroot: str = "/4/generation_run_jan9"
    last_num_sequences: int = 0
    last_sample_time: float

    df: pd.DataFrame


def get_num_sequences(st: State):
    return len(os.listdir(st.superroot))


def main():

    fig = plt.figure(figsize=(10, 5))

    df = pd.read_csv("log.csv")
    df = df.iloc[:, 1:]
    # df.plot()

    # df_rolling = df.rolling(window=10).mean()
    # df_new.plot()

    df_ewm = df.ewm(span=15, adjust=False).mean()
    df_ewm.plot(ylim=[0, 100])

    # plt.show()
    fig.savefig("hi.png", dpi=150)
    # plt.savefig("hi.png", dpi=150) #, figsize=(10, 5))


def main():

    df = pd.read_csv("log.csv")
    df = df.iloc[:, 1:]
    # df.plot()

    # df_rolling = df.rolling(window=10).mean()
    # df_new.plot()

    regular_span = 15
    smooth_keys = "images_per_second", "gpu utilization %", "gpu mem used %", "gpu mem utilization %", "cpu usage %"

    num_sample_keys = set([str(d) for d in range(7)])

    all_keys = set(df.keys())

    not_num_sample_keys = all_keys.difference(num_sample_keys)

    print(df.keys())

    for idx, key in enumerate(smooth_keys):
        df[key] = df[key].ewm(alpha=0.5).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(y=not_num_sample_keys, ylim=[0, 100], ax=ax)
    fig.savefig("hi.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(y=num_sample_keys, ax=ax)
    fig.savefig("num.png", dpi=150)


if __name__ == "__main__":
    main()
