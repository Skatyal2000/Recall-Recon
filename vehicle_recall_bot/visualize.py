import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_recalls_by_manufacturer(df):
    top_mfr = df['manufacturer'].value_counts().nlargest(10)
    fig, ax = plt.subplots()
    top_mfr.plot(kind='barh', ax=ax)
    ax.set_title("Top 10 Manufacturers by Number of Recalls")
    ax.set_xlabel("Number of Recalls")
    ax.invert_yaxis()
    return fig

def plot_potentially_affected_distribution(df):
    fig, ax = plt.subplots()
    df['potentially_affected'].dropna().clip(upper=50000).hist(bins=30, ax=ax)
    ax.set_title("Distribution of Potentially Affected Units")
    ax.set_xlabel("Units Affected (capped at 50,000)")
    return fig

def plot_recalls_over_time(df):
    df['report_received_date'] = pd.to_datetime(df['report_received_date'], errors='coerce')
    timeline = df.groupby(df['report_received_date'].dt.to_period('M')).size()
    fig, ax = plt.subplots()
    timeline.plot(ax=ax)
    ax.set_title("Recalls Over Time")
    ax.set_ylabel("Number of Recalls")
    ax.set_xlabel("Date")
    return fig
