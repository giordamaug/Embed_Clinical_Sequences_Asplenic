import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_group_distribution_with_event_boxplot(df, groupby="primary_disease_area", label_desc=None):
    df = df.copy()

    if label_desc is not None:
        mapping = {float(i): label for i, label in enumerate(label_desc)}
        label_order = [label_desc[i] for i in df[groupby].dropna().unique().tolist()]
        df[groupby] = df[groupby].astype(float).map(mapping)
    else:
        label_order = df[groupby].dropna().unique().tolist()

    df['event_length'] = df['events'].apply(len)

    df = df.dropna(subset=[groupby, 'event_length'])

    counts = df[groupby].value_counts().reindex(label_order, fill_value=0)

    values = counts.values
    labels = counts.index

    if values.sum() == 0:
        raise ValueError(
            f"Nessun dato valido per {groupby}. "
            f"Controlla il mapping e i valori originali della colonna."
        )

    palette = sns.color_palette("tab10", len(label_order))
    palette_dict = dict(zip(label_order, palette))

    def autopct_abs(pct):
        total = np.sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}' if val > 0 else ''

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    wedges, texts, autotexts = axes[0].pie(
        values,
        labels=None,
        colors=[palette_dict[l] for l in labels],
        autopct=autopct_abs,
        pctdistance=0.75,
        startangle=90
    )

    axes[0].set_title("(A) Cohort Distribution")
    axes[0].legend(
        wedges,
        labels,
        title=groupby,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    sns.boxplot(
        data=df,
        x=groupby,
        y='event_length',
        order=label_order,
        hue=groupby,
        palette=palette_dict,
        dodge=False,
        legend=False,
        ax=axes[1]
    )

    axes[1].set_title("(B) Event Length Distribution")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()