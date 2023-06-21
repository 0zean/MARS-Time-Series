import matplotlib.pyplot as plt


def graph(x, y, y2, a, b, Title):
    fig = plt.figure()
    plt.plot(x[a:b], y[a:b], 'r', label='Actual')
    plt.plot(x[a:b], y2[a:b], 'b', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(Title)
    plt.legend(loc='upper left')
    plt.show()
    return fig


def table(table_df):
    """
    Parameters
    ----------
    table_df : pd.DataFrame
    ----------
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    rows = 5
    cols = 4

    ax.set_ylim(-.5, rows)
    ax.set_xlim(0, cols)

    for row in table_df.iterrows():
        ax.text(x=0, y=row[0], s=row[1][""], va="center",
                ha="left", weight="bold", size=14)
        ax.text(x=1.5, y=row[0], s=row[1]["MSE"], va="center",
                ha="center", size=14)
        ax.text(x=2.5, y=row[0], s=row[1]["R\u00b2"], va="center",
                ha="center", size=14)
        ax.text(x=3.5, y=row[0], s=row[1]["Standard Error"], va="center",
                ha="center", size=14)

    ax.axis('off')

    ax.text(1.5, rows-1, 'MSE', weight='bold', ha='center', size=16)
    ax.text(2.5, rows-1, 'R\u00b2', weight='bold', ha='center', size=16)
    ax.text(3.5, rows-1, 'Standard Error', weight='bold', ha='center', size=16)

    ax.set_title(
            'Model Metrics In and Out of Sample',
            loc='center',
            fontsize=18,
            weight='bold')

    ax.plot([0, cols+8], [rows-1.2, rows-1.2], ls="-", lw=1, c="black")

    for row in table_df.iterrows():
        ax.plot(
            [0, cols-.2],
            [row[0] - .5, row[0] - .5],
            ls=':',
            lw='.5',
            c='grey'
        )
    plt.show()
    return fig
