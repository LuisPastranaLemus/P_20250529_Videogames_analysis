# Exploratory Data Analysis for Visualizations and summary statistics

from IPython.display import display, HTML
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Function to detect outlier boundaries with optional clamping of lower bound to zero
def outlier_limit_bounds(df, column, bound='both', clamp_zero=False):
    """
    Detects outlier thresholds based on the IQR method and returns rows beyond those limits.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the numerical column to analyze.
    bound (str): One of 'both', 'lower', or 'upper' to indicate which bounds to evaluate.
    clamp_zero (bool): If True, clamps the lower bound to zero (useful for non-negative metrics).

    Returns:
    DataFrame(s): Rows identified as outliers, depending on the bound selected.
    """

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = max(q1 - 1.5 * iqr, 0) if clamp_zero else q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if bound == 'both':
        df_outliers_lb = df[df[column] < lower_bound]
        df_outliers_ub = df[df[column] > upper_bound]
        display(HTML(f"> Lower outlier threshold for column <i>'{column}'</i>: <b>{lower_bound}</b>"))
        display(HTML(f"> Upper outlier threshold for column <i>'{column}'</i>: <b>{upper_bound}</b>"))

        if df_outliers_lb.empty and df_outliers_ub.empty:
            display(HTML(f"> No outliers found in column <i>'{column}'</i>."))

        return df_outliers_lb, df_outliers_ub

    elif bound == 'upper':
        df_outliers_ub = df[df[column] > upper_bound]
        display(HTML(f"> Upper outlier threshold for column <i>'{column}'</i>: <b>{upper_bound}</b>"))

        if df_outliers_ub.empty:
            display(HTML(f"> No upper outliers found in column <i>'{column}'</i>."))

        return df_outliers_ub

    elif bound == 'lower':
        df_outliers_lb = df[df[column] < lower_bound]
        display(HTML(f"> Lower outlier threshold for column <i>'{column}'</i>: <b>{lower_bound}</b>"))

        if df_outliers_lb.empty:
            display(HTML(f"> No lower outliers found in column <i>'{column}'</i>."))

        return df_outliers_lb

    else:
        display(HTML(f"> Invalid 'bound' parameter. Use <b>'both'</b>, <b>'upper'</b>, or <b>'lower'</b>."))
        return None

# Function to evaluate the central tendency of a numerical feature
def evaluate_central_trend(df, column):
    """
    Evaluates the central tendency of a given column using the coefficient of variation (CV).
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): Name of the numerical column to evaluate.
    
    Output:
    Displays the coefficient of variation and recommends the most reliable measure of central tendency
    based on the level of variability.
    """
    
    cv = (df[column].std() / df[column].mean()) * 100
    display(HTML(f"> Coefficient of variation for column <i>'{column}'</i>: <b>{cv:.2f} %</b>"))

    if 0 <= cv <= 10:
        display(HTML("> Very low variability: <i>highly reliable mean</i>. Recommended central measure: <b>mean</b>."))
    elif 10 < cv <= 20:
        display(HTML("> Moderate variability: <i>reasonably reliable mean</i>. Recommended central measure: <b>mean</b>."))
    elif 20 < cv <= 30:
        display(HTML("> Considerable variability: <i>potentially biased mean</i>. Recommended central measure: <b>mean</b> with caution."))
    else:
        display(HTML("> High variability: <i>mean may be misleading</i>. Recommended central measure: <b>median</b>."))
    
    print()

# Function to evaluate pairwise correlations among numerical columns
def evaluate_correlation(df):
    """
    Evaluates pairwise Pearson correlations between numerical columns in a DataFrame.
    
    Parameters:
    - df (DataFrame): Input DataFrame with at least two numeric columns.
    
    Output:
    - Displays correlation coefficients with interpretation:
        > Strong correlation (|r| > 0.7)
        > Moderate correlation (0.3 < |r| ≤ 0.7)
        > Weak or no linear relationship (|r| ≤ 0.3)
        > Positive vs. Negative direction
    """

    numeric_cols = df.select_dtypes(include='number').columns

    seen_pairs = set()

    for col_x in numeric_cols:
        for col_y in numeric_cols:
            if col_x != col_y and (col_y, col_x) not in seen_pairs:
                corr = df[col_x].corr(df[col_y])

                strength = ''
                direction = 'positive' if corr > 0 else 'negative' if corr < 0 else 'neutral'

                abs_corr = abs(corr)
                if abs_corr > 0.7:
                    strength = 'Strong'
                elif abs_corr > 0.3:
                    strength = 'Moderate'
                elif abs_corr == 0:
                    strength = 'No linear relationship'
                else:
                    strength = 'Weak'

                if abs_corr > 0:
                    if strength in ['Strong', 'Moderate']:  
                        display(HTML(
                            f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                            f"<b>{strength} {direction} correlation</b><br><br>"
                        ))
                    else:
                        display(HTML(
                            f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                            f"{strength} {direction} correlation<br><br>"
                        ))
                else:
                    if strength in ['Strong', 'Moderate']: 
                        display(HTML(
                            f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                            f"<b>{strength}</b><br><br>"
                        ))
                    else:
                        display(HTML(
                            f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                            f"{strength}<br><br>"
                        ))
                        
                seen_pairs.add((col_x, col_y))

# Function to visualize missing values within a DataFrame using a heatmap
def missing_values_heatmap(df):
    """
    Displays a heatmap of missing (NaN) values in the given DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame to analyze.
    
    Output:
    A heatmap visualization showing the presence of missing values per column and row.
    """
    plt.figure(figsize=(15, 7))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.show()

# Function for heatmap data visualization
# plot_heatmap(data=heatmap_data, title='User Segmentation by Repurchase Rate', xlabel='Repurchase Rate Bins', ylabel='Repurchase Segment',
#              cmap='YlGnBu', cbar_label='User Count')
def plot_heatmap(data, title='', xlabel='', ylabel='', cmap='YlGnBu', annot=True, fmt='d', cbar_label='Count', figsize=(15, 7)):
    """
    Plots a heatmap with customization options.

    Parameters:
    data (DataFrame): Pivot table to visualize.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    cmap (str): Color map for heatmap.
    annot (bool): Whether to show values inside the heatmap cells.
    fmt (str): Format for annotation text.
    cbar_label (str): Label for the color bar.
    figsize (tuple): Figure size (width, height).

    Returns:
    None: Displays the heatmap.
    """
    # fmt='d' | fmt='.0f'
    
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, cbar_kws={'label': cbar_label})

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Function to plot multiple boxplots side by side for comparison
# plot_boxplots(ds_list=[serie1, serie2, serie3], xlabels=['Group A', 'Group B', 'Group C'], ylabel='Values', 
#               title='Comparison of Value Distributions Across Groups', yticks_range=(0, 40, 5), rotation=45,
#               color=['skyblue', 'lightgreen', 'salmon']
def plot_boxplots(ds_list, xlabels, ylabel, title, yticks_range=None, rotation=0, color='grey', showfliers=True):
    """
    Plots multiple boxplots side by side, allowing for visual comparison across groups.

    Parameters:
    ds_list (list of Series): List of numerical pandas Series to plot.
    xlabels (list of str): Corresponding labels for each dataset.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    yticks_range (tuple, optional): Range for y-axis ticks, e.g., (min, max, step).
    rotation (int, optional): Rotation angle for x and y tick labels.
    color (str or list, optional): Either a single color or a list of colors matching the groups.

    Raises:
    ValueError: If the number of datasets and labels do not match.

    Output:
    Displays a customized boxplot figure for group-wise value comparison.
    """

    if len(ds_list) != len(xlabels):
        raise ValueError("*** Error *** > The data list and labels must be the same length.")
    
    df = pd.DataFrame({
        'value': pd.concat(ds_list, ignore_index=True),
        'group': sum([[label] * len(s) for label, s in zip(xlabels, ds_list)], [])
    })

    plt.figure(figsize=(15, 7))

    # If color is a list, assign a custom palette; if string, use a solid color
    if isinstance(color, (list, tuple)) and len(color) == len(xlabels):
        palette = dict(zip(xlabels, color))
        sns.boxplot(x='group', y='value', hue='group', data=df, palette=palette, showfliers=showfliers)
    else:
        sns.boxplot(x='group', y='value', data=df, color=color, showfliers=showfliers)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)

    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a histogram with mean and median reference lines
# plot_histogram(ds=df_product_purchase_quantity['total_products'], bins=range(0, 65, 1), color='grey', title='Distribution for Product Quantity by Orders',
#                xlabel='Products', ylabel='Frequency', xticks_range=range(0, 65, 5), yticks_range=range(0, 200, 20), rotation=45)
def plot_histogram(ds, bins=10, color='grey', title='', xlabel='', ylabel='Frequency', xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots a histogram for a given numerical Series with optional customization.

    Parameters:
    ds (Series): The numerical data to plot.
    bins (int or array-like): Number or range of histogram bins.
    color (str): Fill color for the bars.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xticks_range (range, optional): Range object for x-ticks (e.g., range(0, 100, 10)).
    yticks_range (range, optional): Range object for y-ticks (e.g., range(0, 10, 1)).
    rotation (int): Angle of tick label rotation.

    Output:
    Displays a histogram with vertical lines for mean and median.
    """

    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, edgecolor='black', color=color, kde=False)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if isinstance(xticks_range, range):
        plt.xlim(xticks_range.start, xticks_range.stop)
        plt.xticks(xticks_range, rotation=rotation)

    if isinstance(yticks_range, range):
        plt.ylim(yticks_range.start, yticks_range.stop)
        plt.yticks(yticks_range, rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a stacked histogram by group (hue)
# plot_hue_histogram(df=my_df, x_col='duration', hue_col='subscription_type', bins=20, title='Distribution of Durations by Subscription Type',
#                    xlabel='Duration (minutes)', ylabel='Frequency', legend_title='Subscription', legend_labels=['Free', 'Premium'])


def plot_hue_histogram(df, x_col='', hue_col='', bins=30, color='grey', title='', xlabel='', ylabel='',
                       legend_title='', legend_labels=[]):
    """
    Plots a stacked histogram with grouping by a categorical variable (hue).

    Parameters:
    df (DataFrame): Input dataset.
    x_col (str): Numerical column to plot on the x-axis.
    hue_col (str): Categorical column used to group data.
    bins (int): Number of histogram bins.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    legend_title (str): Title for the legend.
    legend_labels (list, optional): Custom labels for legend categories.

    Output:
    Displays a stacked histogram with hue-based grouping.
    """
    
    plt.figure(figsize=(15, 7))
    sns.histplot(data=df, x=x_col, hue=hue_col, multiple='stack', bins=bins, color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend_labels:
        plt.legend(title=legend_title, labels=legend_labels)
    else:
        plt.legend(title=legend_title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to compare two distributions using overlapping histograms
# plot_dual_histogram(ds1=ages_no_show, ds2=ages_showed_up, bins=18, color1='tomato', color2='mediumseagreen', 
#                     title='Age Distribution: No-Show vs Show', xlabel='Age', ylabel='Number of Patients', label1='No Show',
#                     label2='Showed Up', xticks_range=(0, 100, 10), yticks_range=(0, 80, 10), rotation=45)
def plot_dual_histogram(ds1, ds2, bins=10, color1='black', color2='grey', title='Histogram Comparison', xlabel='', ylabel='',
                        label1='', label2='', xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots two overlapping histograms to visually compare distributions.

    Parameters:
    ds1 (Series): First numerical dataset.
    ds2 (Series): Second numerical dataset.
    bins (int): Number of bins for the histogram.
    color1 (str): Color for the first dataset.
    color2 (str): Color for the second dataset.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    label1 (str): Legend label for the first dataset.
    label2 (str): Legend label for the second dataset.
    xticks_range (tuple, optional): Range and step for x-ticks (min, max, step).
    yticks_range (tuple, optional): Range and step for y-ticks (min, max, step).
    rotation (int): Tick label rotation angle.

    Output:
    Displays overlapping histograms with mean and median lines for both datasets.
    """

    # Clean missing values
    ds1 = ds1.dropna()
    ds2 = ds2.dropna()

    # Compute statistics
    mean1_val = ds1.mean()
    median1_val = ds1.median()
    mean2_val = ds2.mean()
    median2_val = ds2.median()

    plt.figure(figsize=(15, 7))

    sns.histplot(ds1, bins=bins, edgecolor='black', kde=False, color=color1, label=label1, alpha=0.8)
    sns.histplot(ds2, bins=bins, edgecolor='black', kde=False, color=color2, label=label2, alpha=0.6)

    plt.axvline(mean1_val, color='red', linestyle='dashed', linewidth=1.5, label=f'{label1} Mean: {mean1_val:.2f}')
    plt.axvline(mean2_val, color='darkred', linestyle='dashed', linewidth=1.5, label=f'{label2} Mean: {mean2_val:.2f}')
    plt.axvline(median1_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'{label1} Median: {median1_val:.2f}')
    plt.axvline(median2_val, color='darkblue', linestyle='dashdot', linewidth=1.5, label=f'{label2} Median: {median2_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range is not None:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)
    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a frequency density histogram with optional KDE overlay
# plot_frequency_density(ds=series1, bins=np.arange(0, 1200, 50), color='grey', title='Frequency Density of Duration', 
#                        xlabel='Duration (minutes)', ylabel='Density', xticks_range=(0, 1200, 100), show_kde=True, rotation=45)
def plot_frequency_density(ds, bins=10, color='grey', title='', xlabel='', ylabel='Density',
                           xticks_range=None, rotation=0, show_kde=True):
    """
    Plots a frequency density histogram with optional KDE curve.

    Parameters:
    ds (Series): Numerical data to plot.
    bins (int or array-like): Number or range of bins for the histogram.
    color (str): Histogram bar color.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis (default: 'Density').
    xticks_range (tuple, optional): Tuple (min, max, step) for x-tick configuration.
    rotation (int, optional): Angle for tick label rotation.
    show_kde (bool, optional): Whether to overlay a KDE curve.

    Output:
    Displays a histogram normalized to show frequency density, with mean/median lines and optional KDE.
    """

    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, stat='density', edgecolor='black', color=color, alpha=0.7)

    if show_kde:
        sns.kdeplot(ds, color='darkblue', linewidth=2, label='KDE')

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a grouped barplot (categorical x-axis, grouped by hue)
# plot_grouped_barplot(ds=dataframe, x_col='month', y_col='median_duration', hue_col='plan', palette=['black', 'grey'],
#                      title='Average Call Duration by Plan and Month', xlabel='Month', ylabel='Average Call Duration (min)',
#                      xticks_range=range(0, 13, 1), yticks_range=range(0, 500, 50), rotation=65)
def plot_grouped_barplot(ds, x_col, y_col, hue_col=None, palette=sns.color_palette("PRGn", n_colors=50), title='', xlabel='', ylabel='', 
                         xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0, alpha=0.95, show_legend=True, show_values=True):
    """
    Plots a grouped bar chart with categorical grouping (hue).

    Parameters:
    ds (DataFrame): The dataset to use for plotting.
    x_col (str): The column to use for the x-axis (categorical).
    y_col (str): The column to plot as the bar height (numerical).
    hue_col (str, optional): The column to group by within each x-category.
    palette (list, optional): List of colors for each hue category.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xticks_range (range, optional): Tick range and step for the x-axis.
    yticks_range (range, optional): Tick range and step for the y-axis.
    x_rotation (int): Rotation angle for x-axis ticks.
    y_rotation (int): Rotation angle for y-axis ticks.
    alpha (float): Transparency of bars.
    show_legend (bool): Whether to display the legend (default: True).
    show_values (bool): Whether to display value labels on top of bars.

    Output:
    Displays a grouped bar plot with optional axis customization and legend.
    """

    fig, ax = plt.subplots(figsize=(15, 7))
    palette = palette
    strong_palette = palette[:13] + palette[-12:]
    sns.barplot(data=ds, x=x_col, y=y_col, hue=hue_col, palette=strong_palette, alpha=alpha, ax=ax)

    if show_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            height + (0.01 * height),
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks_range is not None:
        ax.set_xticks(ticks=xticks_range)
    ax.tick_params(axis='x', rotation=x_rotation)

    if yticks_range is not None:
        ax.set_yticks(ticks=yticks_range)
    ax.tick_params(axis='y', rotation=y_rotation)

    if not show_legend:
        ax.legend().remove()

    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a horizontal bar chart from categorical data
# plot_horizontal_bar(ds=series_categorica, colors=['skyblue', 'salmon', 'lightgreen'], xlabel='Count', ylabel='Categories',
#                     title='Distribution of Categorical Values', xticks_range=(0, 100, 10), rotation=0)
def plot_horizontal_bar(ds, colors=['black', 'grey'], xlabel='', ylabel='', title='', xticks_range=None, rotation=0, show_values=True):
    """
    Plots a horizontal bar chart for a categorical pandas Series.

    Parameters:
    ds (Series): Categorical data to summarize and visualize.
    colors (list): Color palette for each category.
    xlabel (str): Label for the x-axis (typically counts).
    ylabel (str): Label for the y-axis (categories).
    title (str): Title of the plot.
    xticks_range (tuple, optional): Tuple (min, max, step) for x-axis ticks.
    rotation (int): Rotation angle for x-axis tick labels.
    show_values (bool): Whether to display the value at the end of each bar.

    Output:
    Displays a horizontal bar chart with optional hue differentiation.
    """

    categories = ds.value_counts().index
    values = ds.value_counts().values

    plt.figure(figsize=(15, 7))
    ax = sns.barplot(y=categories, x=values, hue=categories, dodge=False, palette=colors)
    
    if show_values:
        for i, v in enumerate(values):
            ax.text(v + max(values)*0.01, i, f'{v}', va='center', fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks_range is not None:
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot grouped bar charts from a DataFrame with multiple columns
# plot_grouped_bars(df=music_activity_df.set_index('city'), title='Music Activity per City', xlabel='City', ylabel='Activity Count',
#                   x_rotation=45, y_rotation=0, grid_axis='y')
def plot_grouped_bars(df, title='', xlabel='', ylabel='', x_rotation=0, y_rotation=0, grid_axis='y', color='grey', show_values=True):
    """
    Plots grouped (clustered) bar charts for comparing multiple categories across an index.

    Parameters:
    df (DataFrame): A DataFrame where the index defines groups (e.g., cities) and columns are subcategories.
    title (str): Title of the chart.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    x_rotation (int): Rotation angle for x-axis tick labels.
    y_rotation (int): Rotation angle for y-axis tick labels.
    grid_axis (str): Axis along which to display grid lines ('x', 'y', or 'both').
    show_values (bool): Whether to display values on top of each bar.

    Output:
    Displays a grouped bar chart comparing values across index categories and columns.
    """

    ax = df.plot(kind='bar', figsize=(15, 7), color=color)
    
    if show_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + (0.01 * height),
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    plt.grid(axis=grid_axis)
    plt.tight_layout()
    plt.show()


# Function to plot a grouped bar chart with a specified index column
# plot_grouped_bars_indx(df=music_df, index_name='city', title='Music Activity per City', xlabel='City', ylabel='Activity Count',
#                        rotation=45, grid_axis='y')
def plot_grouped_bars_indx(df, index_name='', title='', xlabel='', ylabel='', rotation=0, grid_axis='y', color='grey', show_values=True):
    """
    Plots a grouped bar chart where rows are grouped by a specified index column
    (e.g., city) and bars represent multiple numeric columns.

    Parameters:
    df (DataFrame): Input DataFrame with categorical and numeric columns.
    index_name (str): Name of the column to use as the index (grouping variable).
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    rotation (int): Rotation angle for x-axis tick labels.
    grid_axis (str): Axis to show grid lines on ('x', 'y', or 'both').
    show_values (bool): Whether to display the value above each bar.

    Returns:
    None: Displays the bar chart.
    """

    df_plot = df.set_index(index_name)
    ax = df_plot.plot(kind='bar', figsize=(15, 7), color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(axis=grid_axis)
    plt.tight_layout()
    
    if show_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:  # avoid placing labels on zero-height bars
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01 * height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8
                    )
    
    plt.show()

# Function to generate a customizable seaborn pairplot for exploratory correlation analysis
# plot_pairplot(df_music_activity_city_cov)
def plot_pairplot(df, height=3, aspect=2.5, point_color='grey'):
    """
    Plots a Seaborn pairplot for all numeric columns in a DataFrame.

    Parameters:
    df (DataFrame): The dataset to plot.
    height (float): Height (in inches) of each facet (subplot).
    aspect (float): Aspect ratio of each facet (width = height × aspect).

    Returns:
    None: Displays the pairplot.
    """
    sns.pairplot(df, height=height, aspect=aspect, plot_kws={'color': point_color})
    plt.tight_layout()
    plt.show()

# Function to plot a scatter matrix for exploring pairwise relationships
def plot_scatter_matrix(df, figsize=(15, 7), diagonal='hist', color='grey', alpha=0.3):
    """
    Plots a scatter matrix for all numeric columns in a DataFrame using pandas' plotting tools.

    Parameters:
    df (DataFrame): The dataset to visualize.
    figsize (tuple): Size of the overall figure.
    diagonal (str): Type of plot on the diagonal ('hist' or 'kde').

    Returns:
    None: Displays the scatter matrix.
    """
    pd.plotting.scatter_matrix(df, figsize=figsize, diagonal=diagonal, color=color, alpha=alpha)
    plt.tight_layout()
    plt.show()

# Function for scatter plot
# plot_scatter(df=df_product_purchase, x_col='add_to_cart_order', y_col='reorder_rate', title='Add-to-Cart Order vs Reorder Rate',
#              xticks_range=range(0, 50, 5), yticks_range=range(0, 2), x_rotation=45, y_rotation=0, alpha=0.4, color='teal')
def plot_scatter(df, x_col, y_col, title=None, xlabel=None, ylabel=None, figsize=(15, 7), alpha=0.3, color='grey', marker='o',
                 hue=None, palette=None, xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0):
    """
    Plots a scatterplot for two numerical columns with optional customization.

    Parameters:
    df (DataFrame): Data source.
    x_col (str): Column name for x-axis.
    y_col (str): Column name for y-axis.
    title (str, optional): Plot title.
    xlabel (str, optional): Label for the x-axis.
    ylabel (str, optional): Label for the y-axis.
    figsize (tuple): Figure size (width, height).
    alpha (float): Transparency level for points.
    color (str): Color of the points.
    marker (str): Marker style for scatter points.
    hue (str, optional): Column name to use for color grouping.
    palette: color palette if hue is used
    xticks_range (range, optional): Custom range for x-axis ticks.
    yticks_range (range, optional): Custom range for y-axis ticks.
    x_rotation (int): Rotation angle for x-axis tick labels.
    y_rotation (int): Rotation angle for y-axis tick labels.
    
    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=alpha, color=None if hue else color, palette=palette, marker=marker)

    plt.title(title if title else f'Scatter: {x_col} vs. {y_col}')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)

    if isinstance(xticks_range, range):
        plt.xticks(xticks_range, rotation=x_rotation)
        plt.xlim(xticks_range.start, xticks_range.stop)

    if isinstance(yticks_range, range):
        plt.yticks(yticks_range, rotation=y_rotation)
        plt.ylim(yticks_range.start, yticks_range.stop)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Function to plot ECDF - Empirical Cumulative Distribution Function.
# plot_ecdf(df=df_products, x_col='reorder_rate', threshold=0.75, title='ECDF of Product Repurchase Rate', xlabel='Repurchase Rate',
#           xticks_range=range(0, 2), yticks_range=range(0, 2), x_rotation=0, y_rotation=0)
def plot_ecdf(df, x_col, threshold=None, title=None, xlabel=None, ylabel='Cumulative Percentage', figsize=(10, 6), color='blue', 
              linestyle='--', lw=1.5, xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0, grid=True):
    """
    Plots an Empirical Cumulative Distribution Function (ECDF) for a given column.

    Parameters:
    df (DataFrame): Data source.
    x_col (str): Column to plot ECDF for.
    threshold (float, optional): Draw vertical reference line at this x value.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    figsize (tuple): Figure size.
    color (str): Color of the ECDF line.
    linestyle (str): Style for the vertical threshold line.
    lw (float): Line width for the threshold.
    xticks_range (range, optional): Custom xtick positions.
    yticks_range (range, optional): Custom ytick positions.
    x_rotation (int): Rotation angle for x-axis labels.
    y_rotation (int): Rotation angle for y-axis labels.
    grid (bool): Whether to display the grid.

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.ecdfplot(data=df, x=x_col, color=color)

    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle=linestyle, linewidth=lw,
                    label=f'Threshold: {threshold}')

    plt.title(title if title else f'ECDF of {x_col}')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel)

    if isinstance(xticks_range, range):
        plt.xticks(xticks_range, rotation=x_rotation)
        plt.xlim(xticks_range.start, xticks_range.stop)

    if isinstance(yticks_range, range):
        plt.yticks(yticks_range, rotation=y_rotation)
        plt.ylim(yticks_range.start, yticks_range.stop)

    if threshold is not None:
        plt.legend()

    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Function to plot ECDF - Empirical Cumulative Distribution Function.
# plot_plan_revenue_by_city(df_city_revenue, color1='darkblue', color2='silver', title='Ingresos por ciudad y plan')
def plot_bar_comp(df, x_col, y_col, title, xlabel, ylabel, color1='black', color2='grey', alpha2=0.7,
                  figsize=(15, 7), rotation=0, fontsize=8, show_values=True):
    """
    Plots a grouped bar chart comparing revenue by city for two different plans.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the X-axis (e.g., city)
    - plan1: Column name for first plan revenue
    - plan2: Column name for second plan revenue
    - color1: Color for the first plan bars
    - color2: Color for the second plan bars
    - alpha2: Transparency for second plan bars
    - title: Title of the plot
    - xlabel: Label for X-axis
    - ylabel: Label for Y-axis
    - figsize: Size of the figure
    - rotation: Rotation angle for x-axis labels
    - fontsize: Font size for x-axis labels
    - show_values: Whether to display the value above each bar
    """
    plt.figure(figsize=figsize)
    bars1 = plt.bar(df[x_col], df[y_col[0]], label=y_col[0].upper(), color=color1)
    bars2 = plt.bar(df[x_col], df[y_col[1]], label=y_col[1].upper(), color=color2, alpha=alpha2)

    if show_values:
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(range(len(df[x_col])), df[x_col], rotation=rotation, fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a frequency density histogram with KDE, showing distribution shape and key statistics.
# Highlights mean, median, ±1σ, ±2σ, ±3σ boundaries, and marks outliers beyond 3σ.
# plot_distribution_with_statistics(df, 'sales', bins=10)
def plot_distribution_dispersion(data, column_name, bins=30, color='grey'):
    """
    Plot a frequency density histogram with KDE and key distribution statistics.

    Parameters:
    - data: DataFrame containing the data.
    - column_name: Name of the column to analyze.
    - bins: Number of bins for the histogram.
    - color: Base color for histogram and KDE (default: 'skyblue').

    Displays:
    - Histogram (frequency density) with KDE.
    - Mean, median lines.
    - ±1σ, ±2σ, ±3σ boundaries.
    - Outliers marked beyond 3σ.
    """
    
    values = data[column_name].dropna()

    # Basic stats
    mean = values.mean()
    median = values.median()
    std = values.std()

    # Sigma boundaries
    sigma_bounds = {
        '1σ': (mean - std, mean + std),
        '2σ': (mean - 2*std, mean + 2*std),
        '3σ': (mean - 3*std, mean + 3*std)
    }

    plt.figure(figsize=(15, 7))

    # Frequency density histogram with KDE
    sns.histplot(values, bins=bins, kde=True, stat='density',
                 color=color, edgecolor='black', alpha=0.6)

    # Mean and median
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='orange', linestyle=':', linewidth=2, label=f'Median: {median:.2f}')

    # Sigma boundaries
    colors = ['green', 'blue', 'purple']
    for i, (label, (low, high)) in enumerate(sigma_bounds.items()):
        plt.axvline(low, color=colors[i], linestyle='--', alpha=0.7, label=f'{label} Lower: {low:.2f}')
        plt.axvline(high, color=colors[i], linestyle='--', alpha=0.7, label=f'{label} Upper: {high:.2f}')
    
    # Outliers (beyond 3σ)
    outliers = values[(values < sigma_bounds['3σ'][0]) | (values > sigma_bounds['3σ'][1])]
    if not outliers.empty:
        plt.scatter(outliers, np.zeros_like(outliers), color='black', s=40, label='Outliers (3σ+)', marker='x')

    plt.title(f'Distribution of {column_name} with Statistical Boundaries')
    plt.xlabel(column_name)
    plt.ylabel('Frequency Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plots a vertical bar chart from a pandas Series with customizable labels, size, and ticks.
# plot_bar_from_series(sales_series, title='Total Sales by Platform and Year', xlabel='Platform - Year', ylabel='Total Sales (millions)',
#                      color='salmon', xticks=['Año 2015 - PS4', 'Año 2016 - PS4', 'Año 2015 - X360'], yticks=[0, 5, 10, 15])
def plot_bar_series(series, title='', xlabel='', ylabel='', figsize=(15, 7), color=None, rotation=45, show_values=True, xticks=None, 
                    yticks=None):
    """
    Crea una gráfica de barras verticales a partir de una Pandas Series.

    Parámetros:
    - series: pd.Series — Serie con índices como categorías y valores numéricos.
    - title: str — Título de la gráfica.
    - xlabel: str — Etiqueta del eje X.
    - ylabel: str — Etiqueta del eje Y.
    - figsize: tuple — Tamaño de la figura (ancho, alto).
    - color: str o lista — Color(es) de las barras.
    - rotation: int — Rotación de las etiquetas en el eje X.
    - show_values: bool — Mostrar los valores encima de las barras.
    - xticks: list o None — Lista personalizada de valores para el eje X.
    - yticks: list o None — Lista personalizada de valores para el eje Y.

    Retorna:
    - fig, ax: objetos matplotlib.
    """

    fig, ax = plt.subplots(figsize=figsize)

    series.plot(kind='bar', ax=ax, color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)

    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # Show values above bars
    if show_values:
        for i, val in enumerate(series):
            ax.text(i, val + max(series)*0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plots horizontal lines.
#
def plot_horizontal_lines(df, start_col='', end_col='', y_col='', title='', xlabel='', ylabel='', figsize=(15, 7), marker='o', grid=True, color='tab:grey'):
    """
    Plots horizontal lines.

    Parameters:
    - df: DataFrame containing data.
    - start_col: column name with the first column values.
    - end_col: column name with the last column values.
    - y_col: column name with the oject values.
    - title: title of the chart.
    - xlabel: label for the X axis.
    - ylabel: label for the Y axis.
    - figsize: figure size (width, height).
    - marker: marker style for endpoints (default is 'o').
    - grid: whether to display a grid (default is True).
    - color: line color (string or list of colors for each platform).

    Returns:
    - fig, ax: matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in df.iterrows():
        line_color = color[i] if isinstance(color, list) else color
        ax.plot([row[start_col], row[end_col]],
                [row[y_col], row[y_col]],
                marker=marker,
                color=line_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
