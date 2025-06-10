import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataPlotter:
    def __init__(self, df):
        self.df = df

    def scatter_with_agg_line(self, x_col, y_col, title='Scatter Plot', overlay_by=None,
                              agg_func=None, split_by_row=None, split_by_col=None,
                              ref_lines_x=None, ref_lines_y=None,output_path=None):
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError(f"Columns {x_col} and {y_col} must be in the dataframe")

        row_categories = self.df[split_by_row].unique() if split_by_row and split_by_row in self.df.columns else [None]
        col_categories = self.df[split_by_col].unique() if split_by_col and split_by_col in self.df.columns else [None]

        num_rows = len(row_categories)
        num_cols = len(col_categories)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=True, sharey=True)

        if num_rows == 1 and num_cols == 1:
            axes = [[axes]]
        elif num_rows == 1:
            axes = [axes]
        elif num_cols == 1:
            axes = [[ax] for ax in axes]

        for i, row_val in enumerate(row_categories):
            for j, col_val in enumerate(col_categories):
                ax = axes[i][j]
                subset_df = self.df
                if row_val is not None:
                    subset_df = subset_df[subset_df[split_by_row] == row_val]
                if col_val is not None:
                    subset_df = subset_df[subset_df[split_by_col] == col_val]

                sns.scatterplot(data=subset_df, x=x_col, y=y_col, ax=ax, hue=overlay_by)

                if agg_func == 'mean' and overlay_by:
                    agg_df = subset_df.groupby([x_col, overlay_by])[y_col].mean().reset_index()
                    sns.lineplot(data=agg_df, x=x_col, y=y_col, ax=ax, hue=overlay_by, linewidth=2, legend=False)

                if j == num_cols - 1 and split_by_row:
                    ax.text(1.05, 0.5, f"{row_val}", transform=ax.transAxes, fontsize=12, va='center', ha='left')
                if i == 0 and split_by_col:
                    ax.set_title(f"{split_by_col}={col_val}")

                if ref_lines_x:
                    for x_val, text in ref_lines_x:
                        ax.axvline(x=x_val, linestyle='--', color='red', alpha=0.7)
                        ax.text(x_val, ax.get_ylim()[1] * 0.95, text, color='red', fontsize=10, ha='right')
                if ref_lines_y:
                    for y_val, text in ref_lines_y:
                        ax.axhline(y=y_val, linestyle='--', color='blue', alpha=0.7)
                        ax.text(ax.get_xlim()[0] * 1.05, y_val, text, color='blue', fontsize=10, va='bottom')

                ax.grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        if (output_path):
            plt.savefig(output_path)
        else:
            plt.show()
            
