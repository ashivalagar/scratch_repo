#########################################################################################
# Changes from plotting_with_new_example/plotting_experiments_scatterplot_boxplot_with_new_example_v4		
# 1. Iterate through files in the raw data folder and pass each to the plotting function.
# 2. Handle NaN values before stacking RWB: Drop columns where all values are NaN.
# Example: Remove columns V1 to V6 in the SLC WL dataset.
# 3. Handle NaN values before plotting RBER: Drop rows where the "check_na_col" column contains NaN.
# Example: Remove rows with NaN in the "max_min_bbc_hrer" column.
# 4. Separate the stacking process for Edge, Valley, and ESUMS data.	
# 5. Define plot_E0/plot_E1_onwards/plot_all_edges so MLC mode/SLC mode plot E0&E1 together. Purpose is to cut down number of plots.			
# 6. Fix color palette by moving it out from i,j loop
# 7. Add boxplot_with_split 	
#########################################################################################

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pptx import Presentation
from pptx.util import Inches
from matplotlib.patches import Patch

matplotlib.use('Agg')

class DataPlotter:
	def __init__(self, df):
		self.df = df
	
	def add_caption_box(self, subset_df, overlay_by, y_col, ax, num_cols, caption_func=None):
		if caption_func == 'mean' and overlay_by:
			# caption = subset_df.groupby([overlay_by])[y_col].mean().astype(int)
			caption = subset_df.groupby([overlay_by])[y_col].mean().apply(pd.to_numeric, errors='coerce')
			caption_df = caption.reset_index()
			caption_df.columns = ['label', 'mean']
			caption = caption_df.to_string(index=False)
		
		elif caption_func == 'median' and overlay_by:
			# caption = subset_df.groupby([overlay_by])[y_col].median().astype(int)
			
			caption = subset_df.groupby([overlay_by])[y_col].median().apply(pd.to_numeric, errors='coerce')

			caption_df = caption.reset_index()
			caption_df.columns = ['label', 'median']
			caption = caption_df.to_string(index=False)

		elif caption_func == 'max' and overlay_by:
			# caption = subset_df.groupby([overlay_by])[y_col].max().astype(int)
			caption = subset_df.groupby([overlay_by])[y_col].max().apply(pd.to_numeric, errors='coerce')
			caption_df = caption.reset_index()
			caption_df.columns = ['label', 'max']
			caption = caption_df.to_string(index=False)
			
		elif caption_func == 'min' and overlay_by:
			# caption = subset_df.groupby([overlay_by])[y_col].min().astype(int)
			caption = subset_df.groupby([overlay_by])[y_col].min().apply(pd.to_numeric, errors='coerce')
			caption_df = caption.reset_index()
			caption_df.columns = ['label', 'min']
			caption = caption_df.to_string(index=False)
		
		if num_cols == None or num_cols>15:
			caption_fontsize=6
		else:
			caption_fontsize=11
			
		if caption_func:
			# Add caption box inside the plot
			ax.text(
			0.95, 0.95, caption,
			transform=ax.transAxes,
			clip_on=True,
			fontsize=caption_fontsize,
			verticalalignment='top',
			horizontalalignment='right',
			bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.5)
			)

	def scatter_with_agg_line(self, x_col, y_col, title='Scatter Plot', overlay_by=None, loop_by=None,
							  agg_func=None, caption_func=None, split_by_row=None, split_by_col=None,
							  ref_lines_x=None, ref_lines_y=None,log_scale=None, output_path=None,save_ppt=None):
		if isinstance(y_col, str):
			if x_col not in self.df.columns or y_col not in self.df.columns:
				raise ValueError(f"Columns {x_col} and {y_col} must be in the dataframe")

			
		if loop_by and loop_by in self.df.columns:
			items = self.df[f'{loop_by}'].unique()
		else:
			items = [None]

		for item in items:
			if loop_by and loop_by in self.df.columns:
				loop_df = self.df[self.df[f'{loop_by}'] == item]
				
			else:
				loop_df = self.df.copy()

			#Define title based on loop_by
			if loop_by and loop_by in self.df.columns:
				loop_df = loop_df[loop_df[loop_by] == item]
				temp_title = f"{title} {item}"
				temp_output_path = output_path.replace('.jpg', f'_{item}.jpg') if output_path else None
			else:
				temp_title = title
				temp_output_path = output_path
			
			#Search for unique cols or rows to be splitted into subplots
			row_categories = loop_df[split_by_row].unique() if split_by_row and split_by_row in loop_df.columns else [None]
			col_categories = loop_df[split_by_col].unique() if split_by_col and split_by_col in loop_df.columns else [None]

			y_cols = y_col


			num_cols = len(col_categories)
			
			#Single y col
			if isinstance(y_cols, str):
				num_rows = len(row_categories)
				fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})

			#Multipe y col
			if isinstance(y_cols, list):
				num_rows = len(y_cols)
				row_val = None
				fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 10), sharex=True, sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})
										# Fix inverted y-axis if it happens

			#Define axes
			if num_rows == 1 and num_cols == 1:
				axes = [[axes]]
			elif num_rows == 1:
				axes = [axes]
			elif num_cols == 1:
				axes = [[ax] for ax in axes]

			#User-defined color codes
			overlay_categories = sorted(loop_df[overlay_by].unique())
			point_colors = ['#7EB9FF','#FF98A6','#9ED7A3','#D29AFF','#FFB17D','#0FFFBC','#FD96FF','#F0E521','#67FFF7','#FD96FF','#D8EAA2','#7BE1E1','#C9DB82','#ADDCC0','#91B5EC','#CFB0D7','#EBB8A7','#BED183','#9DF2D5','#CFD8E9'] 
			point_palette = dict(zip(overlay_categories, point_colors))

			line_colors = ['#4270DD','#D44958','#3DAE46','#A22BDD','#CC7929','#28B68F','#C529C9','#C4BD2B','#27AEAE','#CF299C','#8EB028','#289ABE','#8DA530','#6ABF8C','#174384','#9051A0','#DC8569','#618830','#29E4A1','#9BA2C7'] 
			line_palette = dict(zip(overlay_categories, line_colors))
				
			#Multipe y col
			if isinstance(y_cols, list):
				row_categories=y_cols
			# Iterate over each row and column category
			for i, row_val in enumerate(row_categories):
				#Multipe y col
				if isinstance(y_cols, list):
					y_col=row_val
					row_val = None
				for j, col_val in enumerate(col_categories):
						
					ax = axes[i][j]
					subset_df = loop_df

					# Filter the DataFrame by the current row and column category if row_val and col_val is not None
					if row_val is not None:
						subset_df = subset_df[subset_df[split_by_row] == row_val]
						
					if col_val is not None:
						subset_df = subset_df[subset_df[split_by_col] == col_val]

					if not subset_df.empty:

						#Single y col
						if isinstance(log_scale, int):
							
							if (log_scale):
								# print(log_scale)
								ax.set_yscale('log')
								
								
						#Multipe y col
						if isinstance(log_scale, list):
							# Set y-axis scale based on row index
							if log_scale and i < len(log_scale):
								ax.set_yscale('log' if log_scale[i] == 1 else 'linear')
			
						# Create shared legend handles
						legend_handles = [
							Patch(facecolor=point_palette[cat], edgecolor='black', label=cat)
							for cat in overlay_categories
						]
						fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), title=overlay_by)
						sns.scatterplot(data=subset_df, x=x_col, y=y_col, ax=ax, hue=overlay_by, palette=point_palette, s=5, edgecolor='none', linewidth=0.3, legend=False)

						#Aggregation Line
						if (agg_func == 'min' or agg_func == 'max' or agg_func == 'median'or agg_func == 'mean') and overlay_by:
							# Convert y_col(s) to numeric, keeping NaNs
							subset_df.loc[:, y_col] = pd.to_numeric(subset_df[y_col], errors='coerce')
							# Group and aggregate (NaNs will be skipped in aggregation)
							agg_df = subset_df.groupby([x_col, overlay_by])[y_col].agg(agg_func).reset_index()

						if (agg_func):
							#sns.lineplot(data=agg_df, x=x_col, y=y_col, ax=ax, hue=overlay_by, linewidth=2, legend=False)
							crt_plot=sns.lineplot(data=agg_df, x=x_col, y=y_col, ax=ax, hue=overlay_by, palette = line_palette, linewidth=2, legend=False)
							#sns.move_legend(crt_plot, "upper left", bbox_to_anchor=(1, 1))
							# crt_plot.legend(fontsize=6)
							# crt_plot.legend(loc='upper right', fontsize=6)

							self.add_caption_box(subset_df, overlay_by, y_col, ax, num_cols, caption_func)

						#Title
						if j == num_cols - 1 and split_by_row:
							ax.text(1.05, 0.5, f"{row_val}", transform=ax.transAxes, fontsize=9, va='center', ha='left')
						if i == 0 and split_by_col:
							ax.set_title(f"{split_by_col}={col_val}")
							ax.set_title(f"{col_val}")
						
						#Reference line
						if ref_lines_x:
							for x_val, text in ref_lines_x:
								ax.axvline(x=x_val, linestyle='--', color='red', alpha=0.7)
								ax.text(x_val, ax.get_ylim()[1] * 0.95, text, color='red', fontsize=10, ha='right')

						# Reference lines for each y_col
						current_ref_lines_y = []
						if isinstance(ref_lines_y, dict) and y_col in ref_lines_y:
							current_ref_lines_y = ref_lines_y[y_col]
						elif isinstance(ref_lines_y, list):
							current_ref_lines_y = ref_lines_y

						# Plot the reference lines
						for y_val, text in current_ref_lines_y:
							ax.axhline(y=y_val, linestyle='--', color='blue', alpha=0.7)
							ax.text(ax.get_xlim()[0] * 1.05, y_val, text, color='blue', fontsize=10, va='bottom')


						# if j==0:
						# 	ax.set_ylim(sorted(ax.get_ylim()))
						ax.grid(True)

		
			plt.suptitle(temp_title, fontsize=16)
			plt.tight_layout(rect=[0, 0, 0.95, 0.95])
			
			#unmask this to show plot on ipynb
			# plt.show() 

			#unmask this to save plot into output folder
			if (temp_output_path):
				plt.savefig(temp_output_path)
			else:
				plt.show()

			plt.close()
		
	def boxplot(self, x_col, y_col, title='Box Plot', overlay_by=None, loop_by=None, caption_func=None,
							  ref_lines_x=None, ref_lines_y=None,output_path=None,save_ppt=None,log_scale=None):
		if x_col not in self.df.columns or y_col not in self.df.columns:
			raise ValueError(f"Columns {x_col} and {y_col} must be in the dataframe")
		
		if loop_by and loop_by in self.df.columns:
			items = self.df[f'{loop_by}'].unique()
		else:
			items = [None]

		for item in items:
			subset_df = self.df.copy()

			if loop_by and loop_by in self.df.columns:
				subset_df = subset_df[subset_df[loop_by] == item]
				temp_title = f"{title} {item}"
				temp_output_path = output_path.replace('.jpg', f'_{item}.jpg') if output_path else None
			else:
				temp_title = title
				temp_output_path = output_path

			# User-defined color palette
			overlay_categories = sorted(subset_df[overlay_by].unique())
			line_colors = ['#4270DD','#D44958','#3DAE46','#A22BDD','#CC7929','#28B68F','#C529C9','#C4BD2B','#27AEAE','#CF299C','#8EB028','#289ABE','#8DA530','#6ABF8C','#174384','#9051A0','#DC8569','#618830','#29E4A1','#9BA2C7'] 
			line_palette = dict(zip(overlay_categories, line_colors))

			fig, ax = plt.subplots(figsize=(10, 6))
			ax=sns.boxplot(data=subset_df, x=x_col, y=y_col, hue=overlay_by,fill=False, palette=line_palette, ax=ax, legend=False)

			
			# Rotate y-axis tick labels
			ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


			# Create custom legend handles
			legend_handles = [
				Patch(facecolor=line_palette[cat], edgecolor='black', label=cat)
				for cat in overlay_categories
			]

			# Add shared legend outside the plot
			fig.legend(
				handles=legend_handles,
				loc='center left',
				bbox_to_anchor=(1.01, 0.95),
				fontsize=8,
				title=overlay_by,
				borderaxespad=0.1,
				borderpad=0.2,
				labelspacing=0.3,
				handlelength=1.0,
				handletextpad=0.3,
				frameon=True

			)

			
			self.add_caption_box(subset_df, overlay_by, y_col, ax, None, caption_func)

			if (log_scale):
				plt.yscale('log')
			

			if ref_lines_x:
				for x_val, text in ref_lines_x:
					ax.axvline(x=x_val, linestyle='--', color='red', alpha=0.7)
					ax.text(x_val, ax.get_ylim()[1] * 0.95, text, color='red', fontsize=10, ha='right')
			if ref_lines_y:
				for y_val, text in ref_lines_y:
					ax.axhline(y=y_val, linestyle='--', color='blue', alpha=0.7)
					ax.text(ax.get_xlim()[0] * 1.05, y_val, text, color='blue', fontsize=10, va='bottom')


			plt.subplots_adjust(right=0.8)
			plt.suptitle(temp_title, fontsize=16)
			plt.tight_layout(rect=[0, 0, 0.95, 0.95])


			#unmask this to show plot on ipynb
			# plt.show() 

			#unmask this to save plot into output folder
			if (temp_output_path):
				plt.savefig(temp_output_path)
			else:
				plt.show()
			
			plt.close()

	def boxplot_with_split(self, x_col, y_col, title='Box Plot', split_by_row=None, split_by_col=None, overlay_by=None, loop_by=None, caption_func=None,
								ref_lines_x=None, ref_lines_y=None,output_path=None,save_ppt=None,log_scale=None):
		if x_col not in self.df.columns or y_col not in self.df.columns:
			raise ValueError(f"Columns {x_col} and {y_col} must be in the dataframe")
		
		if loop_by and loop_by in self.df.columns:
			items = self.df[f'{loop_by}'].unique()
		else:
			items = [None]

		for item in items:
			loop_df = self.df.copy()

			if loop_by and loop_by in self.df.columns:
				loop_df = loop_df[loop_df[loop_by] == item]
				temp_title = f"{title} {item}"
				temp_output_path = output_path.replace('.jpg', f'_{item}.jpg') if output_path else None
			else:
				temp_title = title
				temp_output_path = output_path


			
			#Search for unique cols or rows to be splitted into subplots
			row_categories = loop_df[split_by_row].unique() if split_by_row and split_by_row in loop_df.columns else [None]
			col_categories = loop_df[split_by_col].unique() if split_by_col and split_by_col in loop_df.columns else [None]
			num_cols = len(col_categories)
			num_rows = len(row_categories)
			fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})


			#Define axes
			if num_rows == 1 and num_cols == 1:
				axes = [[axes]]
			elif num_rows == 1:
				axes = [axes]
			elif num_cols == 1:
				axes = [[ax] for ax in axes]



			# User-defined color palette
			overlay_categories = sorted(loop_df[overlay_by].unique())
			line_colors = ['#4270DD','#D44958','#3DAE46','#A22BDD','#CC7929','#28B68F','#C529C9','#C4BD2B','#27AEAE','#CF299C','#8EB028','#289ABE','#8DA530','#6ABF8C','#174384','#9051A0','#DC8569','#618830','#29E4A1','#9BA2C7'] 
			line_palette = dict(zip(overlay_categories, line_colors))
				
			# Iterate over each row and column category
			for i, row_val in enumerate(row_categories):
				for j, col_val in enumerate(col_categories):
					ax = axes[i][j]
					subset_df = loop_df
					

					# Filter the DataFrame by the current row and column category if row_val and col_val is not None
					if row_val is not None:
						subset_df = subset_df[subset_df[split_by_row] == row_val]
						
					if col_val is not None:
						subset_df = subset_df[subset_df[split_by_col] == col_val]

					if not subset_df.empty:
						if (log_scale):
							# print(log_scale)
							ax.set_yscale('log')
						'''							
						# User-defined color palette
						overlay_categories = sorted(subset_df[overlay_by].unique())
						line_colors = ['#4270DD','#D44958','#3DAE46','#A22BDD','#CC7929','#28B68F','#C529C9','#C4BD2B','#27AEAE','#CF299C','#8EB028','#289ABE','#8DA530','#6ABF8C','#174384','#9051A0','#DC8569','#618830','#29E4A1','#9BA2C7'] 
						line_palette = dict(zip(overlay_categories, line_colors))
						'''	
						
						ax=sns.boxplot(data=subset_df, x=x_col, y=y_col, hue=overlay_by,fill=False, palette=line_palette, ax=ax, legend=False)

						
						# Rotate y-axis tick labels
						xticks = ax.get_xticks()
						ax.set_xticks(xticks)
						ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

						# Create shared legend handles
						legend_handles = [
							Patch(facecolor=line_palette[cat], edgecolor='black', label=cat)
							for cat in overlay_categories
						]
						fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), title=overlay_by)
						sns.scatterplot(data=subset_df, x=x_col, y=y_col, ax=ax, hue=overlay_by, palette=line_palette, s=5, edgecolor='none', linewidth=0.3, legend=False)


						#Title
						if j == num_cols - 1 and split_by_row:
							ax.text(1.05, 0.5, f"{row_val}", transform=ax.transAxes, fontsize=9, va='center', ha='left')
						if i == 0 and split_by_col:
							ax.set_title(f"{split_by_col}={col_val}")
							ax.set_title(f"{col_val}")

						
						self.add_caption_box(subset_df, overlay_by, y_col, ax, None, caption_func)

						if (log_scale):
							plt.yscale('log')
						

						if ref_lines_x:
							for x_val, text in ref_lines_x:
								ax.axvline(x=x_val, linestyle='--', color='red', alpha=0.7)
								ax.text(x_val, ax.get_ylim()[1] * 0.95, text, color='red', fontsize=10, ha='right')
						if ref_lines_y:
							for y_val, text in ref_lines_y:
								ax.axhline(y=y_val, linestyle='--', color='blue', alpha=0.7)
								ax.text(ax.get_xlim()[0] * 1.05, y_val, text, color='blue', fontsize=10, va='bottom')

			plt.suptitle(temp_title, fontsize=16)
			plt.tight_layout(rect=[0, 0, 0.95, 0.95])


			#unmask this to show plot on ipynb
			# plt.show() 

			#unmask this to save plot into output folder
			if (temp_output_path):
				print("save")
				plt.savefig(temp_output_path)
			else:
				print("it's trying to show plot")
				plt.show()
			
			plt.close()
