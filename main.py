# initialise db
import sys
import os
import pandas as pd
import re
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from utilities.query.query import BigQueryManager
from utilities.plotting.plotting import DataPlotter

cache=config.gcp_cache()

bq = BigQueryManager("gdw-prod-maf-pilot", r"./utilities/query/templates")

## if output path doesn't exist create it
if not os.path.exists(config.user_input_config["output_path"]):
    os.makedirs(config.user_input_config["output_path"])

    # Get query filter parameters
# split query by filters step, read_point,SU_mode_PB_Type, mode(only for rwb)

distinct_items_cwer_rber_wl = cache.query(f"""SELECT DISTINCT did,progname,step,read_point,blkgrp_prefix,blkgrp,setup_mode,pb_setup FROM gcp_caching WHERE lot IN ({','.join(f"'{item}'" for item in config.user_input_config['lot'])}) and step IN ({','.join(f"'{item}'" for item in config.user_input_config['step'])}) and data_type='dyn_read' """)
distinct_items_rwb_wl = cache.query(f""" SELECT DISTINCT did,progname,step,read_point,blkgrp_prefix,blkgrp,setup_mode,pb_setup,mode FROM gcp_caching WHERE lot IN ({','.join(f"'{item}'" for item in config.user_input_config['lot'])}) and step IN ({','.join(f"'{item}'" for item in config.user_input_config['step'])}) and data_type='rwb' """)

df = pd.DataFrame(distinct_items_cwer_rber_wl, columns=["did","progname","step","read_point","blkgrp_prefix","blkgrp","setup_mode","pb_setup"])
df_rwb = pd.DataFrame(distinct_items_rwb_wl, columns=["did","progname","step","read_point","blkgrp_prefix","blkgrp","setup_mode","pb_setup","mode"])

df['pbendpage'] = df['pb_setup'].apply(lambda x: 'AP' if 'AP' in x else 'PB' + re.search(r'\d+', x).group(0))
df_rwb['pbendpage'] = df_rwb['pb_setup'].apply(lambda x: 'AP' if 'AP' in x else 'PB' + re.search(r'\d+', x).group(0))

context_list_avg_cwer = []
context_list_avg_cwer_rber_wl = []
create_lut_functions = []
agg_lut_list = []

for lut_functions in config.user_input_config["lut_functions"]:
    create_lut_functions.append(f"gdw-prod-data.ww_nve_lut.{lut_functions}")
    agg_lut_list.append(f"AVG(min_cwer_{lut_functions}) AS die_avg_cwer_{lut_functions}")

progname = df["progname"].unique()[0]

context_list_avg_cwer.append(
    {
        "file_name": f"avg_cwer_progname-{progname}.csv",
        "context":{
        "source_table": f"gdw-prod-data.ww_nve_maf.n69r_dyn_read_mv2-{progname}",
        "lut_functions": create_lut_functions,
        "filters": {
            "step": config.user_input_config["step"],
            "lot": config.user_input_config["lot"]
        },
        "outer_group_by": ["fid", "step", "lot", "read_point", "cycle_group", "pattern"],
        "outer_aggregates": [
            "MAX(min_bbc) AS max_min_bbc",
            "MAX(min_bbc_hrer) AS max_min_bbc_hrer"
        ] + agg_lut_list
    }}
)

for step in df["step"].unique():
    for read_point in df[df["step"] == step]["read_point"].unique():
        for setup_mode in df[(df["step"] == step) & (df["read_point"] == read_point)]["setup_mode"].unique():
            for pb_setup in df[(df["step"] == step) & (df["read_point"] == read_point) & (df["setup_mode"] == setup_mode)]["pbendpage"].unique():
                filtered_df = df[(df["step"] == step) & (df["read_point"] == read_point) & (df["setup_mode"] == setup_mode) & (df["pbendpage"] == pb_setup)].copy()
                filtered_df.loc[:, 'query_key'] = (
                    filtered_df['blkgrp_prefix'] + '_' +
                    filtered_df['blkgrp'].astype(str) + '_' +
                    filtered_df['setup_mode'].astype(str) + '_' +
                    filtered_df['pb_setup']
                )
                cycle_group = list(filtered_df["query_key"].unique())

                context_list_avg_cwer_rber_wl.append(
                    {
                        "file_name": f"avg_cwer_rber_wl_progname-{progname}_step-{step}_read_point-{read_point}_setup_mode-{setup_mode}_pb_setup-{pb_setup}.csv",
                        "context": {
                        "source_table": f"gdw-prod-data.ww_nve_maf.n69r_dyn_read_mv2-{progname}",
                        "lut_functions": create_lut_functions,
                        "filters": {
                            "step": f"{step}",
                            "lot": config.user_input_config["lot"],
                            "read_point": int(read_point),
                            "cycle_group": cycle_group
                        },
                        "outer_group_by": ["fid", "step", "lot", "read_point", "cycle_group", "pattern", "wl", "page_type",],
                        "outer_aggregates": [
                            "MAX(min_bbc) AS max_min_bbc",
                            "MAX(min_bbc_hrer) AS max_min_bbc_hrer"
                        ] + agg_lut_list
                    }}
                )

context_rwb = []

progname = df_rwb["progname"].unique()[0]

for step in df_rwb["step"].unique():
    for read_point in df_rwb[df_rwb["step"] == step]["read_point"].unique():
        for setup_mode in df_rwb[(df_rwb["step"] == step) & (df_rwb["read_point"] == read_point)]["setup_mode"].unique():
            for pb_setup in df_rwb[(df_rwb["step"] == step) & (df_rwb["read_point"] == read_point) & (df_rwb["setup_mode"] == setup_mode)]["pbendpage"].unique():
                for mode in df_rwb[(df_rwb["step"] == step) & (df_rwb["read_point"] == read_point) & (df_rwb["setup_mode"] == setup_mode) & (df_rwb["pbendpage"] == pb_setup)]["mode"].unique():
                    filtered_df = df_rwb[(df_rwb["step"] == step) & (df_rwb["read_point"] == read_point) & (df_rwb["setup_mode"] == setup_mode) & (df_rwb["pbendpage"] == pb_setup) & (df_rwb["mode"] == mode)].copy()
                    filtered_df.loc[:, 'query_key'] = (
                        filtered_df['blkgrp_prefix'] + '_' +
                        filtered_df['blkgrp'].astype(str) + '_' +
                        filtered_df['setup_mode'].astype(str) + '_' +
                        filtered_df['pb_setup']
                    )
                    cycle_group = list(filtered_df["query_key"].unique())
                    create_agg_list = ["MIN(ESUM) AS ESUM", "MIN(ESUM_E1) AS ESUM_E1"]

                    if (setup_mode == "QLC"):
                        max_edge = 30
                    elif(setup_mode == "TLC"):
                        max_edge = 15
                    elif(setup_mode == "SLC"):
                        max_edge = 2
                    else:
                        max_edge=30
                    
                    for edge in range(max_edge):
                        create_agg_list.append(f"MIN(E{edge}) AS E{edge}")
                        if edge / 2:
                            create_agg_list.append(f"MIN(V{edge/2}) AS V{edge/2}")
                    

                    context_rwb.append(
                        {
                            "file_name": f"avg_rwb_progname-{progname}_step-{step}_read_point-{read_point}_setup_mode-{setup_mode}_pb_setup-{pb_setup}_mode-{mode}.csv",
                            "context":{
                                "source_table": f"gdw-prod-data.ww_nve_maf.n69r_rwb_mv2-{progname}",
                                "max_edge": max_edge,  # or however many edge values you expect
                                "filters": {
                                    "step": f"{step}",
                                    "lot": config.user_input_config["lot"],
                                    "read_point": int(read_point),
                                    "cycle_group": cycle_group,
                                    "mode": mode
                                },
                                "outer_group_by": [
                                    "fid", "lot", "session", "bits_per_cell", "step", "read_point",
                                    "pattern", "cycle_group", "time_stamp", "wl", "mode"
                                ],
                                "outer_aggregates": create_agg_list
                            }}
                    )


import threading
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class Decoder:
    def __init__(self, df):
        self.df = df

    def decode_cycle_group(self):
        # Split 'cycle_group' into multiple columns
        self.df[['blkgrp_type', 'blkgrp', 'runmode', 'pbcondition']] = self.df['cycle_group'].str.split('_', expand=True)

        # Extract 'pbendpage' based on 'pbcondition'
        self.df['pbendpage'] = self.df['pbcondition'].apply(
            lambda x: 'AP' if 'AP' in x else 'PB' + re.search(r'\d+', x).group(0)
        )
        return self.df
    def concatenate_columns(self, concat_cols):
        # Generate dynamic column name
        new_col_name = '_'.join(concat_cols)

        # Concatenate columns as strings
        self.df[new_col_name] = self.df[concat_cols].astype(str).agg('_'.join, axis=1)

        # Return unique values
        return self.df
	

def rwb_summary(df, output_path, file_name):
    filename = file_name.replace('.csv', '') 
    # df = pd.read_csv(os.path.join(output_path, file_name), keep_default_na=False, low_memory=False)

    # Define group_by and agg_info
    group_by = ['runmode','mode']
    agg_info = ['blkgrp', 'read_point']

    # Identify columns matching 'min_E\d+' and not containing 'ESUM'
    min_cols = [col for col in df.columns if re.match(r'^E\d+$', col) and 'ESUM' not in col]

    # Convert columns to numeric, forcing errors to NaN
    df[min_cols] = df[min_cols].apply(pd.to_numeric, errors='coerce')


    # Compute the row-wise minimum value and the corresponding column
    df['min_value'] = df[min_cols].min(axis=1, skipna=True)
    df['min_column'] = df[min_cols].idxmin(axis=1)

    # Group by and find the minimum value per group
    group_min = df.groupby(group_by)['min_value'].min().reset_index()

    # Merge to find rows where the min_value matches the group min
    merged_df = pd.merge(df, group_min, on=group_by + ['min_value'], how='inner')


    # Define aggregation logic for each column in agg_info
    agg_dict = {}
    for col in agg_info:
        agg_dict[col] = lambda x: x.iloc[0] if x.nunique() == 1 else ','.join(map(str, sorted(set(x))))

    # Add min_column and min_value to group_by for final aggregation
    summary_df = merged_df.groupby(group_by + ['min_column', 'min_value']).agg(agg_dict).reset_index()

    if not os.path.exists(f"{output_path}\\rwb_summary"):
        os.makedirs(f"{output_path}\\rwb_summary")
    # Display the final result
    summary_df.to_csv(f"{output_path}\\rwb_summary\\rwb_summary_file-{filename}.csv", index=False)

def concat_rwb_summaries(output_path):
    summary_files = glob.glob(os.path.join(f"{output_path}\\rwb_summary", '*rwb_summary*.csv'))

    # Print the filenames that are being combined
    print("Combining the following files:")
    for file in summary_files:
        print(file)

    # Read and combine all summary files into a single DataFrame
    df = pd.concat([pd.read_csv(file, keep_default_na=False) for file in summary_files])

    # Define group_by and agg_info
    group_by = ['runmode','mode']
    agg_info = ['blkgrp', 'read_point']

    # Group by runmode and mode and get the minimum value
    result = df.groupby(['runmode', 'mode'])['min_value'].min().reset_index()

    # Merge to find rows where the min_value matches the group min
    merged_df = pd.merge(df, result, on=group_by + ['min_value'], how='inner')

    # Define aggregation logic for each column in agg_info
    agg_dict = {}
    for col in agg_info:
        agg_dict[col] = lambda x: x.iloc[0] if x.nunique() == 1 else ','.join(map(str, sorted(set(x))))

    # Add min_column and min_value to group_by for final aggregation
    summary_df = merged_df.groupby(group_by + ['min_column', 'min_value']).agg(agg_dict).reset_index()

    summary_df.to_csv(f"{output_path}\\rwb_summary_final.csv", index=False)

def pass_fail_summary(df,output_path,file_name,luts):
    filename = file_name.replace('.csv', '') 
    # df = pd.read_csv(os.path.join(output_path, file_name), keep_default_na=False)

    for lut in luts:
    # Add result column
        df["result"] = df[f"die_avg_cwer_{lut}"].apply(lambda x: "Fail" if x > 4e-12 else "Pass")

        # Ensure blkgrp is numeric
        df["blkgrp"] = pd.to_numeric(df["blkgrp"], errors="coerce")

        qlc_condition = 3000
        tlc_condition = 10000
        slc_condition = 200000

        # Apply filtering conditions based on runmode
        conditions = (
            ((df["runmode"] == "QLC") & (df["blkgrp"] <= qlc_condition)) |
            ((df["runmode"] == "TLC") & (df["blkgrp"] <= tlc_condition)) |
            ((df["runmode"] == "SLC") & (df["blkgrp"] <= slc_condition))
        )
        filtered_df = df[conditions]
    
        # Determine if each fid has any fail in the filtered data
        fid_fail_status = filtered_df.groupby(["runmode", "fid"])["result"].apply(lambda x: "Fail" if "Fail" in x.values else "Pass").reset_index()


        # Calculate fail rate per runmode
        fail_rate = fid_fail_status.groupby("runmode")["result"].apply(lambda x: (x == "Fail").sum() / len(x) * 100).reset_index(name="fail_rate")

        # Format fail rate as percentage string
        fail_rate["fail_rate"] = fail_rate["fail_rate"].map("{:.1f}%".format)

        # Add condition description per runmode
        condition_map = {
            "QLC": f"blkgrp <= {qlc_condition}",
            "TLC": f"blkgrp <= {tlc_condition}",
            "SLC": f"blkgrp <= {slc_condition}"
        }
        fail_rate["condition"] = fail_rate["runmode"].map(condition_map)


        if not os.path.exists(f"{output_path}\\pass_fail_summary"):
            os.makedirs(f"{output_path}\\pass_fail_summary")
        fail_rate.to_csv(f"{output_path}\\pass_fail_summary\\fail_rate_file_name_{lut}.csv", index=False)

def plot_E0(columns_to_stack, df, index_columns,file,output_path):
    ######## E0 
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'E0', item)]
    print("E0 cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)
        
        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='E0 vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138,'')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/E0_by_wl_{file}.jpg"
        )
    else:
        print("Skip stacking.")
    

def plot_E1_onwards(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'E\d+', item) and not re.fullmatch(r'E0', item) and not re.fullmatch(r'ESUM.*', item)]
    print("E1++ cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)
        
        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='EDGE MARGIN vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138,'')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/edge_margin_by_wl_{file}.jpg"
        )
    else:
        print("Skip stacking.")

def plot_all_edges(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'E\d+', item) and not re.fullmatch(r'ESUM.*', item)]
    print("E1++ cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)
        
        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='EDGE MARGIN vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138,'')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/edge_margin_by_wl_{file}.jpg"
        )
    else:
        print("Skip stacking.")

    
def plot_V0(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'V0', item)]
    print("V0 cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)

        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='V0 vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138, 'First MLC WL')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/V0_by_wl_{file}.jpg"
        )

    else:
        print("Skip stacking.")


def plot_V1_onwards(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'V\d+', item) and not re.fullmatch(r'V0', item)] 
    print("V1++ cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)

        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='VALLEY MARGIN vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138, 'First MLC WL')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/valley_by_wl_{file}.jpg"
        )

    else:
        print("Skip stacking.")

def plot_all_valleys(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'V\d+', item)] 
    print("V1++ cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)

        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='VALLEY MARGIN vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138, 'First MLC WL')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/valley_by_wl_{file}.jpg"
        )

    else:
        print("Skip stacking.")


def plot_ESUMS(columns_to_stack, df, index_columns,file,output_path):
    filtered_columns_to_stack = [item for item in columns_to_stack if re.fullmatch(r'ESUM', item)]
    print("ESUMS cols to stack:", filtered_columns_to_stack)
    if filtered_columns_to_stack:
        # Perform the stacking operation
        stacked_df = df.set_index(index_columns)[filtered_columns_to_stack].stack().reset_index(name='rwb')
        stacked_df.rename(columns={'level_16': 'edge'}, inplace=True)
        stacked_df.loc[:, 'rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)

        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title='ESUMS vs WL',
            split_by_row='mode',
            split_by_col='edge',
            # ref_lines_x=[(138, 'First MLC WL')],
            ref_lines_y=[(0, '')],
            output_path=f"{output_path}/saved_plots/ESUMS_by_wl_{file}.jpg"
        )

    else:
        print("Skip stacking.")

def plot_cwer(df, file_name, output_path, luts):
    #filtering df to maintain scale of graph
    filtered_df = df
    filtered_df.head()
    plotter = DataPlotter(filtered_df)
    luts = ["n69r_smi_2259xt3_652b_1h2s_soft_lut_v1","b68s_phison_e31t_mode18_336B_1h2s_soft_lut_v2"]
    # luts = ["n69r_smi_2259xt3_652b_1h2s_soft_lut_v1","b68s_phison_e31t_mode18_336B_1h2s_soft_lut_v2"]
    for lut in luts:
        y_col= "die_avg_cwer_" + lut
        plotter.boxplot_with_split(
            x_col='cycle_group',
            y_col = y_col,
            overlay_by='step_read_point',
            loop_by = '',
            title=f'DIE CWER {lut.upper()}',
            split_by_row='',
            split_by_col='step_read_point',
            # ref_lines_x=[(138, 'First MLC WL')],
            ref_lines_y=[(1E-8, 'CWER Threshold')],
            log_scale=1,
            output_path=f"{output_path}/saved_plots/die_cwer_{lut}_{file_name}.jpg"
        )

def plot_cwer_wl(df, file_name, output_path):
    # Replace empty strings and whitespace-only strings with pd.NA
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # check_na_col = 'max_min_bbc_hrer'
    # # Count the number of rows where the 'hrer' column contains NaN values
    # na_row_count = df[check_na_col].isna().sum()
    # print(f"Number of rows with NaN in 'max_min_bbc_hrer' column: {na_row_count}")

    # #Drop rows with NaN values. 
    # df = df.dropna(subset=[check_na_col])
    # print(f"Rows Dropped: {na_row_count}")

    plotter = DataPlotter(df)
    plotter.scatter_with_agg_line(
        x_col='wl',
        # y_col='max_min_bbc', #use str for sinlge y cols
        y_col=["max_min_bbc","max_min_bbc_hrer"], #use list for multiple y col
        overlay_by='page_type',
        loop_by='',
        agg_func='max',
        caption_func='max',
        title='RBER/HRER vs WL',
        split_by_row='blkgrp', 
        split_by_col='blkgrp', #Don't care for multiple y col
        # ref_lines_x=[(138, 'First MLC WL')],
        # ref_lines_y=[(183, 'RBER Threshold')],
        # ref_lines_y={'die_avg_cwer_n69r_smi_2259xt3_652b_1h2s_soft_lut_v1': [(4e-12, 'CWER Limit')]}, #use dictionary for multiple y col ref line
        # log_scale=0, #use int for sinlge y col
        log_scale=[0,0], #use list for multiple y col
        output_path=f"{output_path}/saved_plots/cwer_rber_hrer_by_wl_{file_name}.jpg"
    )

def plot_rwb_wrapper(df,file,output_path):
    index_columns = ['fid','lot','session','bits_per_cell','step','read_point','pattern','cycle_group','time_stamp', 'blkgrp_type', 'blkgrp', 'runmode', 'pbcondition','pbendpage','wl','mode']

    # Identify columns where all values are NaN
    all_nan_columns = df.columns[df.isna().all()]

    # Determine columns to stack: exclude group by columns and all-NaN columns
    columns_to_stack = [col for col in df.columns if col not in index_columns and col not in all_nan_columns]
    print("cols to stack:", columns_to_stack)

    #Plot
    is_qlc_or_tlc = "setup_mode-QLC" in file or "setup_mode-TLC" in file
    is_mwl_or_swl = "mode-MWL" in file or "mode-SWL" in file

    if is_qlc_or_tlc and not is_mwl_or_swl:
        plot_E0(columns_to_stack, df, index_columns,file,output_path)
        plot_E1_onwards(columns_to_stack, df, index_columns,file,output_path)
        plot_V0(columns_to_stack, df, index_columns,file,output_path)
        plot_V1_onwards(columns_to_stack, df, index_columns,file,output_path)
        plot_ESUMS(columns_to_stack, df, index_columns,file,output_path)
    else:
        plot_all_edges(columns_to_stack, df, index_columns,file,output_path)
        plot_all_valleys(columns_to_stack, df, index_columns,file,output_path)
        plot_ESUMS(columns_to_stack, df, index_columns,file,output_path)

# Set the maximum number of concurrent threads
MAX_CONCURRENT_THREADS = 10
semaphore = threading.Semaphore(MAX_CONCURRENT_THREADS)

def process_context(entry):
    with semaphore:
        file_name = entry["file_name"]
        context = entry["context"]
        
        logging.info(f"Thread started for: {file_name}")
        
        try:
            # if "cwer" in file_name:
            #     results = bq.run_template_query("cwer_rber_hrer.sql", context)
            # else:
            #     results = bq.run_template_query("rwb.sql", context)
            
            output_path = os.path.join(config.user_input_config["output_path"], file_name)
            # bq.save_to_csv(results, output_path)
            df = pd.read_csv(output_path, keep_default_na=False, low_memory=False)

            #Decode cycle group
            decoder = Decoder(df)
            df = decoder.decode_cycle_group()
            df = decoder.concatenate_columns(['step', 'read_point'])

            if not os.path.exists(os.path.join(config.user_input_config["output_path"], "saved_plots")):
                os.makedirs(os.path.join(config.user_input_config["output_path"], "saved_plots"), exist_ok=True)

            if "cwer" in file_name and "wl" not in file_name:
                pass_fail_summary(df,config.user_input_config["output_path"], file_name, config.user_input_config["lut_functions"])
                plot_cwer(df,file_name,config.user_input_config["output_path"],config.user_input_config["lut_functions"])
            
            if "cwer" in file_name and "wl" in file_name:
                plot_cwer_wl(df,file_name,config.user_input_config["output_path"])
            
            if "rwb" in file_name:
                rwb_summary(df,config.user_input_config["output_path"],file_name)
                plot_rwb_wrapper(df,file_name,config.user_input_config["output_path"])
            
            logging.info(f"Saved: {file_name}")
        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

# Combine all context entries
all_contexts = context_list_avg_cwer + context_list_avg_cwer_rber_wl + context_rwb

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 10  # Adjust based on your system's capacity

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_context, entry) for entry in all_contexts]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logging.error(f"Thread failed: {e}")

# # Create and start threads
# threads = []
# for entry in all_contexts:
#     thread = threading.Thread(target=process_context, args=(entry,))
#     thread.start()
#     threads.append(thread)

# # Wait for all threads to complete
# for thread in threads:
#     thread.join()

if "rwb_summary_final.csv" not in os.listdir(config.user_input_config["output_path"]) and os.path.exists(f"""{config.user_input_config["output_path"]}\\rwb_summary"""):
    concat_rwb_summaries(config.user_input_config["output_path"])
