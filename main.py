import sys
import os
import pandas as pd
import re
import glob
import threading
import logging
import random
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from utilities.query.query import BigQueryManager
from utilities.plotting.plotting import DataPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DataAnalyzer:
    """Main class for handling data analysis pipeline"""
    
    def __init__(self):
        self.config = config.user_input_config
        self.cache = config.gcp_cache()
        self.bq = BigQueryManager("gdw-prod-maf-pilot", r"./utilities/query/templates")
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories if they don't exist"""
        dirs = [
            self.config["output_path"],
            f"{self.config['output_path']}/saved_plots",
            f"{self.config['output_path']}/rwb_summary",
            f"{self.config['output_path']}/pass_fail_summary"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_distinct_items(self, data_type: str, additional_cols: List[str] = None) -> pd.DataFrame:
        """Get distinct items from cache based on data type"""
        base_cols = ["did", "progname", "step", "read_point", "blkgrp_prefix", "blkgrp", "setup_mode", "pb_setup"]
        cols = base_cols + (additional_cols or [])
        
        query = f"""
        SELECT DISTINCT {','.join(cols)}
        FROM gcp_caching 
        WHERE lot IN ({','.join(f"'{item}'" for item in self.config['lot'])}) 
        AND step IN ({','.join(f"'{item}'" for item in self.config['step'])}) 
        AND data_type='{data_type}'
        """
        
        return pd.DataFrame(self.cache.query(query), columns=cols)
    
    def create_pbendpage_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pbendpage column based on pb_setup"""
        df['pbendpage'] = df['pb_setup'].apply(
            lambda x: 'AP' if 'AP' in x else 'PB' + re.search(r'\d+', x).group(0)
        )
        return df
    
    def generate_cwer_contexts(self, df: pd.DataFrame) -> List[Dict]:
        """Generate context configurations for CWER queries"""
        contexts = []
        progname = df["progname"].unique()[0]
        
        # LUT functions setup
        lut_functions = [f"gdw-prod-data.ww_nve_lut.{lut}" for lut in self.config["lut_functions"]]
        agg_lut_list = [f"AVG(min_cwer_{lut}) AS die_avg_cwer_{lut}" for lut in self.config["lut_functions"]]
        
        # Overall average CWER context
        contexts.append({
            "file_name": f"avg_cwer_progname-{progname}.csv",
            "context": {
                "source_table": f"gdw-prod-data.ww_nve_maf.n69r_dyn_read_mv2-{progname}",
                "lut_functions": lut_functions,
                "filters": {
                    "step": self.config["step"],
                    "lot": self.config["lot"]
                },
                "outer_group_by": ["fid", "step", "lot", "read_point", "cycle_group", "pattern"],
                "outer_aggregates": [
                    "MAX(min_bbc) AS max_min_bbc",
                    "MAX(min_bbc_hrer) AS max_min_bbc_hrer"
                ] + agg_lut_list
            }
        })
        
        # Detailed CWER contexts by combinations
        group_cols = ["step", "read_point", "setup_mode", "pbendpage"]
        for combination in self._get_unique_combinations(df, group_cols):
            filtered_df = self._filter_dataframe(df, group_cols, combination)
            cycle_group = self._create_query_key(filtered_df)
            
            contexts.append({
                "file_name": f"avg_cwer_rber_wl_progname-{progname}_" + 
                           "_".join(f"{col}-{val}" for col, val in zip(group_cols, combination)) + ".csv",
                "context": {
                    "source_table": f"gdw-prod-data.ww_nve_maf.n69r_dyn_read_mv2-{progname}",
                    "lut_functions": lut_functions,
                    "filters": {
                        "step": combination[0],
                        "lot": self.config["lot"],
                        "read_point": int(combination[1]),
                        "cycle_group": cycle_group
                    },
                    "outer_group_by": ["fid", "step", "lot", "read_point", "cycle_group", "pattern", "wl", "page_type"],
                    "outer_aggregates": [
                        "MAX(min_bbc) AS max_min_bbc",
                        "MAX(min_bbc_hrer) AS max_min_bbc_hrer"
                    ] + agg_lut_list
                }
            })
        
        return contexts
    
    def generate_rwb_contexts(self, df: pd.DataFrame) -> List[Dict]:
        """Generate context configurations for RWB queries"""
        contexts = []
        progname = df["progname"].unique()[0]
        group_cols = ["step", "read_point", "setup_mode", "pbendpage", "mode"]
        
        for combination in self._get_unique_combinations(df, group_cols):
            filtered_df = self._filter_dataframe(df, group_cols, combination)
            cycle_group = self._create_query_key(filtered_df)
            
            # Determine max_edge based on setup_mode
            setup_mode = combination[2]
            max_edge = {"QLC": 30, "TLC": 15, "SLC": 2}.get(setup_mode, 30)
            
            # Create aggregation list
            agg_list = ["MIN(ESUM) AS ESUM", "MIN(ESUM_E1) AS ESUM_E1"]
            for edge in range(max_edge):
                agg_list.append(f"MIN(E{edge}) AS E{edge}")
                if edge % 2 == 0 and edge > 0:  # Fixed division check
                    agg_list.append(f"MIN(V{edge//2}) AS V{edge//2}")
            
            contexts.append({
                "file_name": f"avg_rwb_progname-{progname}_" + 
                           "_".join(f"{col}-{val}" for col, val in zip(group_cols, combination)) + ".csv",
                "context": {
                    "source_table": f"gdw-prod-data.ww_nve_maf.n69r_rwb_mv2-{progname}",
                    "max_edge": max_edge,
                    "filters": {
                        "step": combination[0],
                        "lot": self.config["lot"],
                        "read_point": int(combination[1]),
                        "cycle_group": cycle_group,
                        "mode": combination[4]
                    },
                    "outer_group_by": [
                        "fid", "lot", "session", "bits_per_cell", "step", "read_point",
                        "pattern", "cycle_group", "time_stamp", "wl", "mode"
                    ],
                    "outer_aggregates": agg_list
                }
            })
        
        return contexts
    
    def _get_unique_combinations(self, df: pd.DataFrame, columns: List[str]) -> List[tuple]:
        """Get unique combinations of specified columns"""
        combinations = []
        if not columns:
            return combinations
            
        def generate_combinations(df, cols, current_combo=[]):
            if not cols:
                combinations.append(tuple(current_combo))
                return
            
            col = cols[0]
            remaining_cols = cols[1:]
            
            # Filter dataframe based on current combination
            if current_combo:
                mask = pd.Series(True, index=df.index)
                for i, val in enumerate(current_combo):
                    mask &= (df.iloc[:, df.columns.get_loc(columns[i])] == val)
                filtered_df = df[mask]
            else:
                filtered_df = df
            
            for value in filtered_df[col].unique():
                generate_combinations(filtered_df, remaining_cols, current_combo + [value])
        
        generate_combinations(df, columns)
        return combinations
    
    def _filter_dataframe(self, df: pd.DataFrame, columns: List[str], values: tuple) -> pd.DataFrame:
        """Filter dataframe by column-value pairs"""
        mask = pd.Series(True, index=df.index)
        for col, val in zip(columns, values):
            mask &= (df[col] == val)
        return df[mask].copy()
    
    def _create_query_key(self, df: pd.DataFrame) -> List[str]:
        """Create query key from filtered dataframe"""
        df['query_key'] = (
            df['blkgrp_prefix'] + '_' +
            df['blkgrp'].astype(str) + '_' +
            df['setup_mode'].astype(str) + '_' +
            df['pb_setup']
        )
        return list(df["query_key"].unique())


class DataProcessor:
    """Handles data processing and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def decode_cycle_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decode cycle group into separate columns"""
        df[['blkgrp_type', 'blkgrp', 'runmode', 'pbcondition']] = df['cycle_group'].str.split('_', expand=True)
        df['pbendpage'] = df['pbcondition'].apply(
            lambda x: 'AP' if 'AP' in x else 'PB' + re.search(r'\d+', x).group(0)
        )
        return df
    
    def concatenate_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Concatenate specified columns"""
        new_col_name = '_'.join(cols)
        df[new_col_name] = df[cols].astype(str).agg('_'.join, axis=1)
        return df
    
    def rwb_summary(self, df: pd.DataFrame, output_path: str, file_name: str) -> None:
        """Generate RWB summary"""
        filename = file_name.replace('.csv', '')
        group_by = ['runmode', 'mode']
        agg_info = ['blkgrp', 'read_point']
        
        # Find E columns (excluding ESUM)
        min_cols = [col for col in df.columns if re.match(r'^E\d+$', col) and 'ESUM' not in col]
        df[min_cols] = df[min_cols].apply(pd.to_numeric, errors='coerce')
        
        # Find minimum values
        df['min_value'] = df[min_cols].min(axis=1, skipna=True)
        df['min_column'] = df[min_cols].idxmin(axis=1)
        
        # Group and aggregate
        group_min = df.groupby(group_by)['min_value'].min().reset_index()
        merged_df = pd.merge(df, group_min, on=group_by + ['min_value'], how='inner')
        
        agg_dict = {col: lambda x: x.iloc[0] if x.nunique() == 1 else ','.join(map(str, sorted(set(x)))) 
                   for col in agg_info}
        
        summary_df = merged_df.groupby(group_by + ['min_column', 'min_value']).agg(agg_dict).reset_index()
        summary_df.to_csv(f"{output_path}/rwb_summary/rwb_summary_file-{filename}.csv", index=False)
    
    def pass_fail_summary(self, df: pd.DataFrame, output_path: str, file_name: str, luts: List[str]) -> None:
        """Generate pass/fail summary"""
        filename = file_name.replace('.csv', '')
        
        conditions_map = {"QLC": 3000, "TLC": 10000, "SLC": 200000}
        
        for lut in luts:
            df["result"] = df[f"die_avg_cwer_{lut}"].apply(lambda x: "Fail" if x > 4e-12 else "Pass")
            df["blkgrp"] = pd.to_numeric(df["blkgrp"], errors="coerce")
            
            # Apply filtering conditions
            conditions = pd.Series(False, index=df.index)
            for mode, limit in conditions_map.items():
                conditions |= ((df["runmode"] == mode) & (df["blkgrp"] <= limit))
            
            filtered_df = df[conditions]
            
            # Calculate fail rates
            fid_fail_status = filtered_df.groupby(["runmode", "fid"])["result"].apply(
                lambda x: "Fail" if "Fail" in x.values else "Pass"
            ).reset_index()
            
            fail_rate = fid_fail_status.groupby("runmode")["result"].apply(
                lambda x: (x == "Fail").sum() / len(x) * 100
            ).reset_index(name="fail_rate")
            
            fail_rate["fail_rate"] = fail_rate["fail_rate"].map("{:.1f}%".format)
            fail_rate["condition"] = fail_rate["runmode"].map(
                {mode: f"blkgrp <= {limit}" for mode, limit in conditions_map.items()}
            )
            
            fail_rate.to_csv(f"{output_path}/pass_fail_summary/fail_rate_file_name_{lut}.csv", index=False)


class PlotManager:
    """Handles all plotting operations"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
    
    def plot_cwer(self, df: pd.DataFrame, file_name: str, luts: List[str]) -> None:
        """Plot CWER data"""
        plotter = DataPlotter(df)
        for lut in luts:
            y_col = f"die_avg_cwer_{lut}"
            plotter.boxplot_with_split(
                x_col='cycle_group',
                y_col=y_col,
                overlay_by='step_read_point',
                loop_by='',
                title=f'DIE CWER {lut.upper()}',
                split_by_row='',
                split_by_col='step_read_point',
                ref_lines_y=[(1E-8, 'CWER Threshold')],
                log_scale=1,
                output_path=f"{self.output_path}/saved_plots/die_cwer_{lut}_{file_name}.jpg"
            )
    
    def plot_cwer_wl(self, df: pd.DataFrame, file_name: str) -> None:
        """Plot CWER vs WL"""
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        plotter = DataPlotter(df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col=["max_min_bbc", "max_min_bbc_hrer"],
            overlay_by='page_type',
            loop_by='',
            agg_func='max',
            caption_func='max',
            title='RBER/HRER vs WL',
            split_by_row='blkgrp',
            split_by_col='blkgrp',
            ref_lines_y=[(0, '')],
            log_scale=[0, 0],
            output_path=f"{self.output_path}/saved_plots/cwer_rber_hrer_by_wl_{file_name}.jpg"
        )
    
    def plot_rwb_data(self, df: pd.DataFrame, file: str) -> None:
        """Plot RWB data with dynamic column detection"""
        index_columns = [
            'fid', 'lot', 'session', 'bits_per_cell', 'step', 'read_point', 'pattern',
            'cycle_group', 'time_stamp', 'blkgrp_type', 'blkgrp', 'runmode', 'pbcondition',
            'pbendpage', 'wl', 'mode'
        ]
        
        # Get columns to stack
        all_nan_columns = df.columns[df.isna().all()]
        columns_to_stack = [col for col in df.columns 
                          if col not in index_columns and col not in all_nan_columns]
        
        plot_configs = [
            ('E0', r'^E0$', 'E0 vs WL'),
            ('E1+', r'^E\d+$', 'EDGE MARGIN vs WL', r'^E0$|^ESUM.*$'),
            ('V0', r'^V0$', 'V0 vs WL'),
            ('V1+', r'^V\d+$', 'VALLEY MARGIN vs WL', r'^V0$'),
            ('ESUM', r'^ESUM$', 'ESUMS vs WL')
        ]
        
        for plot_type, pattern, title, *exclude in plot_configs:
            filtered_cols = self._filter_columns(columns_to_stack, pattern, exclude[0] if exclude else None)
            if filtered_cols:
                self._create_rwb_plot(df, filtered_cols, index_columns, file, title.replace('vs WL', f'by_wl_{file}'))


    def _filter_columns(self, columns: List[str], include_pattern: str, exclude_pattern: str = None) -> List[str]:
        """Filter columns based on include/exclude patterns"""
        filtered = [col for col in columns if re.match(include_pattern, col)]
        if exclude_pattern:
            filtered = [col for col in filtered if not re.match(exclude_pattern, col)]
        return filtered
    
    def _create_rwb_plot(self, df: pd.DataFrame, columns: List[str], index_cols: List[str], 
                        file: str, plot_name: str) -> None:
        """Create RWB plot with stacked data"""
        stacked_df = df.set_index(index_cols)[columns].stack().reset_index(name='rwb')
        stacked_df.rename(columns={f'level_{len(index_cols)}': 'edge'}, inplace=True)
        stacked_df['rwb'] = pd.to_numeric(stacked_df['rwb'], errors='coerce').astype(float)
        
        plotter = DataPlotter(stacked_df)
        plotter.scatter_with_agg_line(
            x_col='wl',
            y_col='rwb',
            overlay_by='blkgrp',
            loop_by='',
            agg_func='mean',
            caption_func='min',
            title=plot_name.replace('_by_wl_', ' vs WL '),
            split_by_row='mode',
            split_by_col='edge',
            ref_lines_y=[(0, '')],
            output_path=f"{self.output_path}/saved_plots/{plot_name}.jpg"
        )


class ThreadedProcessor:
    """Handles threaded processing of contexts"""
    
    def __init__(self, bq_manager: BigQueryManager, config: Dict, max_threads: int = 5):
        self.bq = bq_manager
        self.config = config
        self.semaphore = threading.Semaphore(max_threads)
        self.lock = threading.Lock()
        self.processor = DataProcessor(config)
        self.plotter = PlotManager(config["output_path"])
    
    def process_context(self, entry: Dict) -> None:
        """Process a single context entry"""
        with self.semaphore:
            file_name = entry["file_name"]
            context = entry["context"]
            
            logging.info(f"Thread started for: {file_name}")
            
            try:
                # Run query
                template = "cwer_rber_hrer.sql" if "cwer" in file_name else "rwb.sql"
                results = self.bq.run_template_query(template, context)
                
                # Save and load data
                output_path = os.path.join(self.config["output_path"], file_name)
                self.bq.save_to_csv(results, output_path)
                df = pd.read_csv(output_path, keep_default_na=False, low_memory=False)
                
                # Process data
                df = self.processor.decode_cycle_group(df)
                df = self.processor.concatenate_columns(df, ['step', 'read_point'])
                
                # Generate outputs based on file type
                if "cwer" in file_name:
                    if "wl" not in file_name:
                        self.processor.pass_fail_summary(df, self.config["output_path"], 
                                                       file_name, self.config["lut_functions"])
                        with self.lock:
                            self.plotter.plot_cwer(df, file_name, self.config["lut_functions"])
                    else:
                        with self.lock:
                            self.plotter.plot_cwer_wl(df, file_name)
                
                elif "rwb" in file_name:
                    self.processor.rwb_summary(df, self.config["output_path"], file_name)
                    with self.lock:
                        self.plotter.plot_rwb_data(df, file_name)
                
                logging.info(f"Completed: {file_name}")
                
            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")
    
    def process_all_contexts(self, contexts: List[Dict]) -> None:
        """Process all contexts using threading"""
        random.shuffle(contexts)
        threads = []
        
        for entry in contexts:
            thread = threading.Thread(target=self.process_context, args=(entry,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()


def concat_rwb_summaries(output_path: str) -> None:
    """Concatenate RWB summary files"""
    summary_files = glob.glob(os.path.join(f"{output_path}/rwb_summary", '*rwb_summary*.csv'))
    
    if not summary_files:
        return
    
    print("Combining the following files:")
    for file in summary_files:
        print(file)
    
    df = pd.concat([pd.read_csv(file, keep_default_na=False) for file in summary_files])
    
    group_by = ['runmode', 'mode']
    agg_info = ['blkgrp', 'read_point']
    
    result = df.groupby(group_by)['min_value'].min().reset_index()
    merged_df = pd.merge(df, result, on=group_by + ['min_value'], how='inner')
    
    agg_dict = {col: lambda x: x.iloc[0] if x.nunique() == 1 else ','.join(map(str, sorted(set(x)))) 
               for col in agg_info}
    
    summary_df = merged_df.groupby(group_by + ['min_column', 'min_value']).agg(agg_dict).reset_index()
    summary_df.to_csv(f"{output_path}/rwb_summary_final.csv", index=False)


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Get distinct items
    df_cwer = analyzer.create_pbendpage_column(
        analyzer.get_distinct_items('dyn_read')
    )
    df_rwb = analyzer.create_pbendpage_column(
        analyzer.get_distinct_items('rwb', ['mode'])
    )
    
    # Generate contexts
    cwer_contexts = analyzer.generate_cwer_contexts(df_cwer)
    rwb_contexts = analyzer.generate_rwb_contexts(df_rwb)
    all_contexts = cwer_contexts + rwb_contexts
    
    # Process contexts
    processor = ThreadedProcessor(analyzer.bq, analyzer.config)
    processor.process_all_contexts(all_contexts)
    
    # Final RWB summary
    rwb_summary_path = f"{analyzer.config['output_path']}/rwb_summary"
    if (not os.path.exists(f"{analyzer.config['output_path']}/rwb_summary_final.csv") and 
        os.path.exists(rwb_summary_path)):
        concat_rwb_summaries(analyzer.config["output_path"])


if __name__ == "__main__":
    main()