import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
plt.rcParams.update({
    'figure.titlesize': 22,   # For fig.suptitle()
    'axes.titlesize': 20,      # For ax.set_title()
    'axes.labelsize': 16,      # For x and y labels
    'xtick.labelsize': 12,     # For x-axis tick labels
    'ytick.labelsize': 12,     # For y-axis tick labels
    'legend.fontsize': 12,     # For legend text
    'legend.title_fontsize': 14 # For legend title
})
def generate_unified_pareto_plot(df):
    """
    Generates a single figure with a unified impact metric on the X-axis.
    """
    # --- Data Preparation (No changes needed here) ---
    df['cost_l_local_imprecision'] = pd.to_numeric(df['cost_l_local_imprecision'], errors='coerce')
    df['structural_impact'] = pd.to_numeric(df['structural_impact'], errors='coerce')
    df['total_function_nodes'] = pd.to_numeric(df['total_function_nodes'], errors='coerce')

    df_skip = df[df['patch_type'].isin(['Guarded Block', 'Early Exit'])].copy()
    df_skip['L_local_norm'] = df_skip.apply(
        lambda row: row['cost_l_local_imprecision'] / row['total_function_nodes'] if row['total_function_nodes'] > 0 else 0,
        axis=1
    )
    df_skip['L_global_norm'] = df_skip.apply(
        lambda row: row['structural_impact'] / row['total_function_nodes'] if row['total_function_nodes'] > 0 else 0,
        axis=1
    )
    df_skip['normalized_total_impact'] = df_skip['L_local_norm'] + df_skip['L_global_norm']

    # --- Plotting Changes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # CHANGE 1: Create a single subplot instead of two. Adjust figsize for a single plot.
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 7.2))

    # CHANGE 2: Remove the data splitting logic. We will plot the entire df_skip DataFrame.
    # The helper function `_plot_single_unified_pareto` will handle distinguishing the tools.
    
    # CHANGE 3: Make a single call to the plotting helper function.
    # The title is now more direct and will be set by the helper.
    _plot_single_unified_pareto(ax, df_skip, 'Control-Flow Patch Quality: Overhead vs. Unified Impact')

    # CHANGE 4: Simplify the layout and saving logic.
    plt.tight_layout()
    plt.savefig('unified_pareto_comparison.png')
    print("Saved unified Pareto comparison to unified_pareto_comparison.png")
    plt.close()

def _plot_single_unified_pareto(ax, df, title):
    """Helper to plot a single Pareto chart using the unified impact metric."""
    if df.empty:
        ax.text(0.5, 0.5, 'No data available.', ha='center'); ax.set_title(title); return

    sns.scatterplot(
        data=df, x='normalized_total_impact', y='cost_g_overhead',
        hue='tool_name', style='patch_type', s=150, ax=ax, alpha=0.8, edgecolor='black'
    )
    ax.set_title(title)
    ax.set_xlabel('Normalized Total Impact')
    ax.set_ylabel('Absolute Overhead (G)')
    ax.legend(title='Tool / Patch Type')

def print_replace_patches_table(df):
    replace_df = df[df['patch_type'] == 'Replace'].copy()
    if replace_df.empty:
        print("\n--- No Data-Flow (Replace) Patches Found ---")
        return
    
    table_df = replace_df[['bug_id', 'tool_name', 'cost_rep_modification']]
    print("\n--- Data-Flow (Replace) Patch Analysis ---")
    print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

def print_monobrow_alias_table(df):
    mono_df = df[(df['tool_name'] == 'monobrow') & (df['patch_type'] != 'Replace')].copy()
    mono_df['total_aliases'] = pd.to_numeric(mono_df['total_aliases'], errors='coerce')
    alias_df = mono_df[mono_df['total_aliases'] > 1]
    
    if alias_df.empty:
        print("\n--- No Monobrow patches with multiple aliases found ---")
        return
        
    table_df = alias_df[['bug_id', 'total_aliases', 'cost_g_overhead']]
    print("\n--- Monobrow Alias Handling Analysis (for Skip patches) ---")
    print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

def print_direct_comparison_table(df, function_name_str):
    comp_df = df[df['function_name'].str.strip() == function_name_str].copy()
    if comp_df.shape[0] < 2:
        print(f"\nCannot generate direct comparison for '{function_name_str}': Fewer than 2 tools patched this function.")
        return
    table_df = comp_df[['tool_name', 'patch_type', 'cost_g_overhead', 'cost_l_local_imprecision', 'structural_impact', 'cost_rep_modification']]
    print(f"\n--- Direct Comparison Table: `{function_name_str}` ---")
    print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

def main():
    try:
        results_df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("Error: results.csv not found. Please run batch_analyzer.py first.")
        return
    
    # Generate the main side-by-side Pareto plot with the new unified metric
    generate_unified_pareto_plot(results_df)
    
    # Generate the text table for Replace patches
    print_replace_patches_table(results_df)

    # Generate the text table for Monobrow's alias handling
    print_monobrow_alias_table(results_df)
    
    # Print the direct comparison tables for overlapping bugs
    print_direct_comparison_table(results_df, 'dtls1_buffer_message')
    print_direct_comparison_table(results_df, 'CRYPTO_strdup')

if __name__ == '__main__':
    main()
