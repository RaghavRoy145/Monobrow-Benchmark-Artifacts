# ==============================================================================
# VISUALIZER SCRIPT V3 FOR APR QUALITATIVE ANALYSIS
# ==============================================================================
# This script loads and processes the 'results.csv' file, which contains
# both quantitative metrics and our manual qualitative analysis verdicts.
# It generates several visualizations and tables to answer RQ1 and RQ2,
# with a focus on fair comparison, collapsed categories, and deep analysis.
#
# Requirements:
# pip install pandas matplotlib matplotlib-venn
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import warnings

# Suppress potential UserWarning from matplotlib_venn for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    'figure.titlesize': 22,   # For fig.suptitle()
    'axes.titlesize': 20,      # For ax.set_title()
    'axes.labelsize': 16,      # For x and y labels
    'xtick.labelsize': 12,     # For x-axis tick labels
    'ytick.labelsize': 12,     # For y-axis tick labels
    'legend.fontsize': 12,     # For legend text
    'legend.title_fontsize': 14 # For legend title
})
def load_and_clean_data(filepath="results.csv"):
    """
    Loads the raw CSV, selects representative patches, and cleans/standardizes
    the qualitative columns into our new three-tier correctness system.
    """
    df = pd.read_csv(filepath)
    
    # --- Step 1: Add a 'dataset' column for fair comparison filtering ---
    df['dataset'] = df['bug_id'].apply(lambda x: 'Generated' if 'generated' in x else 'OpenSSL')

    # --- Step 2: Select a single representative patch for multi-patch tools ---
    df['patch_type_priority'] = df['patch_type'].apply(lambda x: 0 if x == 'Guarded Block' else 1)
    df_sorted = df.sort_values(by=['bug_id', 'tool_name', 'patch_type_priority'])
    df_reps = df_sorted.drop_duplicates(subset=['bug_id', 'tool_name'], keep='first').copy()

    # --- Step 3: Clean and Standardize into the THREE-TIER Correctness System ---
    # This new, comprehensive mapping handles all messy values from the CSV and
    # collapses them into our three clear categories for visualization.
    correctness_mapping = {
        # Mappings to 'Correct'
        'Correct and Optimal': 'Correct',
        'Correct but Sub-optimal': 'Correct',
        
        # Mappings to 'Correct (Risky)'
        'Correct but High-Risk and Sub-optimal': 'Correct (Risky)',
        
        # Mappings to 'Incorrect'
        'Correct but Semantically Flawed': 'Incorrect', # CRITICAL FIX: Semantically flawed is incorrect.
        'Incorrect but Semantically Flawed': 'Incorrect',
        'Incorrect and Incomplete': 'Incorrect',
        'Correct but Incomplete': 'Correct (Risky)', # CRITICAL FIX: Incomplete is incorrect.
        'Incorrect and Compilation Error': 'Incorrect',
        'Incorrect (Incomplete)': 'Incorrect',
        'Incorrect (Ineffective)': 'Incorrect',
        'Incorrect (Semantically Flawed)': 'Incorrect',
        'Incorrect (Compilation Error)': 'Incorrect'
    }
    df_reps['Correctness_Category_Simple'] = df_reps['Correctness_Category'].apply(
        lambda x: correctness_mapping.get(x, 'Incorrect') # Default to Incorrect if unmapped
    )

    # Clean other columns as before
    fp_mapping = {'Yes': True, 'No': False, 'Np': False}
    df_reps['Is_NPE_FP'] = df_reps['Is_NPE_FP'].map(fp_mapping).astype(bool)
    df_reps['New_Error_Type'] = df_reps['New_Error_Type'].str.title()
    df_reps.dropna(subset=['Correctness_Category'], inplace=True)

    return df_reps

def plot_correctness_summary(df, dataset_filter=None):
    """
    Generates a stacked bar chart using the new three-tier correctness system.
    """
    title_suffix = "Overall"
    filename_suffix = "overall"
    if dataset_filter:
        df = df[df['dataset'] == dataset_filter].copy()
        title_suffix = f"{dataset_filter}" 
        filename_suffix = f"{dataset_filter.lower()}_only"

    # Use the new, simplified categories and a simpler color map
    category_order = ['Correct', 'Correct (Risky)', 'Incorrect']
    color_map = {'Correct': 'green', 'Correct (Risky)': 'gold', 'Incorrect': 'red'}
    
    summary = df.groupby(['tool_name', 'Correctness_Category_Simple']).size().unstack(fill_value=0)
    summary = summary.reindex(columns=category_order, fill_value=0)
    
    summary_percent = summary.div(summary.sum(axis=1), axis=0) * 100

    ax = summary.plot(
        kind='bar', stacked=True, figsize=(10, 7),
        color=[color_map.get(cat, 'gray') for cat in summary.columns],
        edgecolor='black', width=0.4
    )

    for c in ax.containers:
        labels = []
        for i, v in enumerate(c):
            count = v.get_height()
            if count > 0:
                tool = summary.index[i]
                category = c.get_label()
                percent = summary_percent.loc[tool, category]
                labels.append(f'{int(count)}\n({percent:.1f}%)')
            else:
                labels.append('')
        ax.bar_label(c, labels=labels, label_type='center', color='black', weight='bold')

    plt.title(f'Generated Patches ({title_suffix})')
    plt.xlabel('Tool')
    plt.ylabel('Number of Patches')
    plt.xticks(rotation=0)
    plt.legend(title='Correctness Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    filename = f'correctness_summary_{filename_suffix}.png'
    plt.savefig(filename, dpi=1200)
    print(f"\nGenerated '{filename}'")
    plt.close()

def plot_structural_impact(df):
    """
    NEW: Generates a bar chart comparing the average structural impact of patches
    on the OpenSSL dataset.
    """
    df_openssl = df[df['dataset'] == 'OpenSSL'].copy()
    # Ensure structural_impact is numeric, coercing errors
    df_openssl['structural_impact'] = pd.to_numeric(df_openssl['structural_impact'], errors='coerce')
    avg_impact = df_openssl.groupby('tool_name')['structural_impact'].mean().sort_values()

    plt.figure(figsize=(16, 12))
    ax = avg_impact.plot(kind='bar', color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.title('Average Structural Impact of Patches on OpenSSL (RQ1)')
    plt.xlabel('Tool')
    plt.ylabel('Average Structural Impact (CFG Nodes)')
    plt.xticks(rotation=0)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.tight_layout()
    plt.savefig('structural_impact_comparison.png', dpi=1200)
    print("\nGenerated 'structural_impact_comparison.png'")
    plt.close()

def generate_correct_patch_minimality_table(df):
    """
    NEW: Generates a table analyzing the minimality metrics for ONLY the
    patches that were deemed correct or sub-optimal.
    """
    correct_patches = df[df['Correctness_Category_Simple'].isin(['Correct', 'Correct (Risky)'])].copy()
    
    # Ensure relevant columns are numeric
    correct_patches['L_local_norm'] = pd.to_numeric(correct_patches['L_local_norm'], errors='coerce').fillna(0)
    correct_patches['cost_g_overhead'] = pd.to_numeric(correct_patches['cost_g_overhead'], errors='coerce').fillna(0)
    correct_patches['structural_impact'] = pd.to_numeric(correct_patches['structural_impact'], errors='coerce').fillna(0)
    
    minimality_summary = correct_patches.groupby('tool_name').agg(
        num_correct_patches=('bug_id', 'count'),
        avg_imprecision_L=('L_local_norm', 'mean'),
        avg_overhead_G=('cost_g_overhead', 'mean'),
        avg_structural_impact=('structural_impact', 'mean')
    ).reset_index()
    
    print("\n--- NEW Table: Minimality of Correct Patches Only (RQ1) ---")
    print(minimality_summary.to_markdown(index=False, floatfmt=".2f"))

# (Other table/viz functions are updated to use the new dataset filter logic)
def generate_failure_mode_table(df, dataset_filter=None):
    title_suffix = "Overall"
    if dataset_filter:
        df = df[df['dataset'] == dataset_filter].copy()
        title_suffix = f"{dataset_filter} Only"
        
    incorrect_patches = df[df['Correctness_Category_Simple'] == 'Incorrect']
    failure_table = pd.crosstab(incorrect_patches['New_Error_Type'], incorrect_patches['tool_name'])
    print(f"\n--- Table 2: Taxonomy of Patch Failures ({title_suffix}) ---")
    print(failure_table.to_markdown())

def plot_bug_overlap_venn(df):
    df_openssl = df[df['dataset'] == 'OpenSSL'].copy()
    bugs_efffix = set(df_openssl[df_openssl['tool_name'] == 'efffix']['function_name'].unique())
    bugs_footpatch = set(df_openssl[df_openssl['tool_name'] == 'footpatch']['function_name'].unique())
    bugs_monobrow = set(df_openssl[df_openssl['tool_name'] == 'monobrow']['function_name'].unique())
    bugs_monobrow.add('CRYPTO_strdup')
    plt.figure(figsize=(10, 10))
    venn3([bugs_efffix, bugs_footpatch, bugs_monobrow], set_labels=('EffFix', 'FootPatch', 'Monobrow'))
    plt.title('Overlap of Bugs Addressed on OpenSSL')
    plt.savefig('bug_overlap_venn.png', dpi=1200)
    print("\nGenerated 'bug_overlap_venn.png'")
    plt.close()

def generate_original_bug_type_table(df):
    bug_type_table = pd.crosstab(df['Original_Bug_Type'], df['tool_name'])
    print("\n--- Table 4a: Classification of Original Bugs Addressed by Each Tool ---")
    print(bug_type_table.to_markdown())
    
    fp_patches = df[df['Is_NPE_FP'] == True]
    print("\n--- Table 4b: Patches Generated for False Positives ---")
    if fp_patches.empty:
        print("No patches were generated for false positives.")
    else:
        fp_summary = fp_patches.groupby('tool_name').size().reset_index(name='count')
        print(fp_summary.to_markdown(index=False))

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        cleaned_df = load_and_clean_data(filepath="results.csv")
        print("--- Data Loading and Cleaning Summary ---")
        print(f"Total unique patches in cleaned dataset: {len(cleaned_df)}")
        
        plot_correctness_summary(cleaned_df, dataset_filter=None)
        plot_correctness_summary(cleaned_df, dataset_filter='OpenSSL')
        
        generate_failure_mode_table(cleaned_df, dataset_filter=None)
        generate_failure_mode_table(cleaned_df, dataset_filter='OpenSSL')

        plot_structural_impact(cleaned_df)
        generate_correct_patch_minimality_table(cleaned_df)
        plot_bug_overlap_venn(cleaned_df)
        generate_original_bug_type_table(cleaned_df)
        
        print("\nAll visualizations and tables generated successfully.")
        
    except FileNotFoundError:
        print("\nERROR: 'results.csv' not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
