import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Ensure output directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    csv_path = "dementia_dataset_2.csv"
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Print Shape
    print("\n" + "=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # 2. Print Info
    print("\n" + "=" * 60)
    print("2. DATASET INFO")
    print("=" * 60)
    df.info()
    
    # 3. Print Head
    print("\n" + "=" * 60)
    print("3. DATASET HEAD")
    print("=" * 60)
    print(df.head())
    
    # 4. Missing Values per Column
    print("\n" + "=" * 60)
    print("4. MISSING VALUES PER COLUMN")
    print("=" * 60)
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_pct.round(2)
    })
    print(missing_df)
    
    # 5. Class Distribution of CDR Column
    print("\n" + "=" * 60)
    print("5. CLASS DISTRIBUTION OF CDR COLUMN")
    print("=" * 60)
    cdr_counts = df['CDR'].value_counts(dropna=False).sort_index()
    cdr_pct = (df['CDR'].value_counts(dropna=False, normalize=True).sort_index() * 100).round(2)
    cdr_df = pd.DataFrame({
        "Count": cdr_counts,
        "Percentage (%)": cdr_pct
    })
    print(cdr_df)
    
    # Set seaborn theme style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Age Distribution
    print("\nGenerating Age Distribution plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'].dropna(), kde=True, bins=20, color='skyblue')
    plt.title("Age Distribution of Participants", fontsize=14, fontweight='bold')
    plt.xlabel("Age (years)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    age_plot_path = os.path.join(output_dir, "age_distribution.png")
    plt.savefig(age_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {age_plot_path}")
    
    # Plot 2: MMSE Distribution Split by CDR
    print("Generating MMSE Distribution Split by CDR plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_cdr_valid = df.copy()
    df_cdr_valid['CDR_Str'] = df_cdr_valid['CDR'].astype(str).replace({'nan': 'Missing'})
    
    # Boxplot
    sns.boxplot(
        data=df_cdr_valid, 
        x='CDR_Str', 
        y='MMSE', 
        palette='Set2', 
        ax=axes[0],
        hue='CDR_Str',
        legend=False
    )
    axes[0].set_title("MMSE Score Distribution by CDR (Boxplot)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Clinical Dementia Rating (CDR)", fontsize=11)
    axes[0].set_ylabel("MMSE Score", fontsize=11)
    
    # Histogram / KDE
    sns.histplot(
        data=df_cdr_valid.dropna(subset=['CDR', 'MMSE']), 
        x='MMSE', 
        hue='CDR_Str', 
        element='step', 
        palette='Set2', 
        ax=axes[1]
    )
    axes[1].set_title("MMSE Score Histogram by CDR Category", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("MMSE Score", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    
    plt.tight_layout()
    mmse_plot_path = os.path.join(output_dir, "mmse_by_cdr.png")
    plt.savefig(mmse_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {mmse_plot_path}")
    
    # Plot 3: Correlation Heatmap of Numeric Features
    print("Generating Correlation Heatmap plot...")
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1
    )
    plt.title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_plot_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {corr_plot_path}")
    
    print("\nEDA script completed successfully!")

if __name__ == "__main__":
    main()
