import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_recall_curve, roc_curve, auc, root_mean_squared_error
)
from scipy.stats import wasserstein_distance
import argparse

parser = argparse.ArgumentParser(description='Script to evaluate/score imputation results.')
parser.add_argument('--impute_path',type=str,default=None)
parser.add_argument('--data_path',type=str,default=None)
parser.add_argument('--output_path',type=str,default=None)
parser.add_argument('--method',type=str,default=None)
parser.add_argument('--num_replicates',type=int,default=3)

def load_binmap(fpath):
    with open(fpath, 'r') as file:
        lines = file.readlines()

    # Extract the 'isbinary' column
    isbinary_values = [line.strip().split(',')[1] for line in lines[1:]]  # Skip header

    # Convert the 'isbinary' values to a boolean vector
    isbinary_vector = np.array(isbinary_values) == 'True'

    return isbinary_vector

def compute_average_r2(imputed_df, ground_df, mask_df, binary_col_vector):
    r2_scores = []
    quality_scores = []
    r2_bin_scores = []
    aupr_scores = []
    auroc_scores = []
    mse_scores = []
    mae_scores = []
    wasserstein_score = []
    rmse_score = []
    for indicies,col in enumerate(ground_df.columns):
        # Mask to select only the imputed values for this column
        if sum(mask_df[col]) == 0:
            continue
        mask = mask_df[col].astype(bool)
        # Select the imputed and ground truth values based on the mask
        imputed_values = imputed_df[col].values[mask]
        ground_values = ground_df[col].values[mask]
        if(len(imputed_values)!=len(ground_values)):
            print("len not equal")
        if(np.isnan(imputed_values).any()):
            print("imputed has na")
        if(np.isnan(ground_values).any()):
            print("ground has na")       
        # Ensure there are enough values to compute R²
        if len(ground_values) > 1:
            r2 = np.corrcoef(
                imputed_values,
                ground_values
                )[0, 1]
            r2 **= 2
            #r2=r2_score(ground_values,imputed_values)
            r2_scores.append(r2)

            var_imp = imputed_values.var()
            var_obs = ground_df[col].values.var()
            var_info = var_imp / var_obs
            quality_scores.append(var_info)

            wasserstein=wasserstein_distance(ground_values,imputed_values) 
            wasserstein_score.append(wasserstein)

            rmse = root_mean_squared_error(ground_values,imputed_values)
            rmse_score.append(rmse)

            if binary_col_vector[indicies]:
                # For binary columns, calculate R², AUC-PR, and AUC-ROC
                r2_bin = np.corrcoef(
                imputed_values,
                ground_values
                )[0, 1]

                r2_bin **= 2
                r2_bin_scores.append(r2_bin)
                
                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(ground_values, imputed_values)
                auc_pr = auc(recall, precision)
                aupr_scores.append(auc_pr)
                
                # ROC AUC
                fpr, tpr, _ = roc_curve(ground_values, imputed_values)
                auc_roc = auc(fpr, tpr)
                auroc_scores.append(auc_roc)
                
            else:
                # For continuous columns, calculate MSE and MAE
                mse = mean_squared_error(ground_values, imputed_values)
                mae = mean_absolute_error(ground_values, imputed_values)
                mse_scores.append(mse)
                mae_scores.append(mae)

    # Average the R² scores across columns
    average_r2 = np.nanmean(r2_scores) 
    average_quality = np.nanmean(quality_scores) 
    average_mse = np.nanmean(mse_scores)
    average_mae = np.nanmean(mae_scores)
    average_r2_bin = np.nanmean(r2_bin_scores)
    average_aupr = np.nanmean(aupr_scores)
    average_auroc = np.nanmean(auroc_scores)
    sum_wasserstein = np.nansum(wasserstein_score)
    average_rmse = np.nanmean(rmse_score)
    return {
        "r2": average_r2,
        "mse": average_mse,
        "mae": average_mae,
        "aupr": average_aupr,
        "auroc": average_auroc,
        "r2_bin": average_r2_bin,
        "quality": average_quality,
        "wasserstein": sum_wasserstein,
        "rmse": average_rmse
    }

if __name__ == "__main__":
    args = parser.parse_args()

    num_replicates = args.num_replicates
    dataset = ['bike','california','default','income','letter','magic','obesity','shoppers','spam','students']
    miss_type = ['mar','mcar','mnar','mnarsm']
    index = [i for i in range(1, num_replicates+1)]
    missing_pct = [10, 30, 50, 70]

# Create an empty list to store the results
    results = []

    # Iterate through datasets and miss types
    for dataset_name in dataset:
        for miss_type_name in miss_type:
            if(dataset_name == 'shoppers' and miss_type_name == 'mnarsm'):
                continue
            for phase in ['train', 'val']:
                for mask_ratio in missing_pct:
                    try:
                        average_r2_scores = []
                        average_quality_scores = []
                        average_r2_bin_scores = []
                        average_aupr_scores = []
                        average_auroc_scores = []
                        average_mse_scores = []
                        average_mae_scores = []
                        sum_wasserstein_scores = []
                        average_rmse_scores = []
                        for i in index:
                            # Load the ground truth, mask, and imputed data
                            ground_df = pd.read_csv(f'{args.data_path}/{dataset_name}/{dataset_name}-{miss_type_name}-{mask_ratio}/p{phase}-true-{i}.tsv', sep='\t')
                            mask_df = pd.read_csv(f'{args.data_path}/{dataset_name}/{dataset_name}-{miss_type_name}-{mask_ratio}/p{phase}-mask-{i}.tsv', sep='\t')
                            isbinary_vector = load_binmap(f'{args.data_path}/{dataset_name}/{dataset_name}-{miss_type_name}-{mask_ratio}/cats.csv')

                            #imputed_ofname = f'{impute_path}/{method}_{phase}_{dataset_name}_{miss_type_name}_{mask_ratio}_{i}_predictions.tsv'
                            imputed_ofname = f'{args.impute_path}/{args.method}/{dataset_name}/{dataset_name}-{miss_type_name}-{mask_ratio}/p{phase}-predicted-{i}.tsv'

                            imputed_df = pd.read_csv(imputed_ofname, sep='\t')
                            
                            # Compute average R²
                            print(f"{dataset_name}-{miss_type_name}-{mask_ratio}/p{phase}-true-{i}")
                            metrics = compute_average_r2(imputed_df, ground_df, mask_df,isbinary_vector)
                            average_r2_scores.append(metrics['r2'])
                            average_quality_scores.append(metrics['quality'])
                            average_mse_scores.append(metrics['mse'])
                            average_mae_scores.append(metrics['mae'])
                            average_aupr_scores.append(metrics['aupr'])
                            average_auroc_scores.append(metrics['auroc'])
                            average_r2_bin_scores.append(metrics['r2_bin'])
                            sum_wasserstein_scores.append(metrics['wasserstein'])
                            average_rmse_scores.append(metrics['rmse'])
                            # Calculate mean and std for each metric
                        print(average_r2_scores)
                        mean_r2 = np.nanmean(average_r2_scores)
                        std_r2 = np.nanstd(average_r2_scores) 
                        se_r2 = std_r2 / np.sqrt(len(average_r2_scores))

                        mean_quality = np.nanmean(average_quality_scores)
                        std_quality = np.nanstd(average_quality_scores)
                        se_quality = std_quality / np.sqrt(len(average_quality_scores))

                        mean_mse = np.nanmean(average_mse_scores)
                        std_mse = np.nanstd(average_mse_scores)
                        se_mse = std_mse / np.sqrt(len(average_mse_scores))

                        mean_mae = np.nanmean(average_mae_scores)
                        std_mae = np.nanstd(average_mae_scores)
                        se_mae = std_mae / np.sqrt(len(average_mae_scores))

                        mean_aupr = np.nanmean(average_aupr_scores)
                        std_aupr = np.nanstd(average_aupr_scores)
                        se_aupr = std_aupr / np.sqrt(len(average_aupr_scores))

                        mean_auroc = np.nanmean(average_auroc_scores)
                        std_auroc = np.nanstd(average_auroc_scores)
                        se_auroc = std_auroc / np.sqrt(len(average_auroc_scores))

                        mean_r2_bin = np.nanmean(average_r2_bin_scores)
                        std_r2_bin = np.nanstd(average_r2_bin_scores)
                        se_r2_bin = std_r2_bin / np.sqrt(len(average_r2_bin_scores))

                        mean_wasserstein_distance = np.nanmean(sum_wasserstein_scores)
                        std_wasserstein_distance = np.nanstd(sum_wasserstein_scores)
                        se_wasserstein_distance = std_wasserstein_distance / np.sqrt(len(sum_wasserstein_scores))

                        mean_rmse = np.nanmean(average_rmse_scores)
                        std_rmse = np.nanstd(average_rmse_scores)                     
                        se_rmse = std_rmse / np.sqrt(len(average_rmse_scores))

                        # Append the results to the list
                        results.append({
                            'dataset': dataset_name,
                            'miss_type': miss_type_name,
                            'miss_prop': mask_ratio,
                            'model': args.method,
                            'split': phase,
                            'R2': mean_r2,
                            'R2_STD': std_r2,
                            'R2_SE': se_r2,
                            'imp_qual': mean_quality,
                            'imp_qual_STD': std_quality,
                            'imp_qual_SE': se_quality,
                            'MSE': mean_mse,
                            'MSE_STD': std_mse,
                            'MSE_SE': se_mse,
                            'MAE': mean_mae,
                            'MAE_STD': std_mae,
                            'MAE_SE': se_mae,
                            'AUPR': mean_aupr,
                            'AUPR_STD': std_aupr,
                            'AUPR_SE': se_aupr,
                            'AUROC': mean_auroc,
                            'AUROC_STD': std_auroc,
                            'AUROC_SE': se_auroc,
                            'R2_BIN': mean_r2_bin,
                            'R2_BIN_STD': std_r2_bin,
                            'R2_BIN_SE': se_r2_bin,
                            'WD': mean_wasserstein_distance,
                            'WD_STD': std_wasserstein_distance,
                            'WD_SE': se_wasserstein_distance,
                            'RMSE': mean_rmse,
                            'RMSE_STD': std_rmse,
                            'RMSE_SE': se_rmse
                        })
                    except:
                        print(f'{args.impute_path}/{args.method}/{dataset_name}/{dataset_name}-{miss_type_name}-{mask_ratio}/p{phase}-predicted-{i}.tsv not found')


    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)
    outfile = f'{args.output_path}/scoreevals_{args.method}.tsv'
    # Save the DataFrame to a CSV file
    results_df.to_csv(outfile, index=False,sep='\t')

    print(f"Results saved to {outfile}")
