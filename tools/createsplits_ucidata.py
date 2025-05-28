import argparse
from os import path as ospath
import os
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process UCI datasets with various missingness mechanisms.")
    
    parser.add_argument('--data', type=str, default='all',
                        help="Specify the dataset to use. Use 'all' for all datasets. (default: all)")
    
    parser.add_argument('--mechanism', type=str, default='all',
                        choices=['all', 'MCAR', 'MAR', 'MNAR'],
                        help="Specify the missingness mechanism to simulate. Use 'all' for all mechanisms. (default: all)")
    
    parser.add_argument('--pmiss', type=float, default=0.3,
                        help="Proportion of missing data to simulate. (default: 0.3)")
    
    parser.add_argument('--pval', type=float, default=0.2,
                        help="Proportion of data to use for validation. (default: 0.2)")

    parser.add_argument('--nrep', type=int, default=1,
                        help="Number of replicates to make. (default: 1)")
    
    parser.add_argument('--odir', type=str, required=True,
                        help="Output directory for saving files. (default: output)")
    
    parser.add_argument('--dpath', type=str, default=None,
                        help="Path to the data directory for datasets requiring manual download. (default: None)")
                

    return parser.parse_args()

class DataBunch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def process_califdata():
    raw_data = fetch_california_housing(as_frame=True)
    
    # Extract features and target
    X = raw_data.data
    y = raw_data.target
    
    # Combine features and target into a single DataFrame
    combined_data = pd.concat([X, y], axis=1)
    
    # Create a DataFrame for variable descriptions
    descriptions = pd.DataFrame({
        'name': raw_data.feature_names + [raw_data.target_names],
        'description': ['median income in block group', 
                            'median house age in block group',
                            'average number of rooms per household',
                            'average number of bedrooms per household',
                            'block group population',
                            'average number of household members',
                            'block group latitude',
                            'block group longitude'] + [raw_data.target_names],
        'role': ['Feature'] * len(raw_data.feature_names) + ['Target']
    })

    # Create a Bunch object to mimic UCI dataset structure
    return DataBunch(
        data=DataBunch(
            features=X,
            targets=y
        ),
        variables=descriptions,
        description=raw_data.DESCR
    )

def process_bikedata():
    raw_data = fetch_ucirepo(id=560)
    # Create mappings for categorical variables
    holiday_map = {'No Holiday': 0, 'Holiday': 1}

    season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}

    # Apply mappings
    raw_data.data.features['Holiday'] = raw_data.data.features['Holiday'].map(holiday_map)
    raw_data.data.features['Seasons'] = raw_data.data.features['Seasons'].map(season_map)


    raw_data.data.features = raw_data.data.features.drop(columns=['Date'])

    raw_data.variables.description = raw_data.variables.description[1:]

    raw_data.variables.role = raw_data.variables.role[1:]

    return raw_data

def process_shoppersdata():
    raw_data = fetch_ucirepo(id=468)

    raw_data.data.features = raw_data.data.features.dropna()

    # # Convert 'Revenue' to numeric (0 for FALSE, 1 for TRUE)
    # raw_data.data.features['Revenue'] = (raw_data.data.features['Revenue'] == 'TRUE').astype(int)
    # # Store original data before conversion
    # original_data = raw_data.data.features.copy()

    # Convert 'Weekend' to numeric (0 for FALSE, 1 for TRUE)
    raw_data.data.features['Weekend'] = (raw_data.data.features['Weekend'] == 'TRUE').astype(int)

    # Convert 'VisitorType' to numeric
    visitor_type_map = {'Other' : -1, 'Returning_Visitor': 0, 'New_Visitor': 1}
    raw_data.data.features['VisitorType'] = raw_data.data.features['VisitorType'].map(visitor_type_map)

    # Convert 'Month' to numeric (1-12 for Jan-Dec)
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    raw_data.data.features['Month'] = raw_data.data.features['Month'].map(month_map)
    
    # # Convert columns to numeric where possible
    # for col in raw_data.data.features.columns:
    #     raw_data.data.features[col] = pd.to_numeric(raw_data.data.features[col], errors='ignore')
    
    # # Check for NaN values in all columns
    # nan_columns = raw_data.data.features.columns[raw_data.data.features.isna().any()].tolist()
    
    # print("Columns with NaN values:")
    # for col in nan_columns:
    #     nan_count = raw_data.data.features[col].isna().sum()
    #     unique_values = set(original_data[col].dropna().unique())
    #     print(f"Column '{col}': {nan_count} NaN values")
    #     print(f"Unique values in original column where NaN values are present: {unique_values}")
    #     print()
    
    # # Original debug information
    # numeric_columns = raw_data.data.features.select_dtypes(include=[np.number]).columns
    # nan_count = np.isnan(raw_data.data.features[numeric_columns].values).sum()
    # print(f"Number of NaN values in numeric columns: {nan_count}")
    
    # print(raw_data.data.features.columns)
    # print(f"Total number of NaN values: {np.isnan(raw_data.data.features.values).sum()}")

    return raw_data

def preprocess_incomedata():
    raw_data = fetch_ucirepo(id=2)

    # Create mappings for categorical variables
    workclass_map = {
        'Never-worked': 0, 'Without-pay': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3,
        'Local-gov': 4, 'State-gov': 5, 'Federal-gov': 6, 'Private': 7, '?' : -1,
    }
    
    education_map = {
        'Preschool': 0, '1st-4th': 1,  '5th-6th': 2, '7th-8th': 3, '9th': 4, 
        '10th': 5,  '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10,  
        'Assoc-acdm': 11, 'Prof-school': 12, 'Bachelors': 13,  'Masters': 14, 'Doctorate': 15
    }
    
    marital_status_map = {
        'Never-married': 0, 'Married-civ-spouse': 1, 'Married-AF-spouse': 2,
        'Married-spouse-absent': 3, 'Separated': 4, 'Divorced': 5, 'Widowed': 6
    }
    
    occupation_map = {
        'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4,
        'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8,
        'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
        'Armed-Forces': 13, '?' : -1
    }
    
    relationship_map = {
        'Unmarried': 0, 'Own-child': 1, 'Wife': 2, 'Husband': 3, 'Other-relative': 4, 'Not-in-family': 5
    }
    
    race_map = {
    'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 5, 'Other': 4, 
    }

    sex_map = {'Female': 0, 'Male': 1}

    native_country_map = {
    'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5,
    'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10,'China': 11,
    'Cuba': 12, 'Iran': 13, 'Honduras': 14,'Philippines': 15,'Italy': 16, 'Poland': 17, 'Jamaica': 18,
    'Vietnam': 19, 'Mexico': 20,'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24, 'Laos': 25,
    'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31,  'Nicaragua': 32, 'Scotland': 33,
    'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago': 37, 'Peru': 38,
    'Hong': 39, 'Holand-Netherlands': 40,
    '?': -1  # Added for unknown values
    }

    income_map = {'<=50K': 0, '>50K': 1}

    # Apply mappings
    raw_data.data.features = raw_data.data.features.dropna()
    raw_data.data.features['workclass'] = raw_data.data.features['workclass'].map(workclass_map)
    raw_data.data.features['education'] = raw_data.data.features['education'].map(education_map)
    raw_data.data.features['marital-status'] = raw_data.data.features['marital-status'].map(marital_status_map)
    raw_data.data.features['occupation'] = raw_data.data.features['occupation'].map(occupation_map)
    raw_data.data.features['relationship'] = raw_data.data.features['relationship'].map(relationship_map)
    raw_data.data.features['race'] = raw_data.data.features['race'].map(race_map)
    raw_data.data.features['sex'] = raw_data.data.features['sex'].map(sex_map)
    raw_data.data.features['native-country'] = raw_data.data.features['native-country'].map(native_country_map)
    # raw_data.data.features['income'] = raw_data.data.features['income'].map(income_map)

    # # Convert columns to numeric where possible
    # for col in raw_data.data.features.columns:
    #     raw_data.data.features[col] = pd.to_numeric(raw_data.data.features[col], errors='ignore')
    
    # # Check for NaN values in all columns
    # nan_columns = raw_data.data.features.columns[raw_data.data.features.isna().any()].tolist()
    
    # print("Columns with NaN values:")
    # for col in nan_columns:
    #     nan_count = raw_data.data.features[col].isna().sum()
    #     unique_values = set(original_data[col].dropna().unique())
    #     print(f"Column '{col}': {nan_count} NaN values")
    #     print(f"Unique values in original column where NaN values are present: {unique_values}")
    #     print()
    
    # # Original debug information
    # numeric_columns = raw_data.data.features.select_dtypes(include=[np.number]).columns
    # nan_count = np.isnan(raw_data.data.features[numeric_columns].values).sum()
    # print(f"Number of NaN values in numeric columns: {nan_count}")

    # print(raw_data.data.features.columns)
    # print(f"Total number of NaN values: {np.isnan(raw_data.data.features.values).sum()}")

    return raw_data

def process_obesitydata():
    raw_data = fetch_ucirepo(id=544)
    # Create mappings for categorical variables
    gender_map = {'Female': 0, 'Male': 1}
    
    family_history_map = {'no': 0, 'yes': 1}
    
    favc_map = {'no': 0, 'yes': 1}
    
    caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    
    smoke_map = {'no': 0, 'yes': 1}
    
    scc_map = {'no': 0, 'yes': 1}
    
    calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    
    mtrans_map = {
        'Automobile': 0,
        'Motorbike': 1,
        'Bike': 2,
        'Public_Transportation': 3,
        'Walking': 4
    }
    
    nobeyesdad_map = {
        'Insufficient_Weight': 0,
        'Normal_Weight': 1,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }

    # Apply mappings
    raw_data.data.features['Gender'] = raw_data.data.features['Gender'].map(gender_map)
    raw_data.data.features['family_history_with_overweight'] = raw_data.data.features['family_history_with_overweight'].map(family_history_map)
    raw_data.data.features['FAVC'] = raw_data.data.features['FAVC'].map(favc_map)
    raw_data.data.features['CAEC'] = raw_data.data.features['CAEC'].map(caec_map)
    raw_data.data.features['SMOKE'] = raw_data.data.features['SMOKE'].map(smoke_map)
    raw_data.data.features['SCC'] = raw_data.data.features['SCC'].map(scc_map)
    raw_data.data.features['CALC'] = raw_data.data.features['CALC'].map(calc_map)
    raw_data.data.features['MTRANS'] = raw_data.data.features['MTRANS'].map(mtrans_map)

    # Convert to numeric type
    numeric_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
                       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
                       'CALC', 'MTRANS', 'NObeyesdad']
    
    # df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return raw_data

def process_midata():
    raw_data = fetch_ucirepo(id=579)

    # print(raw_data.data.features.columns)
    # print(raw_data.variables.role)

    # Replace None with NaN using fillna()
    raw_data.data.features = raw_data.data.features.fillna(value=np.nan)

    return raw_data

def get_ucidata(uciname, datapath=None):
    dataset_functions = {
        'california' : process_califdata,
        'magic': lambda: fetch_ucirepo(id=159),
        'letter': lambda: fetch_ucirepo(id=59),
        # 'gesture': lambda: pd.read_csv(os.path.join(datapath, 'gesture_phase_segmentation.csv')),
        # 'secom': lambda: pd.read_csv(os.path.join(datapath, 'secom.csv')),
        # 'mi': lambda: fetch_ucirepo(id=579),
        'spam': lambda: fetch_ucirepo(id=94),
        'obesity': process_obesitydata,
        'bike': process_bikedata,
        'default': lambda: fetch_ucirepo(id=350),
        'shoppers':process_shoppersdata,
        'income': preprocess_incomedata,
        'students': lambda: fetch_ucirepo(id=697),
        'support2': lambda: fetch_ucirepo(id=880),
        'mi' : process_midata
    }

    if uciname not in dataset_functions:
        raise ValueError(f"Unknown dataset: {uciname}")

    if uciname in ['gesture', 'secom'] and not datapath:
        raise ValueError(f"Dataset {uciname} requires manual download. Please provide the datapath.")

    if uciname in ['gesture', 'secom']:
        raise NotImplementedError
    else:
        datadict = dataset_functions[uciname]()

    X = np.asarray(datadict.data.features.values)
    colnames = list(datadict.data.features.columns)
    colinfo = list(datadict.variables.description[datadict.variables.role == 'Feature'])
    assert len(colnames) == len(colinfo)

    return {
        'X' : X,
        'colnames' : colnames,
        'colinfo' : colinfo
    }

def ampute(x, mechanism, p_miss): 
    # Xdf =  pd.DataFrame(x, columns=cols)

    if mechanism == 'MNARsm':
        x_simulated = simulate_nan(x, p_miss, "MNAR", 'selfmasked')
    else:
        x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]

    x_miss = x_simulated["X_incomp"]

    return x_miss, mask

def is_binary(ground_truth):
    # Check for NAs
    na_mask = np.isnan(ground_truth)
    
    # Remove NA values for the check
    non_na = ground_truth[~na_mask]
    
    unique_values = np.unique(non_na)
    
    # Check if there are exactly two unique non-NA values
    if len(unique_values) != 2:
        return False
    
    # Check if the values are close to 0 and 1
    return np.allclose(unique_values, [0, 1], atol=1e-6)

def scale_data_wcats(x, cols):
    # Create a DataFrame from the input array and column names
    df = pd.DataFrame(x, columns=cols)
    
    # Initialize a StandardScaler
    scaler = StandardScaler()
    
    # Create a list to store binary status of columns
    is_binary_list = []
    
    for col in cols:
        try : 
            if is_binary(df[col].values):
                is_binary_list.append(True)
            else:
                is_binary_list.append(False)
                # Standardize non-binary columns
                df[col] = scaler.fit_transform(df[[col]])
        except:
            print(col)
            raise
    
    # Create the output DataFrame with binary status
    binary_status_df = pd.DataFrame({
        'cats': cols,
        'isbinary': is_binary_list
    })
    
    return df.to_numpy(), binary_status_df

def trainval_split(x, x_miss, x_mask, p_val):
    ids = np.arange(x.shape[0])

    x_train, x_val, ids_train, ids_val = train_test_split(x, ids, test_size=p_val)

    x_miss_train = x_miss[ids_train]
    x_miss_val = x_miss[ids_val]

    x_mask_train = x_mask[ids_train]
    x_mask_val = x_mask[ids_val]

    return {
        'x_train' : x_train,
        'x_val' : x_val,
        'miss_train' : x_miss_train,
        'miss_val' : x_miss_val,
        'mask_train' : x_mask_train,
        'mask_val' : x_mask_val
    }

def save_files(output_dir, data_dict, colnames, colinfo, cats_df, replicate=None):
    x_train = pd.DataFrame(data_dict['x_train'], columns=colnames)
    x_val = pd.DataFrame(data_dict['x_val'], columns=colnames)

    miss_train = pd.DataFrame(data_dict['miss_train'], columns=colnames)
    miss_val = pd.DataFrame(data_dict['miss_val'], columns=colnames)

    mask_train = pd.DataFrame(data_dict['mask_train'], columns=colnames)
    mask_val = pd.DataFrame(data_dict['mask_val'], columns=colnames)

    colinfo_df = pd.DataFrame({
        'cats' : colnames,
        'info' : colinfo
    })

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames as TSV files
    x_train.to_csv(os.path.join(output_dir, f"ptrain-true{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)
    x_val.to_csv(os.path.join(output_dir, f"pval-true{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)

    miss_train.to_csv(os.path.join(output_dir, f"ptrain{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)
    miss_val.to_csv(os.path.join(output_dir, f"pval{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)

    mask_train.to_csv(os.path.join(output_dir, f"ptrain-mask{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)
    mask_val.to_csv(os.path.join(output_dir, f"pval-mask{'-' + str(replicate) if replicate else ''}.tsv"), sep='\t', index=False)

    colinfo_df.to_csv(os.path.join(output_dir, 'colinfo.tsv'), sep='\t', index=False)
    cats_df.to_csv(os.path.join(output_dir, 'cats.csv'), sep=',', index=False)

    print(f"All files have been saved in the '{output_dir}' directory.")



if __name__ == "__main__":
    args = parse_arguments()

    valid_list = ['california', 'magic', 'letter', 'spam', 'obesity', 'bike', 'default', 'income', 'students', 'shoppers']
    valid_mech = ['MCAR', 'MAR', 'MNAR'] #, 'MNARsm']

    if not args.data in ['all', 'mi'] and args.data not in valid_list:
        raise ValueError(f"{args.data} is not a valid dataset!")

    if not args.mechanism == 'all' and args.mechanism not in valid_mech:
        raise ValueError(f"{args.mechanism} is not a valid missingness mechanism to simulate!")
    
    if not args.data == 'all':
        valid_list = [args.data]

    if not args.mechanism == 'all':
        valid_mech = [args.mechanism]


    if args.data == 'mi':
        # get data
        data_dict = get_ucidata(args.data)

        # scale data and cats df
        X_scaled, cats_df = scale_data_wcats(data_dict['X'], data_dict['colnames'])
        X_miss = X_scaled
        X_mask = np.isnan(X_scaled).astype(float)

        # Make train / val splits
        final_dict = trainval_split(X_scaled, X_miss, X_mask, args.pval)

        # Save final files
        final_odir = f"{args.odir}/uciml/{args.data}/"

        save_files(
                final_odir,
                final_dict,
                data_dict['colnames'],
                data_dict['colinfo'],
                cats_df
            )
        exit(0)

    for data in valid_list:
        print(f"Creating simulations for {data} dataset...")
        # get data
        data_dict = get_ucidata(data, datapath=args.dpath)

        # scale data and cats df
        X_scaled, cats_df = scale_data_wcats(data_dict['X'], data_dict['colnames'])

        for mech in valid_mech:
            # Ampute data and get masks
            if args.nrep <= 1 :
                X_miss, X_mask = ampute(X_scaled, mech, args.pmiss)

                # Make train / val splits
                final_dict = trainval_split(X_scaled, X_miss, X_mask, args.pval)

                # Save final files
                final_odir = f"{args.odir}/uciml/{data}/{data}-{mech.lower()}-{int(args.pmiss*100):02d}"

                save_files(
                    final_odir,
                    final_dict,
                    data_dict['colnames'],
                    data_dict['colinfo'],
                    cats_df
                )
            else:
                for i in range(args.nrep):
                    print(f"Making replicate {i+1} ...")
                    X_miss, X_mask = ampute(X_scaled, mech, args.pmiss)

                    # Make train / val splits
                    final_dict = trainval_split(X_scaled, X_miss, X_mask, args.pval)

                    # Save final files
                    final_odir = f"{args.odir}/uciml/{data}/{data}-{mech.lower()}-{int(args.pmiss*100):02d}"

                    save_files(
                        final_odir,
                        final_dict,
                        data_dict['colnames'],
                        data_dict['colinfo'],
                        cats_df,
                        replicate = i+1
                    )

    print("---DONE---")


