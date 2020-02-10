import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os

from sklearn import metrics
from scipy.stats import pearsonr

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Classes and functions for data extraction
class CellLine(object):
    """Basic, parent class representing cell line regradless of data source"""
    def __init__(self, cell_line_name, cell_line_id, data_source):
        self.name = cell_line_name
        self.id = cell_line_id
        self.data_source = data_source

class Drug(object):
    def __init__(self, drug_name, drug_id, data_source):
        self.name = drug_name
        self.id = drug_id
        self.data_source = data_source
        
class CellLineGDSCv2(CellLine):
    """Class representing single cell line in GDSCv2"""
    def __init__(self, cell_line_name, cell_line_id):
        super().__init__(cell_line_name, cell_line_id, "GDSCv2")
    def extract_full_gene_expression_data(self, gene_expression_df):
        """Extract gene expressions of all available genes"""
        # Just extract column corresponding to the cell line COSMIC ID
        if str(self.id) in gene_expression_df.columns:
            return gene_expression_df[str(self.id)]
        else:
            raise KeyError("Cell line ID not present in gene expression data")
            
class DrugGDSC(Drug):
    """Class representing single drug in GDSC"""
    def __init__(self, drug_name, drug_id, targets):
        super().__init__(drug_name, drug_id, "GDSCv2")
        self.targets = targets
        
    def extract_binary_targets_vector(self, drug_response_df):
        """Extract binary targets vector using just data from GDSC, meaning sparse vector of 
        length [all unique putative targets of GDSC drugs] where 1 indicates presence of a 
        given gene target
        """
        all_targets = []
        for targets in drug_response_df["PUTATIVE_TARGET"].unique():
            all_targets = all_targets + targets.split(", ")
        all_targets = set(all_targets)
        return [1 if x in self.targets else 0 for x in all_targets]
    
class CellLinesDataGDSC(object):
    @staticmethod
    def extract_full_gene_expresson_data(gene_expression_df):
        df = gene_expression_df.transpose()
        df.columns = df.loc["ensembl_gene"].values
        df = df.drop("ensembl_gene")
        df.insert(0, "cosmic_id", df.index)
        # Drop cell lines with strange IDs
        df = df.drop(["1503362.1", "1330983.1", "909976.1", "905954.1"])
        df.index = df.index.map(int)
        return df.apply(pd.to_numeric)


class DrugsDataGDSC(object):
    @staticmethod
    def extract_binary_targets_vectors(drug_response_df, drug_list=None):
        """Extract binary targets vectors for all specified drugs using just data from GDSCv2, meaning 
        sparse vectors of length [all unique putative targets of specified drugs] where 1 indicates presence 
        of a given gene target."""
        # Compute the set of all considered targets
        all_targets = []
        if drug_list:
            for targets in drug_response_df[
                drug_response_df["DRUG_ID"].isin(drug_list)]["PUTATIVE_TARGET"].unique():
                all_targets = all_targets + targets.split(", ")
        else:
            for targets in drug_response_df["PUTATIVE_TARGET"].unique():
                all_targets = all_targets + targets.split(", ")
        all_targets = list(set(all_targets))
        
        # Initialize DataFrame containing binary vectors for drugs
        columns=["drug_id", "drug_name"] + all_targets
        df = pd.DataFrame(columns=columns)
        # Now iterate over drugs and create binary target vector for each of them
        if drug_list:
            for drug_id in drug_list:
                drug_name = drug_response_df[
                    drug_response_df["DRUG_ID"] == drug_id]["DRUG_NAME"].iloc[0]
                drug_targets = drug_response_df[
                    drug_response_df["DRUG_ID"] == drug_id]["PUTATIVE_TARGET"].iloc[0].split(", ")
                # Create binary target vector for this drug
                binary_targets = [1 if x in drug_targets else 0 for x in all_targets]
                row = pd.Series([drug_id, drug_name] + binary_targets, index=columns)
                df = df.append(row, ignore_index=True)
        else:
            for drug_id in drug_response_df["DRUG_ID"].unique():
                drug_name = drug_response_df[
                    drug_response_df["DRUG_ID"] == drug_id]["DRUG_NAME"].iloc[0]
                drug_targets = drug_response_df[
                    drug_response_df["DRUG_ID"] == drug_id]["PUTATIVE_TARGET"].iloc[0].split(", ")
                # Create binary target vector for this drug
                binary_targets = [1 if x in drug_targets else 0 for x in all_targets]
                row = pd.Series([drug_id, drug_name] + binary_targets, index=columns)
                df = df.append(row, ignore_index=True)
        df.index = df["drug_id"]
        df[df.columns[2:]] = df[df.columns[2:]].apply(pd.to_numeric)
        df["drug_id"] = df["drug_id"].astype(int)
        return df
    
    
    

    
def evaluate_predictions(y_true, preds):
    """Compute RMSE and correlation with true values for model predictions"""
    print("RMSE:", metrics.mean_squared_error(y_true, preds) ** 0.5)
    print("Correlation:", pearsonr(y_true, preds))
    
def extract_desired_entries(KINOMEscan_data, drugs_subset,
                               preffered_doses=[10., 1., 0.1]):
    """Compute DataFrame containing inhibition entries only for drugs and doses that you want"""
    # Compute DataFrame only with desired entries
    desired_entries = pd.DataFrame(columns=KINOMEscan_data.columns)
    for molecule in drugs_subset:
        df = KINOMEscan_data[
            KINOMEscan_data["Small Molecule"] == molecule]
        # Drop duplicate rows in terms of protein and dose
        df = df.drop_duplicates(subset=["Protein", "Compound conc. in uM"])
        # If there is only one available dose for that drug, extract all entries
        if df["Compound conc. in uM"].nunique() == 1:
            desired_entries = pd.concat([desired_entries, df], axis=0)
        # Else, add entries only with most preffered available dose    
        else:
            # Establish most preferred dose
            desired_dose = min(df["Compound conc. in uM"].unique(), key= lambda x: preffered_doses.index(x))
            # Add entries for this drug and picked dose
            desired_entries = pd.concat([desired_entries, df[df["Compound conc. in uM"] == desired_dose]],
                                   axis=0)
    return desired_entries

def compute_inhibition_profiles(KINOMEscan_data, 
                               drugs_subset,
                               kinases_set="union",
                               preffered_doses=[10., 1., 0.1]):
    """Compute matrix of shape (no. drugs, no. kinases) containin inhibition levels for kinase
    inhibitors (drugs)"""
    # First, compute entries (drugs and doses) for which you want to include the data
    desired_entries = extract_desired_entries(KINOMEscan_data,
                                drugs_subset,
                                preffered_doses)
    # Create pivot table
    drug_profiles = desired_entries.pivot(index="Small Molecule",
                                         columns="Protein",
                                         values="% Control")
    # Constraint te pivot table if needed
    if kinases_set == "union":
        return drug_profiles
    elif kinases_set == "intersection":
        # Compute intersection in terms of proteins screened for every drug
        kinases = set(list(drug_profiles))
        for molecule in desired_entries["Small Molecule"].unique():
            df = desired_entries[
                desired_entries["Small Molecule"] == molecule]
            kinases = kinases.intersection(set(df["Protein"]))
        return drug_profiles[list(kinases)]
    else:
        return drug_profiles[kinases_set]
    
    
    
class Dataset:
    """Class represening and aggregating data for drugs and cell lines"""
    def __init__(self, name, drugs_datatypes, cell_lines_datatypes,
                 description=None):
        self.name = name
        self.drug_datatypes = drugs_datatypes
        self.cell_line_datatypes = cell_lines_datatypes
        self.description = description
        
    def set_cell_lines_data(self, dataframes, 
                            features_subset=None,
                            id_column_name="cell_line_id", 
                            join_type="inner"):
        """Compute full cell line data by concatenating parsed DataFrames containing particular 
        cell lines datatypes"""
        joint_df = dataframes[0]
        if len(dataframes) > 1:
            for df in dataframes[1:]:
                joint_df = joint_df.merge(df, on="cell_line_id", how="inner")
        if features_subset:
            return joint_df[features_subset]
        else:
            self.full_cell_lines_data = joint_df
        
    def set_drugs_data(self, dataframe):
        """Set data characterizing the drugs"""
        self.drugs_data = dataframe
        
    def set_response_data(self, dataframe, response_metric="AUC"):
        """Set data with response for cell line-drug pairs""" 
        self.response_metric = "AUC"
        self.response_data = dataframe
        
    @staticmethod
    def standardize_data(dataframe, cols_subset=None, rows_subset=None):
        """Standardize data (z-score normalization) across columns."""
        if rows_subset:
            if cols_subset:
                mean = dataframe.loc[rows_subset, cols_subset].mean(axis=0)
                std = dataframe.loc[rows_subset, cols_subset].std(axis=0)
                dataframe_standard = (dataframe[cols_subset] - mean) / std
                return pd.concat([dataframe_standard, dataframe.drop(cols_subset, axis=1)], axis=1)
            else:
                mean = dataframe.loc[rows_subset].mean(axis=0)
                std = dataframe.loc[rows_subset].std(axis=0)
                return (dataframe - mean) / std
                
        else:
            if cols_subset:
                mean = dataframe[cols_subset].mean(axis=0)
                std = dataframe[cols_subset].std(axis=0)
                dataframe_standard = (dataframe[cols_subset] - mean) / std
                return pd.concat([dataframe_standard, dataframe.drop(cols_subset, axis=1)], axis=1)
            else:
                mean = dataframe.mean(axis=0)
                std = dataframe.std(axis=0)
                return (dataframe - mean) / std
            
    @staticmethod
    def samples_train_test_split(samples, num_cell_lines_val, num_cell_lines_test, seed=None,
                                shuffle=True):
        # Fix the seed for pandas and numpy shuffling to get reproducible results
        np.random.seed(seed)
        # Shuffle all the samples if desired
        if shuffle:
             samples = samples.sample(frac=1.)
    
        # Extract test cell lines samples
        cell_lines_test = list(np.random.choice(samples.COSMIC_ID.unique(), size=num_cell_lines_test,
                                               replace=False))
        samples_test = samples[samples.COSMIC_ID.isin(cell_lines_test)]

        # Extract rest
        rest = samples[~samples.COSMIC_ID.isin(cell_lines_test)]

        # Extract validation cell lines samples
        cell_lines_val = list(np.random.choice(rest.COSMIC_ID.unique(), size=num_cell_lines_val,
                                               replace=False))
        samples_val = rest[rest.COSMIC_ID.isin(cell_lines_val)]

        # Extract rest (training set)
        samples_train = rest[~rest.COSMIC_ID.isin(cell_lines_val)]
        
        return samples_train, samples_val, samples_test, cell_lines_test, cell_lines_val
        
class Results:
    def __init__(self, directory):
        self.directory = directory
        
    def get_analysis_results(self, experiment):
        """Get a table with results corresponding to the last epoch of training for every
        trial."""
        exp_dir = os.path.join(self.directory, "Experiment " + str(experiment))
        df = pd.read_csv(os.path.join(exp_dir, "analysis_tuning_results.csv"))
        return df
    
    def get_per_trial_results(self, experiment, trial, result_type="progress"):
        trial_dir = os.path.join(self.directory, "Experiment " + str(experiment), trial)
        with open(os.path.join(trial_dir, "params.json"), "r") as f:
            params = json.load(f)
        if result_type == "progress":
            df = pd.read_csv(os.path.join(trial_dir, "progress.csv"))
        elif result_type == "per_drug_train":
            df = pd.read_csv(os.path.join(trial_dir, "performance_per_drug_train.csv"))
        else:
            df = pd.read_csv(os.path.join(trial_dir, "performance_per_drug_val.csv"))
        return df, params
        

    def get_best_params(self, experiment):
        """Get best params per experiment. If experiment is "all", display best params for all 
        experiments."""
        if experiment == "all":
            for experiment in os.listdir(self.directory):
                exp_dir = os.path.join(self.directory, experiment)
                if os.path.isdir(exp_dir):
                    best_config_dir = os.path.join(exp_dir, "best_config.txt")
                    # Display best config for this experiment
                    print(experiment)
                    with open(best_config_dir, "r") as f:
                        print(f.read())
                    print()
        else:
            exp_dir = os.path.join(self.directory, "Experiment " + str(experiment))
            best_config_dir = os.path.join(exp_dir, "best_config.txt")
            # Display best config for this experiment
            with open(best_config_dir, "r") as f:
                print(f.read())
                
    def get_best_model_learning_curve(self, experiment):
        """Get results achieved for a best model in a given experiment and epochs"""
        exp_dir = os.path.join(self.directory, "Experiment " + str(experiment))
        best_model_learning_curve_df = pd.read_csv(os.path.join(exp_dir, "best_model_test_results.csv")) 
        return best_model_learning_curve_df
    
    def get_averaged_best_model_learning_curve(self):
        """Get results achieved by best models, averaged over experiments"""
        pass
        
    def get_best_model_best_result(self, experiment, metric, mode="max"):
        """Get best result of a given metric achieved by best model of the experiment.
        If experiment == "all", display average of the best metrics across experiments."""
        
        if experiment == "all":
            metrics = []
            for experiment in os.listdir(self.directory):
                exp_dir = os.path.join(self.directory, experiment)
                if os.path.isdir(exp_dir):
                    learning_curve = self.get_best_model_learning_curve(int(experiment[-1]))
                    if mode == "max":
                        metrics.append(learning_curve[metric].max())
                    else:
                        metrics.append(learning_curve[metric].min())
            return metrics
        
        # Get best model's learning curve
        learning_curve = self.get_best_model_learning_curve(experiment)
        if mode == "max":
            return learning_curve[metric].max()
        else:
            return learning_curve[metric].min()
        
    def get_best_model_per_drug_results(self, exp, mode="test"):
        """Get DataFrame with best model's test or train per drug results. If exp is "all",
        get a results from all experiment where rows are also flagged by experiment name."""
        if exp != "all":
            exp_dir = os.path.join(self.directory, "Experiment " + str(exp))
            df = pd.read_csv(os.path.join(exp_dir, "best_model_per_drug_" + mode + "_results.csv"))
            return df
        else:
            dfs = []
            for experiment in os.listdir(self.directory):
                exp_dir = os.path.join(self.directory, experiment)
                if os.path.isdir(exp_dir):
                    df = pd.read_csv(os.path.join(exp_dir, "best_model_per_drug_" + mode + "_results.csv"))
                    df["Experiment"] = [int(experiment[-1])] * df.shape[0]
                    dfs.append(df)
            return pd.concat(dfs, axis=0)
            
    
    def find_trial_with_params(self, exp, params_comb):
        """Find trials (folder names) which contain specified parameters combination."""
        exp_dir = os.path.join(self.directory, "Experiment " + str(exp))
        matching_trials = []
        for trial in os.listdir(exp_dir):
            trial_dir = os.path.join(exp_dir, trial)
            if os.path.isdir(trial_dir):
                with open(os.path.join(trial_dir, "params.json"), "r") as f:
                    trial_params = json.load(f)
                combination_in_trial = True   # Flag determining if given combination is present
                                              # in current trial
                for param in param_comb:
                    if param_comb[param] != trial_params[param]:
                        combination_in_trial = False
                if combination_in_trial:
                    matching_trials.append(trial)
        return matching_trials
    
    def get_averaged_metric_per_param_comb(self, params_comb, metric, results_type, mode):
        """Get a list of all param combinations satysfying provided combination and return
        an averaged metric over all experiments for every combination"""
        exp_dict = {}
        for exp in range(1, 6):
            matching_trials = self.find_trial_with_params(exp, param_comb)
            trial_dict = {}
            for trial in matching_trials:
                df, params = self.get_per_trial_results(exp, trial, results_type)

                param_names_to_drop = ["out_activation", "autoencoders_activ_func", "batch_size",
                                      "cell_line_hidden_dim1", "cell_line_hidden_dim2", 
                                      "drug_hidden_dim1", "drug_hidden_dim2", "epochs", "criterion",
                                      "code_dim"]
                for name in param_names_to_drop:
                    if name in params:
                        del params[name]
                if mode == "max":
                    best_metric = df[metric].max()
                else:
                    best_metric = df[metric].min()
                trial_dict[str(params)] = best_metric
            exp_dict[exp] = trial_dict
        res = {}
        for trial in exp_dict[1]:
            metrics = []
            for exp in exp_dict:
                metrics.append(exp_dict[exp][trial])
            res[trial] = metrics
        return res
                
    @staticmethod
    def plot_learning_curve(df, metric1, metric2=None, title="", ylabel=""):
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("")
        sns.lineplot(range(1, df.shape[0] + 1), df[metric1], label=metric1)
        if metric2:
            sns.lineplot(range(1, df.shape[0] + 1), df[metric2], label=metric2)
        plt.legend()
        
class Model:
    def __init__(self, name, network):
        self.name = name
        self.network = network
        
    def train(self, train_samples, cell_line_features, drug_features,
             batch_size, optimizer, criterion, reg_lambda=0, log=True):
        """Perform training process by looping over training set in batches (one epoch) of the
        training."""
        no_batches = train_samples.shape[0] // batch_size + 1
        
        # Training the model
        self.network.train()
        for batch in range(no_batches):
            # Separate response variable batch
            if batch != no_batches:
                samples_batch = train_samples.iloc[batch * batch_size:(batch + 1) * batch_size]
            else:
                samples_batch = train_samples.iloc[batch * batch_size:]

            # Extract output variable batch
            y_batch = torch.from_numpy(samples_batch["AUC"].values).view(-1, 1)

            # Extract cell lines IDs for which data shall be extracted
            cl_ids = samples_batch["COSMIC_ID"].values
            # Extract corresponding cell line data
            cell_line_input_batch = cell_line_features.loc[cl_ids].values
            cell_line_input_batch = torch.from_numpy(cell_line_input_batch)

            # Extract drug IDs for which data shall be extracted
            drug_ids = samples_batch["DRUG_ID"].values
            # Extract corresponding drug data
            drug_input_batch = drug_features.loc[drug_ids].values
            drug_input_batch = torch.from_numpy(drug_input_batch)

            # Clear gradient buffers because we don't want to accummulate gradients 
            optimizer.zero_grad()

            # Perform forward pass
            batch_output = self.network(drug_input_batch.float(), cell_line_input_batch.float())

            reg_sum = 0
            for param in self.network.parameters():
                reg_sum += 0.5 * (param ** 2).sum()  # L2 norm

            # Compute the loss for this batch
            loss = criterion(batch_output, y_batch.float()) + reg_lambda * reg_sum
            # Get the gradients w.r.t. the parameters
            loss.backward()
            # Update the parameters
            optimizer.step()
        return loss
    
    def predict(self, samples, cell_line_features, drug_features):
        """Predict response on a given set of samples"""
        y_true = samples["AUC"].values

        cl_input = cell_line_features.loc[samples["COSMIC_ID"].values].values
        drug_input = drug_features.loc[samples["DRUG_ID"].values].values

        self.network.eval()
        with torch.no_grad():
            predicted = self.network(torch.from_numpy(drug_input).float(), 
                             torch.from_numpy(cl_input).float())
        return predicted, y_true
    
    @staticmethod
    def per_drug_performance_df(samples, predicted, mean_training_auc=None):
        """Compute evaluation metrics per drug and return them in a DataFrame"""
        sample_with_predictions = samples.copy()
        sample_with_predictions["Predicted AUC"] = predicted.numpy()

        drugs = []
        model_corrs = []
        model_rmses = []
        dummy_corrs = []
        dummy_rmses = []
        no_samples = []

        for drug in sample_with_predictions.DRUG_ID.unique():
            df = sample_with_predictions[sample_with_predictions.DRUG_ID == drug]
            if df.shape[0] < 2:
                continue
            if mean_training_auc:
                dummy_preds = [mean_training_auc] * df.shape[0]
            else:
                dummy_preds = [df["AUC"].mean()] * df.shape[0]
            dummy_rmse = metrics.mean_squared_error(df["AUC"], dummy_preds) ** 0.5
            dummy_corr = pearsonr(df["AUC"], dummy_preds)

            try:
                model_rmse = metrics.mean_squared_error(df["AUC"], df["Predicted AUC"]) ** 0.5
                model_corr = pearsonr(df["AUC"], df["Predicted AUC"])
            except ValueError:
                model_rmse, model_corr = np.nan, (np.nan, np.nan)

            drugs.append(drug)
            dummy_rmses.append(dummy_rmse)
            dummy_corrs.append(dummy_corr[0])

            model_rmses.append(model_rmse)
            model_corrs.append(model_corr[0])

            no_samples.append(df.COSMIC_ID.nunique())

        performance_per_drug = pd.DataFrame()
        performance_per_drug["Drug ID"] = drugs
        performance_per_drug["Model RMSE"] = model_rmses
        performance_per_drug["Model correlation"] = model_corrs

        performance_per_drug["Dummy RMSE"] = dummy_rmses
        performance_per_drug["Dummy correlation"] = dummy_corrs
        performance_per_drug["No. samples"] = no_samples

        return performance_per_drug
        
    @staticmethod
    def evaluate_predictions(y_true, preds):
        """Compute RMSE and correlation with true values for model predictions"""
        return metrics.mean_squared_error(y_true, preds) ** 0.5, pearsonr(y_true, preds)
    
class ModelWithAutoencoders(Model):
    def train(self, train_samples, cell_line_features, drug_features,
             batch_size, optimizer, criterion, reconstruction_term_drug=0.0,
              reconstruction_term_cl=0.0, reg_lambda=0.0, log=True):
        """Perform training process by looping over training set in batches (one epoch) of the
        training."""
        no_batches = train_samples.shape[0] // batch_size + 1
        # Establish the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if log:
          print(device)
        # Move the network into device
        self.network.to(device)
        # Training the model
        self.network.train()
        for batch in range(no_batches):
            # Separate response variable batch
            if batch != no_batches:
                samples_batch = train_samples.iloc[batch * batch_size:(batch + 1) * batch_size]
            else:
                samples_batch = train_samples.iloc[batch * batch_size:]

            # Extract output variable batch
            y_batch = torch.from_numpy(samples_batch["AUC"].values).view(-1, 1).to(device)

            # Extract cell lines IDs for which data shall be extracted
            cl_ids = samples_batch["COSMIC_ID"].values
            # Extract corresponding cell line data
            cell_line_input_batch = cell_line_features.loc[cl_ids].values
            cell_line_input_batch = torch.from_numpy(cell_line_input_batch).to(device)

            # Extract drug IDs for which data shall be extracted
            drug_ids = samples_batch["DRUG_ID"].values
            # Extract corresponding drug data
            drug_input_batch = drug_features.loc[drug_ids].values
            drug_input_batch = torch.from_numpy(drug_input_batch).to(device)

            # Clear gradient buffers because we don't want to accummulate gradients 
            optimizer.zero_grad()

            # Perform forward pass
            batch_output, batch_drug_reconstruction, batch_cl_reconstruction = self.network(
                drug_input_batch.float(), cell_line_input_batch.float())

            reg_sum = 0
            for param in self.network.parameters():
                reg_sum += 0.5 * (param ** 2).sum()  # L2 norm

            # Compute the loss for this batch
            output_loss = criterion(batch_output, y_batch.float()) + reg_lambda * reg_sum
            drug_recounstruction_loss = criterion(batch_drug_reconstruction, drug_input_batch.float())
            cl_reconstruction_loss = criterion(batch_cl_reconstruction, cell_line_input_batch.float())

            # Sum the losses in the final cost function
            loss = output_loss + reconstruction_term_drug * drug_recounstruction_loss + reconstruction_term_cl * cl_reconstruction_loss
            # Get the gradients w.r.t. the parameters
            loss.backward()
            # Update the parameters
            optimizer.step()
            
        return loss, drug_recounstruction_loss, cl_reconstruction_loss
    
    def predict(self, samples, cell_line_features, drug_features):
        """Predict response on a given set of samples"""
        # Establish the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        y_true = samples["AUC"].values

        cl_input = cell_line_features.loc[samples["COSMIC_ID"].values].values
        drug_input = drug_features.loc[samples["DRUG_ID"].values].values

        self.network.eval()
        with torch.no_grad():
            predicted = self.network(torch.from_numpy(drug_input).to(device).float(), 
                             torch.from_numpy(cl_input).to(device).float())
        return predicted, y_true, drug_input, cl_input
    