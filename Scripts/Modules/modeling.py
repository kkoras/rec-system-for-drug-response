import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        