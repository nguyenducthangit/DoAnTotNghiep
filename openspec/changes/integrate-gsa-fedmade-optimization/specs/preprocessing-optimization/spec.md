# Spec: Preprocessing Optimization (GSA Feature Selection)

## Overview

This specification defines the requirements for implementing GSA (Gravitational Search Algorithm) based feature selection in the data preprocessing pipeline to optimize the feature set for IoT network attack detection.

## ADDED Requirements

### Requirement: GSA Feature Selection Implementation

The system MUST implement a Gravitational Search Algorithm to automatically select the most discriminative features from the CICIoT2023 dataset.

#### Scenario: Run GSA on Cleaned Dataset

**Given** a cleaned and merged dataset with 46 features and labeled attack types  
**When** the user runs the GSA feature selection with configured parameters  
**Then** the system should:
- Initialize a population of candidate feature subsets
- Evaluate each subset's fitness using validation accuracy
- Iteratively refine the population using gravitational forces
- Converge to an optimal feature subset within the maximum iterations
- Return a binary mask indicating selected features

**And** the number of selected features should be between 15 and 30  
**And** the selected feature subset should achieve at least 90% validation accuracy

#### Scenario: Save GSA Results

**Given** a completed GSA feature selection run  
**When** the algorithm finishes  
**Then** the system should save:
- A JSON file containing the list of selected feature names
- A PNG image showing the GSA convergence curve (fitness vs iterations)
- A JSON file with feature importance scores

**And** the files should be saved to `Output/gsa_results/` directory  
**And** the JSON format should be valid and include metadata (num_iterations, best_fitness, etc.)

#### Scenario: Filter Dataset to Selected Features

**Given** a dataset with all 46 original features  
**And** a set of GSA-selected features  
**When** the user applies feature filtering  
**Then** the system should:
- Create a new dataframe containing only the selected features
- Preserve the order of samples (rows)
- Maintain the label column unchanged

**And** all subsequent preprocessing steps should use only the filtered features

#### Scenario: Handle GSA Edge Cases

**Given** a GSA feature selection run  
**When** the algorithm selects fewer than 10 features  
**Then** the system should log a warning  
**And** use a fallback strategy (e.g., top N features by variance)

**When** the algorithm fails to converge within max_iterations  
**Then** the system should return the best solution found so far  
**And** log a warning about incomplete convergence

### Requirement: GSA Configuration

The system MUST support configurable GSA parameters through the training configuration file.

#### Scenario: Load GSA Configuration

**Given** a training_config.yaml file with GSA section  
**When** the preprocessing notebook loads the configuration  
**Then** the system should read:
- `gsa.enabled`: Boolean flag to enable/disable GSA
- `gsa.num_features_to_select`: Target number of features
- `gsa.max_iterations`: Maximum GSA iterations
- `gsa.population_size`: Number of candidate solutions
- `gsa.gravitational_constant`: Initial gravitational constant G
- `gsa.alpha`: Gravitational constant decay rate
- `gsa.use_sample`: Whether to use a data sample for speed
- `gsa.sample_fraction`: Fraction of data to use if sampling
- `gsa.random_seed`: Random seed for reproducibility

#### Scenario: Toggle GSA On/Off

**Given** a configuration with `gsa.enabled = false`  
**When** the preprocessing pipeline runs  
**Then** the system should skip GSA feature selection  
**And** use all 46 original features for training

**Given** a configuration with `gsa.enabled = true`  
**When** the preprocessing pipeline runs  
**Then** the system should execute GSA feature selection  
**And** filter the dataset to selected features only

### Requirement: GSA Performance Optimization

The system MUST optimize GSA execution time for large datasets.

#### Scenario: Use Data Sampling for GSA

**Given** a configuration with `gsa.use_sample = true` and `gsa.sample_fraction = 0.2`  
**When** GSA runs fitness evaluation  
**Then** the system should:
- Sample 20% of the dataset for fitness evaluation
- Use stratified sampling to maintain class distribution
- Still apply the selected features to the full dataset after GSA

#### Scenario: Fast Fitness Evaluation

**Given** a feature subset being evaluated for fitness  
**When** the system trains the evaluation classifier  
**Then** the system should use a lightweight model (Random Forest with limited depth)  
**And** the training time per subset should be less than 10 seconds

**And** the fitness evaluation should use 80-20 train-validation split

### Requirement: GSA Reproducibility

The system MUST ensure GSA results are reproducible given the same configuration and random seed.

#### Scenario: Reproducible Feature Selection

**Given** a configuration with `gsa.random_seed = 42`  
**When** GSA is run multiple times with the same configuration  
**Then** the selected feature set should be identical across runs  
**And** the convergence curve should be identical

### Requirement: Integration with Existing Pipeline

GSA feature selection MUST integrate seamlessly with the existing preprocessing workflow.

#### Scenario: GSA in Preprocessing Sequence

**Given** the data preprocessing notebook  
**When** executed in sequence  
**Then** the workflow should be:
1. Load and merge CSV files
2. Clean data (handle missing values, remove duplicates)
3. **[GSA Step]** Run GSA feature selection if enabled
4. **[GSA Step]** Filter dataset to selected features
5. Encode labels to numeric values
6. Normalize features using MinMaxScaler (on selected features only)
7. Partition data for federated clients

**And** the TabTransformer model should be constructed with the reduced feature count

#### Scenario: Feature Metadata Propagation

**Given** a dataset filtered by GSA to 20 features  
**When** the data is partitioned for clients  
**Then** each client dataset should contain exactly 20 features  
**And** the test set should contain exactly 20 features  
**And** the feature names should be saved with each partition for reference

## MODIFIED Requirements

### Requirement: Feature Configuration (Modified)

The existing feature configuration in `includes.py` MUST be updated to support dynamic feature sets.

#### Scenario: Load Selected Features from File

**Given** a file `Output/gsa_results/selected_features.json` exists  
**When** the training notebook loads feature configuration  
**Then** the system should:
- Check if GSA was enabled
- If yes, load the selected feature list from JSON
- Use the selected features instead of the default `X_columns`

**And** all downstream modules should use the dynamically loaded feature set

## Success Criteria

✅ GSA successfully reduces features from 46 to 15-30  
✅ Selected features achieve ≥90% validation accuracy during GSA  
✅ GSA completes within 60 minutes on full dataset (with sampling)  
✅ Results are reproducible with the same random seed  
✅ All output files (JSON, PNG) are generated correctly  
✅ Downstream pipeline works seamlessly with reduced feature set
