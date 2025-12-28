## ADDED Requirements

### Requirement: Data Loading and Preprocessing
The system SHALL provide a data preprocessing pipeline that loads the CICIoT2023 dataset, cleans it, and prepares it for federated learning training.

#### Scenario: Load large CSV dataset efficiently
- **GIVEN** 169 CSV files totaling ~12GB in the `DataSets/` directory
- **WHEN** the data loading function is called
- **THEN** all CSV files SHALL be loaded using chunking to avoid memory overflow
- **AND** the combined dataset SHALL contain all required features from `includes.py::X_columns`
- **AND** memory usage SHALL not exceed 8GB during loading

#### Scenario: Clean and validate data
- **GIVEN** loaded raw dataset
- **WHEN** data cleaning is performed
- **THEN** all rows with missing values SHALL be either imputed or removed
- **AND** duplicate rows SHALL be removed
- **AND** all feature columns SHALL have numeric data types
- **AND** the label column SHALL contain valid attack class names

#### Scenario: Encode attack labels
- **GIVEN** cleaned dataset with string attack labels
- **WHEN** label encoding is applied
- **THEN** all 34 attack classes SHALL be mapped to numeric values (0-33) according to `includes.py::dict_34_classes`
- **AND** the label encoder SHALL be saved to `Output/models/label_encoder.pkl`
- **AND** a human-readable mapping SHALL be saved to `Output/models/labels.json`

#### Scenario: Normalize features
- **GIVEN** cleaned dataset with numeric features
- **WHEN** normalization is applied
- **THEN** all features SHALL be scaled to [0, 1] using MinMaxScaler
- **AND** the fitted scaler SHALL be saved to `Output/models/scaler.pkl`
- **AND** the scaler SHALL be reusable for new data during inference

#### Scenario: Partition data for federated learning (Non-IID)
- **GIVEN** preprocessed and normalized dataset
- **WHEN** data partitioning is performed
- **THEN** the dataset SHALL be split into 5 client partitions with Non-IID distribution
- **AND** each client SHALL have different attack class distributions (simulating real-world heterogeneity)
- **AND** 20% of total data SHALL be reserved as test set
- **AND** all partitions SHALL be saved as `.npz` files in `Output/data/`

---

### Requirement: Federated Learning Model Architecture
The system SHALL define a Deep Neural Network architecture suitable for multi-class IoT attack detection.

#### Scenario: Create DNN model
- **GIVEN** 46 input features and 34 output classes
- **WHEN** the model is created
- **THEN** the model SHALL have the following architecture:
  - Input layer: 46 neurons
  - Dense layer 1: 128 neurons with ReLU activation and Dropout(0.3)
  - Dense layer 2: 64 neurons with ReLU activation and Dropout(0.3)
  - Dense layer 3: 32 neurons with ReLU activation and Dropout(0.3)
  - Output layer: 34 neurons with Softmax activation
- **AND** the model SHALL be compiled with Adam optimizer (lr=0.001)
- **AND** the loss function SHALL be sparse categorical crossentropy
- **AND** the model SHALL track accuracy as a metric

#### Scenario: Extract and set model weights
- **GIVEN** a trained model
- **WHEN** weights are extracted
- **THEN** the function SHALL return a list of numpy arrays representing all layer weights
- **AND** the weights SHALL be settable on another model instance of the same architecture

---

### Requirement: Federated Learning Training Process
The system SHALL implement the FedAvg (Federated Averaging) algorithm to train a global model across multiple simulated clients.

#### Scenario: Client local training
- **GIVEN** a client with local data partition and global model weights
- **WHEN** local training is triggered
- **THEN** the client SHALL train the model on its local data for 5 epochs
- **AND** the client SHALL use batch size 256
- **AND** the client SHALL return updated model weights after training
- **AND** the client SHALL NOT share raw data with the server

#### Scenario: Server aggregates client weights (FedAvg)
- **GIVEN** updated weights from 5 clients after local training
- **WHEN** the server performs aggregation
- **THEN** the server SHALL compute the element-wise average of all client weights
- **AND** the server SHALL update the global model with aggregated weights
- **AND** the aggregation SHALL weight all clients equally (simple average)

#### Scenario: Federated training loop
- **GIVEN** initialized global model and 5 clients with partitioned data
- **WHEN** federated training is executed
- **THEN** the system SHALL run for 30-50 rounds
- **AND** each round SHALL consist of:
  1. Server broadcasts current global model to all clients
  2. All 5 clients train locally and return weights
  3. Server aggregates client weights using FedAvg
  4. Server updates global model
  5. Server evaluates global model on test set
- **AND** training history (loss, accuracy per round) SHALL be saved to `Output/metrics/training_history.json`
- **AND** the final global model SHALL be saved to `Output/models/global_model.h5`

---

### Requirement: Model Evaluation and Metrics
The system SHALL evaluate the trained global model on the test set and generate comprehensive performance metrics.

#### Scenario: Calculate classification metrics
- **GIVEN** trained global model and test set
- **WHEN** evaluation is performed
- **THEN** overall test accuracy SHALL be calculated and SHALL be >95%
- **AND** per-class Precision, Recall, and F1-Score SHALL be calculated for all 34 classes
- **AND** all 34 classes SHALL have F1-Score >0.85
- **AND** a confusion matrix (34x34) SHALL be generated
- **AND** all metrics SHALL be saved to `Output/metrics/metrics_report.json`

#### Scenario: Visualize training performance
- **GIVEN** training history with loss and accuracy per round
- **WHEN** visualization is generated
- **THEN** a line plot SHALL be created showing:
  - Accuracy vs Round number
  - Loss vs Round number
- **AND** the plot SHALL be saved to `Output/metrics/accuracy_plot.png`

#### Scenario: Visualize confusion matrix
- **GIVEN** confusion matrix from test set predictions
- **WHEN** visualization is generated
- **THEN** a heatmap SHALL be created using seaborn
- **AND** the heatmap SHALL clearly show diagonal dominance (correct predictions)
- **AND** misclassifications between similar attack types SHALL be highlighted
- **AND** the heatmap SHALL be saved to `Output/metrics/confusion_matrix.png`

#### Scenario: Visualize per-class performance
- **GIVEN** per-class F1-Scores
- **WHEN** visualization is generated
- **THEN** a bar chart SHALL be created showing F1-Score for each of 34 classes
- **AND** classes with F1-Score <0.85 SHALL be highlighted in red
- **AND** the chart SHALL be saved to `Output/metrics/f1_scores_per_class.png`

---

### Requirement: Model Export and Artifacts
The system SHALL export all necessary artifacts for future use in web application integration.

#### Scenario: Export trained model
- **GIVEN** final trained global model
- **WHEN** export is performed
- **THEN** the model SHALL be saved in Keras H5 format to `Output/models/global_model.h5`
- **AND** the saved model SHALL be loadable using `keras.models.load_model()`
- **AND** the loaded model SHALL produce identical predictions to the trained model

#### Scenario: Export preprocessing artifacts
- **GIVEN** fitted scaler and label encoder
- **WHEN** export is performed
- **THEN** the scaler SHALL be saved as pickle to `Output/models/scaler.pkl`
- **AND** the label encoder SHALL be saved as pickle to `Output/models/label_encoder.pkl`
- **AND** a JSON mapping file SHALL be created at `Output/models/labels.json` with format:
  ```json
  {
    "0": "BenignTraffic",
    "1": "DDoS-RSTFINFlood",
    ...
    "33": "DictionaryBruteForce"
  }
  ```

#### Scenario: Verify exported artifacts for web integration
- **GIVEN** all exported files in `Output/models/` and `Output/metrics/`
- **WHEN** verification is performed
- **THEN** the following files SHALL exist:
  - `Output/models/global_model.h5`
  - `Output/models/scaler.pkl`
  - `Output/models/label_encoder.pkl`
  - `Output/models/labels.json`
  - `Output/metrics/training_history.json`
  - `Output/metrics/metrics_report.json`
  - `Output/metrics/confusion_matrix.png`
  - `Output/metrics/accuracy_plot.png`
- **AND** all files SHALL be readable and loadable
- **AND** a test inference SHALL successfully run: load model → load scaler → normalize sample → predict → decode label

---

### Requirement: Configuration Management
The system SHALL use a YAML configuration file to manage all training hyperparameters.

#### Scenario: Load training configuration
- **GIVEN** a YAML config file at `Notebooks/configs/training_config.yaml`
- **WHEN** training is initiated
- **THEN** all hyperparameters SHALL be loaded from the config file including:
  - `num_clients`: 5
  - `num_rounds`: 30-50
  - `local_epochs`: 5
  - `batch_size`: 256
  - `learning_rate`: 0.001
  - `model_architecture`: [128, 64, 32]
  - `dropout_rate`: 0.3
  - `test_split_ratio`: 0.2
- **AND** the configuration SHALL be versioned and saved with training outputs for reproducibility

#### Scenario: Override configuration via notebook
- **GIVEN** loaded configuration from YAML
- **WHEN** a notebook overrides specific parameters (e.g., num_rounds=40)
- **THEN** the overridden value SHALL be used during training
- **AND** both original and overridden configs SHALL be logged in training history
