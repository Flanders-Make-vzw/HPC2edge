DROP TABLE IF EXISTS results CASCADE;
DROP TABLE IF EXISTS submissions CASCADE;
DROP TABLE IF EXISTS edge_devices CASCADE;

-- Neural network architectures
CREATE TABLE network_architectures (
    id SERIAL PRIMARY KEY,
    created TIMESTAMP DEFAULT now(),
    name VARCHAR (50) NOT NULL,
    hyperparameters JSONB NOT NULL
);


-- It would have been better to use the modern syntax "integer PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY" rather than the old-fashioned "serial PRIMARY KEY", as the latter doesn't automatically propagate GRANT INSERT to appropriate permissions to the underlying SEQUENCE, so  you have to remember to manually GRANT USAGE, SELECT to the SEQUENCE.


-- For quickly looking up recent architectures.
CREATE INDEX IF NOT EXISTS network_architecture_created_idx
  ON network_architectures (created);

-- Benchmarks

-- Corresponds to a subset and slight simplification of the math_function table in OpenML.
-- We have no need for any other types of math_function (KernelFunction and Metric), nor for many of the esoteric EvaluationFunctions.
CREATE TABLE evaluation_measure (
    id SERIAL PRIMARY KEY,
    name VARCHAR (50) NOT NULL,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    unit VARCHAR (50),
    higherIsBetter BOOLEAN,
    description TEXT,
    source_code TEXT,
    created TIMESTAMP DEFAULT now()  -- Corresponds to OpenML's 'date' column.
    -- functionType VARCHAR (50) DEFAULT 'EvaluationFunction'
);

INSERT INTO evaluation_measure (id, name, min_value, max_value, unit, higherIsBetter, description, source_code, created) VALUES
(52, 'relative_absolute_error', 0, 1, '', false, 'The Relative Absolute Error (RAE) is the mean absolute error (MAE) divided by the mean prior absolute error (MPAE).', 'See WEKA''s Evaluation class', '2014-12-31 20:00:00'),
(53, 'root_mean_prior_squared_error', 0, 1, '', false, 'The Root Mean Prior Squared Error (RMPSE) is the Root Mean Squared Error (RMSE) of the prior (e.g., the default class prediction).', 'See WEKA''s Evaluation class', '2014-12-31 20:00:00'),
(54, 'root_mean_squared_error', 0, 1, '', false, 'The Root Mean Squared Error (RMSE) measures how close the model''s predictions are to the actual target values. It is the square root of the Mean Squared Error (MSE), the sum of the squared differences between the predicted value and the actual value. For classification, the 0/1-error is used. See: http://en.wikipedia.org/wiki/Mean_squared_error', 'See WEKA''s Evaluation class', '2014-12-31 20:00:00'),
(55, 'root_relative_squared_error', 0, 1, '', false, 'The Root Relative Squared Error (RRSE) is the Root Mean Squared Error (RMSE) divided by the Root Mean Prior Squared Error (RMPSE). See root_mean_squared_error and root_mean_prior_squared_error.', 'See WEKA''s Evaluation class', '2014-12-31 20:00:00');


CREATE TABLE task_type (
    id SERIAL PRIMARY KEY,
    name VARCHAR (50) NOT NULL,
    description TEXT
);

INSERT INTO task_type (id, name, description) VALUES
(2, 'Supervised Regression', 'Given a dataset with a numeric target and a set of train/test splits, e.g. generated by a cross-validation procedure, train a model and return the predictions of that model.');

-- Taken from https://github.com/openml/OpenML/blob/develop/data/sql/estimation_procedure_type.sql
CREATE TABLE estimation_procedure_type (
    name VARCHAR (50) PRIMARY KEY,
    description TEXT
);

INSERT INTO estimation_procedure_type (name, description) VALUES
('crossvalidation', 'Cross-validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a test set to evaluate it. <br><br>\r\n\r\nIn k-fold cross-validation, the original sample is randomly partitioned into k equal size subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged (or otherwise combined) to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.<br><br>\r\n\r\nFor classification problems, one typically uses stratified k-fold cross-validation, in which the folds are selected so that each fold contains roughly the same proportions of class labels.<br><br>\r\n\r\nIn repeated cross-validation, the cross-validation procedure is repeated n times, yielding n random partitions of the original sample. The n results are again  averaged (or otherwise combined) to produce a single estimation.<br><br>\r\n\r\nOpenML generates train-test splits given the number of folds and repeats, so that different users can evaluate their models with the same splits. Stratification is applied by default for classification problems (unless otherwise specified). The splits are given as part of the task description as an ARFF file with the row id, fold number, repeat number and the class (TRAIN or TEST). The uploaded predictions should be labeled with the fold and repeat number of the test instance, so that the results can be properly evaluated and aggregated. OpenML stores both the per fold/repeat results and the aggregated scores.'),
('customholdout', 'A custom holdout partitions a set of observations into a training set and a test set in a predefined way. This is typically done in order to compare the performance of different predictive algorithms on the same data, as part of  a data mining competition or by the researcher who first uses the dataset.\r\n\r\n'),
('holdout', 'Holdout or random subsampling is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a test set to evaluate it. <br>\r\n\r\nIn a k% holdout, the original sample is randomly partitioned into a test set containing k% of the input sample size, and a 1-k% training set. Sampling is done without replacement. This holdout is usually repeated n times, yielding n random partitions of the original sample. The n results are averaged (or otherwise combined) to produce a single estimation.<br><br>\r\n\r\nFor classification problems, one typically uses stratified sampling, so that the test set contains roughly the same proportions of class labels as the original sample.<br><br>\r\n\r\nOpenML generates train-test splits given the percentage size of the holdout and the number of repeats, so that different users can evaluate their models with the same splits. Stratification is applied by default for classification problems (unless otherwise specified). The splits are given as part of the task description as an ARFF file with the row id, fold number (0/1), repeat number and the class (TRAIN or TEST). The uploaded predictions should be labeled with the fold and repeat number of the test instance, so that the results can be properly evaluated and aggregated. OpenML stores both the per fold/repeat results and the aggregated scores.'),
('learningcurve', 'Description to be added'),
('leaveoneout', 'Leave-on-out is a special case of cross-validation where the number of folds equals the number of instances. Thus, models are always evaluated on one instance and trained on all others.<br><br>\r\n\r\nLeave-one-out is deterministic, bias-free, and does not require repeats or stratification. However, it is very computationally intensive and thus only advised for small data sets.<br><br>\r\n\r\nFor leave-one-out, OpenML does not provide a train-test split file, but does require that the uploaded predictions are labeled with the row id of the test instance, so that the results can be properly evaluated and aggregated. OpenML stores both the per fold/repeat results and the aggregated scores.'),
('testthentrain', 'Description to be added');


-- Based on https://github.com/openml/OpenML/blob/develop/data/sql/estimation_procedure.sql
CREATE TABLE estimation_procedure (
    id SERIAL PRIMARY KEY,
    ttid INTEGER NOT NULL,
    name VARCHAR (50) NOT NULL,
    ep_type VARCHAR(50) NOT NULL,
    repeats INTEGER,
    folds INTEGER,
    samples BOOLEAN,
    percentage DOUBLE PRECISION,
    stratified_sampling BOOLEAN,
    custom_testset BOOLEAN,
    created TIMESTAMP DEFAULT now(),
    FOREIGN KEY (ttid)
        REFERENCES task_type(id),
    FOREIGN KEY (ep_type)
        REFERENCES estimation_procedure_type(name)
);

INSERT INTO estimation_procedure (id, ttid, name, ep_type, repeats, folds, samples, percentage, stratified_sampling, custom_testset) VALUES
(1, 2, '33% Holdout set', 'holdout', 1, NULL, false, 33, true, false);

CREATE TABLE benchmarks (
    id SERIAL PRIMARY KEY,
    estimation_procedure INTEGER NOT NULL,
    name VARCHAR (50) UNIQUE NOT NULL, -- Could function as PK but that would make the benchmark_result table bigger.
    dataset VARCHAR (50),
    target_feature VARCHAR (50),
    task_type INTEGER NOT NULL DEFAULT 2,
    description TEXT,
    created TIMESTAMP DEFAULT now(),
    FOREIGN KEY (estimation_procedure)
        REFERENCES estimation_procedure(id),
    FOREIGN KEY (task_type)
        REFERENCES task_type(id)
);

INSERT INTO benchmarks (id, estimation_procedure, name, dataset, target_feature, description) VALUES
(1, 1, 'RAISE-LPBF-Laser training hold-out', 'RAISE-LPBF-Laser', '', ''),
(2, 1, 'Mini RAISE-LPBF-Laser training hold-out', 'Subset of RAISE-LPBF-Laser', '', '');


CREATE TABLE benchmark_result (
    id SERIAL PRIMARY KEY,
    benchmark INTEGER NOT NULL,
    network_architecture INTEGER NOT NULL,
    evaluation_measure INTEGER NOT NULL,
    value DOUBLE PRECISION,
    created TIMESTAMP DEFAULT now(),
    FOREIGN KEY (benchmark)
        REFERENCES benchmarks(id),
    FOREIGN KEY (network_architecture)
        REFERENCES network_architectures(id),
    FOREIGN KEY (evaluation_measure)
        REFERENCES evaluation_measure(id)
);

CREATE TABLE edge_devices (
    id SERIAL PRIMARY KEY,
    name VARCHAR (50) NOT NULL,
    vram_gib DOUBLE PRECISION -- memory available to the accelerator (GPU) of the device, in gibibytes.
);

CREATE TABLE edge_measurements (
    id SERIAL PRIMARY KEY,
    created TIMESTAMP DEFAULT now(),
    network_architecture_id INTEGER NOT NULL,
    device_id INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL, -- in milliseconds
    results JSONB NOT NULL,
    FOREIGN KEY (network_architecture_id)
        REFERENCES network_architectures (id)
        ON DELETE CASCADE,
    FOREIGN KEY (device_id)
        REFERENCES edge_devices (id)
);

-- For quickly looking up recent results.
CREATE INDEX IF NOT EXISTS edge_measurement_created_idx
  ON edge_measurements (created);

INSERT INTO edge_devices(name, vram_gib) VALUES
    ('Jetson Orin Nano 4GB',4),
    ('Jetson Orin Nano 8GB',8),
    ('Jetson Orin Nano Developer Kit',8),
    ('Jetson Orin NX 8GB',8),
    ('Jetson Orin NX 16GB',16),
    ('Jetson AGX Orin 32GB',32),
    ('Jetson AGX Orin Industrial',64),
    ('Jetson AGX Orin 64GB',64),
    ('Jetson AGX Orin Developer Kit',64);
