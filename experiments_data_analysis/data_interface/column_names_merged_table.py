from enum import Enum

class Col(str, Enum):
    RUN_NAME = "run-name"

    # /// Experiment Settings ///
    NUM_CLIENTS = "num-clients"
    MODEL = "model"
    REGULARIZATION = "regularization"
    DATA_DISTRIBUTION = "data_distribution"
    LOCAL_EPOCHS = "local-epochs"
    NUM_SERVER_ROUNDS = "num-server-rounds"

    # /// Flower Training Metrics ///
    RUNTIME_FLOWER = "Runtime_Flower"
    TEST_ACCURACY_SERVER = "Test-Accuracy (Server)"
    TEST_LOSS_SERVER = "Test-Loss (Server)"
    TRAININGS_ACCURACY_CLIENTS_AGGREGIERT = "Trainings-Accuracy (Clients aggregiert)"
    TRAININGS_LOSS_CLIENTS_AGGREGIERT = "Trainings-Loss (Clients aggregiert)"
    VALIDIERUNGS_ACCURACY_CLIENTS_AGGREGIERT = "Validierungs-Accuracy (Clients aggregiert)"
    VALIDIERUNGS_LOSS_CLIENTS_AGGREGIERT = "Validierungs-Loss (Clients aggregiert)"

    OVERFITTING_GAP_LOSS_CLIENTS = "Overfitting Gap Loss (Clients)"
    OVERFITTING_GAP_ACC_CLIENTS = "Overfitting Gap Acc. (Clients)"
    OVERFITTING_GAP_ACC = "Overfitting Gap Acc."
    OVERFITTING_GAP_LOSS = "Overfitting Gap Loss"

    # /// MIA Settings ///
    NUM_SHADOW_MODELS = "num_shadow_models"
    SHADOW_EPOCHS = "shadow_epochs"
    TRAIN_SIZE = "train_size"

    # /// MIA Metrics ///
    RUNTIME_MIA = "Runtime_MIA"
    METRICS_AUC = "metrics/AUC"
    METRICS_ACCURACY = "metrics/Accuracy"
    METRICS_F1_SCORE = "metrics/F1-Score"
    METRICS_PRECISION = "metrics/Precision"
    METRICS_RECALL = "metrics/Recall"
    STD_STD_AUC = "std/std_AUC"
    STD_STD_ACCURACY = "std/std_Accuracy"
    STD_STD_F1_SCORE = "std/std_F1-Score"
    STD_STD_PRECISION = "std/std_Precision"
    STD_STD_RECALL = "std/std_Recall"

