from enum import Enum


class ColFlower(str, Enum):

    STEP = "_step"
    RUNTIME = "_runtime"

    VALIDIERUNGS_LOSS_CLIENTS_AGGREGIERT = "Validierungs-Loss (Clients aggregiert)"
    VALIDIERUNGS_ACCURACY_CLIENTS_AGGREGIERT = "Validierungs-Accuracy (Clients aggregiert)"
    TRAININGS_LOSS_CLIENTS_AGGREGIERT = "Trainings-Loss (Clients aggregiert)"
    TRAININGS_ACCURACY_CLIENTS_AGGREGIERT = "Trainings-Accuracy (Clients aggregiert)"
    TEST_LOSS_SERVER = "Test-Loss (Server)"
    TEST_ACCURACY_SERVER = "Test-Accuracy (Server)"

    OVERFITTING_GAP_LOSS = "Overfitting-Gap (Server, Loss)"
    OVERFITTING_GAP_ACC = "Overfitting-Gap (Server, Accuracy)"
    OVERFITTING_GAP_LOSS_CLIENTS = "Overfitting-Gap (Clients aggregiert, Loss)"
    OVERFITTING_GAP_ACC_CLIENTS = "Overfitting-Gap (Clients aggregiert, Accuracy)"
