from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from membership_inference_attack.utils.types_config_mia import MiaConfig
from datasets import Dataset
from membership_inference_attack.utils.wandb_logging_mia import log_class_distribution

# !! Not used in the current implementation, but can be useful for future extensions !!
# It splits the shadow train/test datasets into iid or non-iid distributions based on the configuration
# In my tests it doesn't make any differences if we use Shokri's shadow model data pooling approach or this one
# Could be useful for different scenarios


def sample_shadow_datasets_iid(shadow_train_dataset: Dataset, shadow_test_dataset: Dataset,
                               config: MiaConfig) -> tuple[list[Dataset], list[Dataset]]:
    num_shadow_models = config.parameters.num_shadow_models
    seed = config.parameters_static.seed
    class_names = config.parameters_static.class_names

    shadow_train_sets = []
    shadow_test_sets = []

    # Partition shadow_train
    shadow_train_dataset = shadow_train_dataset.shuffle(seed=seed)
    train_partitioner = IidPartitioner(
        num_partitions=num_shadow_models,
    )
    train_partitioner.dataset = shadow_train_dataset

    # Partition shadow_test
    shadow_test_dataset = shadow_test_dataset.shuffle(seed=seed)  # shuffle dataset to ensure reproducibility
    test_partitioner = IidPartitioner(
        num_partitions=num_shadow_models,
    )
    test_partitioner.dataset = shadow_test_dataset

    for i in range(num_shadow_models):
        train_data = train_partitioner.load_partition(partition_id=i)
        shadow_train_sets.append(train_data)

        test_data = test_partitioner.load_partition(partition_id=i)
        shadow_test_sets.append(test_data)

        log_class_distribution(
            hf_dataset=train_data,
            wandb_cluster_name="shadow_train_distribution",
            wandb_plot_prefix=f"shadow_model_{i + 1}",
            class_names=class_names
        )

    return shadow_train_sets, shadow_test_sets


def sample_shadow_datasets_with_non_iid(
        shadow_train_dataset: Dataset,
        shadow_test_dataset: Dataset,
        config: MiaConfig
) -> tuple[list[Dataset], list[Dataset]]:
    num_shadow_models = config.parameters.num_shadow_models
    alpha = 0.5
    seed = config.parameters_static.seed
    class_names = config.parameters_static.class_names

    shadow_train_sets = []
    shadow_test_sets = []

    # Partition shadow_train
    train_partitioner = DirichletPartitioner(
        num_partitions=num_shadow_models,
        partition_by="label",
        alpha=alpha,
        min_partition_size=0,
        seed=seed
    )
    train_partitioner.dataset = shadow_train_dataset

    # Partition shadow_test
    shadow_test_dataset = shadow_test_dataset.shuffle(seed=seed)  # shuffle dataset to ensure reproducibility
    test_partitioner = IidPartitioner(
        num_partitions=num_shadow_models,
    )

    test_partitioner.dataset = shadow_test_dataset

    for i in range(num_shadow_models):
        # Train data for shadow model i
        train_data = train_partitioner.load_partition(partition_id=i)
        shadow_train_sets.append(train_data)

        log_class_distribution(
            hf_dataset=train_data,
            wandb_cluster_name="shadow_train_distribution",
            wandb_plot_prefix=f"shadow_model_{i + 1}_train",
            class_names=class_names
        )

        # Test data for shadow model i
        test_data = test_partitioner.load_partition(partition_id=i)
        shadow_test_sets.append(test_data)

    return shadow_train_sets, shadow_test_sets
