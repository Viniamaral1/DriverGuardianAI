"""
Main training script for DriverGuardianAI.

Production training pipeline:

- Load configuration
- Set seed
- Load dataset
- Split dataset
- Fit preprocessing only on training data
- Transform validation data
- Create dataloaders
- Build model
- Train model
- Evaluate model
"""


import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split


from src.config import load_config

from src.dataset import load_data


from src.preprocess import (
    fit_preprocessor,
    transform_dataset
)


from src.dataloader import create_dataloaders


from src.fatigue_model import FatigueResidualNN


from src.losses import get_loss


from src.trainer import Trainer


from src.evaluate import evaluate_model


from src.utils import (
    save_preprocessor,
    set_seed
)




def main():


    print("=" * 60)

    print("DriverGuardianAI")

    print("Driver Fatigue Detection Training Pipeline")

    print("=" * 60)



    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(42)



    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config(
        "config/config.yaml"
    )


    print("\nConfiguration loaded.")



    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        f"\nUsing device: {device}"
    )



    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print(
        "\nLoading dataset..."
    )


    df = load_data(
        config["dataset"]["path"]
    )


    print(
        f"Dataset shape: {df.shape}"
    )



    # --------------------------------------------------
    # Split before preprocessing
    # --------------------------------------------------

    print(
        "\nSplitting dataset..."
    )


    train_df, val_df = train_test_split(

        df,

        test_size=config["dataset"]["test_size"],

        random_state=config["dataset"]["random_state"],

        stratify=df[
            config["dataset"]["target"]
        ]

    )


    print(
        f"Training samples: {len(train_df)}"
    )


    print(
        f"Validation samples: {len(val_df)}"
    )



    # --------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------

    print(
        "\nFitting preprocessing on training data..."
    )


    train_df, preprocessing = fit_preprocessor(
        train_df
    )


    print(
        "Training preprocessing complete."
    )



    print(
        "\nTransforming validation data..."
    )


    val_df = transform_dataset(

        val_df,

        preprocessing

    )


    print(
        "Validation transformation complete."
    )



    # --------------------------------------------------
    # Save preprocessing
    # --------------------------------------------------

    print(
        "\nSaving preprocessing artifacts..."
    )


    save_preprocessor(

        preprocessing,

        config["paths"]["preprocessing"]

    )


    print(
        "Preprocessing artifacts saved."
    )



    # --------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------

    X_train = train_df.drop(

        config["dataset"]["target"],

        axis=1

    )
    print("\nModel Features:")
    print(X_train.columns.tolist())


    y_train = train_df[

        config["dataset"]["target"]

    ]



    X_val = val_df.drop(

        config["dataset"]["target"],

        axis=1

    )


    y_val = val_df[

        config["dataset"]["target"]

    ]



    print(
        "\nClasses:"
    )


    print(
        y_train.value_counts()
    )



    # --------------------------------------------------
    # No weighted sampler
    #
    # Experiment7 showed that aggressive sampling
    # caused over prediction of fatigue classes.
    #
    # We use controlled class weighting instead.
    # --------------------------------------------------

    print(
        "\nUsing controlled focal loss weighting..."
    )


    sampler = None



    class_weights = torch.tensor(

        [

            1.0,   # Alert

            1.2,   # Mild fatigue

            1.8    # Moderate/Severe fatigue

        ],

        dtype=torch.float32

    )



    class_weights = class_weights.to(device)



    print(
        "Class weights:"
    )

    print(
        class_weights
    )



    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader, val_loader = create_dataloaders(

        X_train,

        X_val,

        y_train,

        y_val,

        sampler=sampler,

        batch_size=config["training"]["batch_size"]

    )


    print(
        "\nDataloaders created."
    )



    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = FatigueResidualNN(

        input_dim=X_train.shape[1],

        hidden_dims=config["model"]["hidden_dims"],

        dropout=config["model"]["dropout"],

        num_classes=config["model"]["num_classes"]

    )


    model.to(device)



    print(
        "\nModel created."
    )



    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = get_loss(

        config["loss"]["type"],

        class_weights

    )



    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(

        model.parameters(),

        lr=config["training"]["learning_rate"]

    )



    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        device=device

    )



    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print(
        "\nStarting training...\n"
    )


    trainer.fit(

        epochs=config["training"]["epochs"],

        patience=config["training"]["patience"],

        save_path=config["paths"]["model"]

    )



    # --------------------------------------------------
    # Load best model
    # --------------------------------------------------

    print(
        "\nLoading best model..."
    )


    model.load_state_dict(

        torch.load(

            config["paths"]["model"],

            map_location=device

        )

    )



    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print(
        "\nEvaluating model..."
    )


    metrics, report, y_true, y_pred = evaluate_model(

        trainer,

        experiment_name=config["evaluation"]["experiment_name"]

    )



    print(
        "\n" + "=" * 60
    )


    print(
        "Training completed successfully!"
    )


    print(
        "=" * 60
    )



    print(
        "\nFinal Metrics:"
    )


    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )




if __name__ == "__main__":

    main()