"""
Trainer class for DriverGuardianAI.

Handles:
- Training loop
- Validation
- Metrics
- Checkpoint saving
- Early stopping
- Probability extraction for threshold tuning
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)



class Trainer:


    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=None
    ):


        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


        print(
            f"Using device: {self.device}"
        )


        self.model = model.to(
            self.device
        )


        self.train_loader = train_loader

        self.val_loader = val_loader


        self.criterion = criterion.to(
            self.device
        )


        self.optimizer = optimizer

        self.scheduler = scheduler



        self.history = {

            "train_loss": [],

            "val_loss": [],

            "accuracy": [],

            "f1": [],

            "recall": []

        }



    def train_one_epoch(self):


        self.model.train()


        total_loss = 0



        for X, y in self.train_loader:


            X = X.to(
                self.device
            )

            y = y.to(
                self.device
            )


            self.optimizer.zero_grad()


            outputs = self.model(
                X
            )


            loss = self.criterion(
                outputs,
                y
            )


            loss.backward()


            self.optimizer.step()



            total_loss += loss.item()



        return total_loss / len(
            self.train_loader
        )




    def validate_one_epoch(
        self,
        return_preds=False
    ):


        self.model.eval()


        total_loss = 0


        predictions = []

        targets = []

        probabilities = []



        with torch.no_grad():


            for X, y in self.val_loader:


                X = X.to(
                    self.device
                )

                y = y.to(
                    self.device
                )


                outputs = self.model(
                    X
                )


                loss = self.criterion(
                    outputs,
                    y
                )


                total_loss += loss.item()



                # Convert logits to probabilities

                probs = torch.softmax(
                    outputs,
                    dim=1
                )



                preds = torch.argmax(
                    probs,
                    dim=1
                )



                predictions.extend(
                    preds.cpu().numpy()
                )


                targets.extend(
                    y.cpu().numpy()
                )


                probabilities.extend(
                    probs.cpu().numpy()
                )



        avg_loss = total_loss / len(
            self.val_loader
        )


        accuracy = accuracy_score(
            targets,
            predictions
        )


        f1 = f1_score(
            targets,
            predictions,
            average="weighted"
        )


        recall = recall_score(
            targets,
            predictions,
            average="weighted"
        )



        if return_preds:


            return (

                avg_loss,

                accuracy,

                f1,

                recall,

                targets,

                predictions,

                probabilities

            )



        return (

            avg_loss,

            accuracy,

            f1,

            recall

        )





    def fit(
        self,
        epochs,
        save_path,
        patience=10
    ):


        best_loss = float(
            "inf"
        )


        counter = 0



        for epoch in range(
            epochs
        ):


            train_loss = self.train_one_epoch()



            val_loss, acc, f1, recall = (
                self.validate_one_epoch()
            )



            self.history["train_loss"].append(
                train_loss
            )


            self.history["val_loss"].append(
                val_loss
            )


            self.history["accuracy"].append(
                acc
            )


            self.history["f1"].append(
                f1
            )


            self.history["recall"].append(
                recall
            )



            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Accuracy: {acc:.4f} | "
                f"F1: {f1:.4f} | "
                f"Recall: {recall:.4f}"
            )



            if val_loss < best_loss:


                best_loss = val_loss

                counter = 0



                torch.save(
                    self.model.state_dict(),
                    save_path
                )


                print(
                    "✅ Best model saved!"
                )


            else:


                counter += 1



            if self.scheduler:

                self.scheduler.step(
                    val_loss
                )



            if counter >= patience:


                print(
                    "\n🛑 Early stopping triggered."
                )

                break