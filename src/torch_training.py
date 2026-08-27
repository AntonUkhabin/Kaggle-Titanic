import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import config
from src.torch_models import build_torch_model
from src.train_functions import build_feature_pipeline
from src.utils import set_seed


def create_data_loader(features, labels, batch_size, shuffle, seed, num_workers):
    '''Create a reproducible PyTorch DataLoader.'''

    features_tensor = torch.as_tensor(features, dtype=torch.float32)

    if labels is None:
        dataset = TensorDataset(features_tensor)
    else:
        labels_tensor = torch.tensor(np.asarray(labels), dtype=torch.float32)
        dataset = TensorDataset(features_tensor, labels_tensor)

    generator = torch.Generator().manual_seed(seed)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=num_workers)

    return data_loader


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    '''Train a PyTorch binary classifier for one epoch.'''

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = loss_function(logits, targets)

        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        batch_size = features.size(0)

        running_loss += loss.item() * batch_size
        correct += (predictions == targets).sum().item()
        total += batch_size

    mean_loss = running_loss / total
    accuracy = correct / total

    return mean_loss, accuracy


def validate_one_epoch(model, data_loader, loss_function, device):
    '''Evaluate a PyTorch binary classifier for one epoch.'''

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = loss_function(logits, targets)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            batch_size = features.size(0)

            running_loss += loss.item() * batch_size
            correct += (predictions == targets).sum().item()
            total += batch_size

    mean_loss = running_loss / total
    accuracy = correct / total

    return mean_loss, accuracy


def train_fold(model, train_loader, val_loader, loss_function, optimizer, device, epochs, early_stopping_rounds, min_delta, checkpoint_path, fold, log_interval, scheduler=None):
    '''Train one fold with mandatory early stopping.'''

    if log_interval < 1:
        raise ValueError('log_interval must be greater than or equal to 1.')

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_function, device)
        learning_rate = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(learning_rate)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            checkpoint = {
                'fold': fold,
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }

            torch.save(checkpoint, checkpoint_path)
            improvement_marker = '*'
        else:
            epochs_without_improvement += 1
            improvement_marker = ''

        # Scheduler обновляется после валидации
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        should_stop = epochs_without_improvement >= early_stopping_rounds
        should_log = epoch == 0 or (epoch + 1) % log_interval == 0 or epoch == epochs - 1 or should_stop

        if should_log:
            print(f'Fold {fold} | {epoch + 1:03d}/{epochs} | loss: {train_loss:.4f}/{val_loss:.4f} | acc: {train_accuracy:.4f}/{val_accuracy:.4f} | lr: {learning_rate:.8f} {improvement_marker}')

        if should_stop:
            print(f'Fold {fold} | Early stopping on epoch {epoch + 1}')
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_idx = best_epoch - 1
    print(
        f'Fold {fold} | {best_epoch:03d}/{epochs} | '
        f'loss: {history["train_loss"][best_idx]:.4f}/'
        f'{history["val_loss"][best_idx]:.4f} | '
        f'acc: {history["train_accuracy"][best_idx]:.4f}/'
        f'{history["val_accuracy"][best_idx]:.4f} | '
        f'lr: {history["learning_rate"][best_idx]:.8f}'
    )
    print(f'Fold {fold} | Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.4f}')

    return model, history, best_epoch, best_val_loss


def predict_probabilities(model, data_loader, device):
    '''Predict positive-class probabilities with a binary PyTorch model.'''

    model.eval()
    probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device)
            logits = model(features)
            batch_probabilities = torch.sigmoid(logits)
            probabilities.append(batch_probabilities.cpu())

    return torch.cat(probabilities).numpy()


def cross_validate_neural_network(train_cv_df, target_col, config):
    '''Run cross-validation for a binary PyTorch neural network.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = train_cv_df[target_col]

    skf = StratifiedKFold(n_splits=config.split.n_splits, shuffle=config.dataloader_params.shuffle, random_state=config.training.fold_seed)

    model_name = config.model.active
    model_params = config.model.models[model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores = []
    best_epochs = []
    fold_models = []
    histories = []
    oof_probabilities = np.full(len(features), np.nan, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f'\nFold {fold + 1}/{config.split.n_splits}')

        set_seed(config.general.seed)

        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]
        labels_train = labels.iloc[train_idx]
        labels_val = labels.iloc[val_idx]

        feature_pipeline = build_feature_pipeline(config)

        features_train_processed = feature_pipeline.fit_transform(features_train, labels_train).astype(np.float32)
        features_val_processed = feature_pipeline.transform(features_val).astype(np.float32)

        train_loader = create_data_loader(features_train_processed, labels_train, model_params.batch_size, True, config.general.seed, model_params.num_workers)
        val_loader = create_data_loader(features_val_processed, labels_val, model_params.batch_size, False, config.general.seed, model_params.num_workers)

        # Choose Loss, Optimizer. Add sheduler if need
        model = build_torch_model(model_name, input_size=features_train_processed.shape[1]).to(device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params.learning_rate)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=30,
        #     eta_min=1e-6,
        # )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=4,
        #     threshold=1e-5,
        #     threshold_mode='abs',
        #     min_lr=1e-7,
        # )

        checkpoint_path = Path(config.paths.path_to_checkpoints) / config.general.experiment_name / f'fold_{fold}.pt'

        model, history, best_epoch, best_val_loss = train_fold(model, train_loader, val_loader, loss_function, optimizer, device, model_params.epochs, model_params.early_stopping_rounds, model_params.min_delta, checkpoint_path, fold, config.logging.torch_log_interval, scheduler)

        validation_probabilities = predict_probabilities(model, val_loader, device)
        validation_predictions = (validation_probabilities >= 0.5).astype(int)

        oof_probabilities[val_idx] = validation_probabilities
        fold_accuracy = accuracy_score(labels_val, validation_predictions)

        scores.append(float(fold_accuracy))
        best_epochs.append(best_epoch)
        histories.append(history)
        fold_models.append((feature_pipeline, model))

        print(f'Fold {fold} | Accuracy: {fold_accuracy:.4f} | Best epoch: {best_epoch} | Best loss: {best_val_loss:.4f}')

    return scores, best_epochs, fold_models, oof_probabilities, histories


def predict_with_neural_network_ensemble(features, fold_models, config, threshold=0.5):
    '''Predict by averaging probabilities from PyTorch fold models.'''

    model_params = config.model.models[config.model.active]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_probabilities = []

    for feature_pipeline, model in fold_models:
        features_processed = feature_pipeline.transform(features).astype(np.float32)
        data_loader = create_data_loader(features_processed, None, model_params.batch_size, False, config.general.seed, model_params.num_workers)

        model = model.to(device)
        probabilities = predict_probabilities(model, data_loader, device)
        fold_probabilities.append(probabilities)

    mean_probabilities = np.mean(fold_probabilities, axis=0)
    predictions = (mean_probabilities >= threshold).astype(int)

    return predictions, mean_probabilities