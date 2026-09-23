"""Reduced delivered-fleet shapes that must not become split relations."""

from __future__ import annotations


def train_accuracy_progress(model, y_train, threshold: float = 0.9):
    """Training-set progress accuracy is not reported evaluation."""
    loss = model.loss(y_train)
    loss.backward()

    train_accuracy = accuracy(model(y_train), y_train)
    if train_accuracy >= threshold:
        return model
    return model


def train_with_disjoint_selection(model, y_train, y_validation):
    """A checkpoint on a distinct target root creates no range relation."""
    train_loss = model.loss(y_train)
    train_loss.backward()

    selection_score = model.loss(y_validation)
    selection_floor = float("inf")
    best_state = None
    if selection_score < selection_floor:
        selection_floor = selection_score
        best_state = {
            name: value.clone()
            for name, value in model.state_dict().items()
        }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
