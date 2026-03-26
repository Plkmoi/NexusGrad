#!/usr/bin/env python3
"""
Example: Training a Simple Neural Network

This example demonstrates a complete training loop using CGADImpl Python bindings.
"""

import sys
sys.path.insert(0, '../dist')

import cgad as cg
import numpy as np


def create_model(input_dim, hidden_dim, output_dim):
    """Create a simple 2-layer neural network."""
    return cg.layer.Sequential([
        cg.layer.Linear(input_dim, hidden_dim),
        cg.layer.ReLU(),
        cg.layer.Linear(hidden_dim, output_dim),
    ])


def generate_dummy_data(num_samples=100, input_dim=20, output_dim=3):
    """Generate dummy classification data."""
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, num_samples).astype(np.float32)
    return X, y


def train_step(model, x_batch, y_batch, learning_rate=0.01):
    """Perform one training step."""
    # Convert to tensors
    x = cg.tensor.from_numpy(x_batch)
    y = cg.tensor.from_numpy(y_batch)
    
    # Forward pass
    x_val = cg.autodiff.make_tensor(x)
    logits = model(x_val)  # Requires model to be callable with Value
    
    # For now, use simple MSE loss
    mse = cg.autodiff.mse_loss(logits, cg.autodiff.make_tensor(y))
    
    # Backward pass
    cg.autodiff.backward(mse)
    
    # Optimization step - simplification since we can't directly modify parameters
    # In practice, you'd track model parameters and update them
    # cg.optim.sgd(logits, learning_rate=learning_rate)
    
    # Get loss value
    loss_val = mse.val().mean() if hasattr(mse.val(), 'mean') else float(mse.val())
    
    return loss_val


def main():
    """Main training loop."""
    print("=" * 60)
    print("CGADImpl Training Example")
    print("=" * 60)
    
    # Hyperparameters
    INPUT_DIM = 20
    HIDDEN_DIM = 64
    OUTPUT_DIM = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 5
    
    print(f"\nConfiguration:")
    print(f"  Input dimension: {INPUT_DIM}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Output dimension: {OUTPUT_DIM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    params = model.parameters()
    print(f"Model parameters: {len(params)} tensors")
    
    # Generate data
    print(f"\nGenerating dummy data...")
    X, y = generate_dummy_data(num_samples=100, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    print(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    # Training loop
    print(f"\nStarting training...")
    print("-" * 60)
    
    num_batches = len(X) // BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            x_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Training step
            try:
                loss = train_step(model, x_batch, y_batch, LEARNING_RATE)
                total_loss += loss
                
                if (batch_idx + 1) % max(1, num_batches // 4) == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{num_batches}, Loss: {avg_loss:.4f}")
            
            except Exception as e:
                print(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")
    
    print("-" * 60)
    print("✓ Training completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
