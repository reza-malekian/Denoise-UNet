import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Import utility functions and model definition
from data_utils import load_images_from_folder
from model_unet import build_unet


# ======================================================
# 1️⃣ Load Training and Validation Data
# ======================================================

# Define data directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_folder = os.path.join(base_dir, 'data', 'noisy', 'train')
val_folder = os.path.join(base_dir, 'data', 'noisy', 'validation')

print("🔹 Loading training and validation data...")
x_train = load_images_from_folder(train_folder)
x_val = load_images_from_folder(val_folder)

print(f"✅ Data loaded successfully.")
print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")


# ======================================================
# 2️⃣ Build U-Net Model
# ======================================================
print("\n🔹 Building U-Net model ...")
model = build_unet(input_shape=(128,128,3))
model.summary()


# ======================================================
# 3️⃣ Compile the Model
# ======================================================
print("\n🔹 Compiling model with optimizer = Adam and loss = MSE ...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)
print("✅ Model is ready for training.\n")


# ======================================================
# 4️⃣ Setup Checkpoints and Early Stopping
# ======================================================
os.makedirs("../results/checkpoints", exist_ok=True)

checkpoint_path = "../results/checkpoints/unet_best.h5"

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)



# ======================================================
# 5️⃣ Train the Model
# ======================================================
print("🚀 Starting model training ...")

history = model.fit(
    x_train,           # Input: noisy images
    x_train,           # Target: reconstruct the same image (denoising)
    epochs=30,
    batch_size=8,
    validation_data=(x_val, x_val),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("✅ Model training completed successfully.")


# ======================================================
# 6️⃣ Save Final Model
# ======================================================
final_model_path = "../results/checkpoints/unet_final.h5"
model.save(final_model_path)
print(f"💾 Final model saved at: {final_model_path}")


# ======================================================
# 7️⃣ Plot Training Curves
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs("../results/visuals", exist_ok=True)
plt.savefig("../results/visuals/training_curve.png")
plt.show()
print("📈 Training curve saved at results/visuals/training_curve.png")
