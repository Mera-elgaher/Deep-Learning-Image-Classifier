import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ImageClassifier:
    def __init__(self, num_classes=10, input_shape=(32, 32, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """Build CNN model from scratch"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='ResNet50'):
        """Build model using transfer learning"""
        if base_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'VGG16':
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
    def prepare_data(self):
        """Load and preprocess CIFAR-10 dataset"""
        # Load CIFAR-10 data
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1
        )
        datagen.fit(x_train)
        
        return (x_train, y_train), (x_test, y_test), class_names, datagen
    
    def train_model(self, x_train, y_train, x_val, y_val, datagen=None, 
                   epochs=50, batch_size=32):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        if datagen:
            # Train with data augmentation
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                steps_per_epoch=len(x_train) // batch_size
            )
        else:
            # Train without data augmentation
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks
            )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test, class_names):
        """Evaluate model performance"""
        # Predictions
        test_loss, test_accuracy, test_top5 = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Detailed predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_names))
        
        return test_accuracy, y_pred_classes, y_true_classes
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def predict_single_image(self, image, class_names):
        """Predict single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return class_names[predicted_class], confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and training script
if __name__ == "__main__":
    print("🧠 Deep Learning Image Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = ImageClassifier(num_classes=10)
    
    # Prepare data
    print("📊 Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test), class_names, datagen = classifier.prepare_data()
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Classes: {class_names}")
    
    # Build model (choose one)
    print("\n🏗️ Building model...")
    
    # Option 1: CNN from scratch
    # classifier.build_cnn_model()
    
    # Option 2: Transfer Learning (recommended)
    classifier.build_transfer_learning_model('ResNet50')
    
    # Compile model
    classifier.compile_model(learning_rate=0.001)
    
    # Display model summary
    print("\n📋 Model Summary:")
    classifier.model.summary()
    
    # Train model
    print("\n🚀 Training model...")
    history = classifier.train_model(
        x_train, y_train, 
        x_test, y_test, 
        datagen=datagen,
        epochs=30,
        batch_size=32
    )
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    test_accuracy, y_pred, y_true = classifier.evaluate_model(x_test, y_test, class_names)
    
    # Plot results
    print("\n📊 Plotting results...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Save model
    classifier.save_model('image_classifier_model.h5')
    
    print(f"\n✅ Training completed! Final accuracy: {test_accuracy:.2%}")
