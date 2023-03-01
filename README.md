# Cats_VS_Dogs_VGG16
Classification of Cats vs Dogs Kaggle dataset using VGG16 ( Transfer Learning )

## Dataset

The Kaggle [cats vs dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) dataset consists of **25,000** labeled images of **cats** and **dogs**. The images are of varying sizes and aspect ratios, and the goal is to train a **CNN** model to correctly classify new images of cats and dogs. The dataset is divided into two sets: a **training set** with **1,000** images and a **validation set** with **5,000** images.

## Transfer Learning

**Transfer learning** is a machine learning technique that involves using a **pre-trained** model for a new task. In transfer learning, the pre-trained model's weights and architecture are used as a starting point for a new model that is fine-tuned for the new task. This approach can be useful when the new dataset is small, as it allows the model to learn from the pre-trained model's knowledge and generalize better to new data.

## VGG16

**VGG16** is a pre-trained **CNN** model that was trained on the **ImageNet** dataset, which consists of millions of **labeled** images of objects from **1,000** different classes. The **VGG16** model is composed of **16 layers**, including **convolutional layers**, **pooling layers**, and **fully connected layers**. The model is known for its simplicity and effectiveness in image **classification** tasks, and its architecture has been used as a starting point for many **CNN** models.

## Code
The code begins by importing several libraries, including **ImageDataGenerator** from **tensorflow.keras.preprocessing.image**, **VGG16** from **tensorflow.keras.applications.vgg16**, and some plotting libraries. It sets the size of the **input** images to be **150x150** pixels and specifies the paths to the **training** and **validation** image datasets.

It then uses the **ImageDataGenerator** class to generate augmented **training** images and rescale **validation** images. The **ImageDataGenerator** class performs data augmentation on the training images, including random rotations, shifts, and flips, to increase the size of the training dataset and reduce overfitting.

Next, the code loads the **pre-trained VGG16** model using the input_shape of **150x150** pixels and three color channels, freezes the weights of all its layers, and adds a **fully connected** output layer with a **softmax** activation function. The code compiles the model by specifying the **loss function**, **optimizer**, and **evaluation metric**.

Finally, it fits the model using the **fit_generator** method, which trains the model using the augmented training images and validates it using the validation images. The code saves the training and validation loss and accuracy metrics for plotting.

Overall, this code demonstrates how transfer learning can be used to train a CNN model for image classification on a small dataset by leveraging a pre-trained model's feature extraction capabilities. The output of this code is a trained CNN model that can classify new images into the classes of the training dataset.
