# char_recognition
MNIST / EMNIST / reconnaissance sur des photos et des dessins de chiffres

Ce readme correspond au brief 13.

Dans ce répertoire, vous retrouverez trois notebooks. Chacun d'eux entrainent un modèle.

## *first_mnist.ipynb*

Ce notebook entraîne au modèle de reconnaissance de chiffres manuscrits. 

Dans mon modèle initialement dans la couche conv2D j'appliquais 32 filtres et lançais 12 epochs. 
J'obtenais  un résultat en 1,24 minutes et une val_accuracy de 0,99.

J'ai changé le nombre de filtre de la couche conv2D j'applique 16 filtres et lance 4 epochs.
J'obtiens un résultat en 17 secondes et une val_accuracy de 0,97.

## *emnist_model.ipynb*

Voici mon modèle initial : 

model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])
J'obtenais un résultat en 15 minutes et 10 secondes et un val_accuracy de 0,92 avec 10 epochs

Voici le modèle modifié :

model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(16,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])

J'obtiens un resultat en 2,29 minutes avec val_accuracy de 0,90 avec 4 epochs

## *recognition_digit.ipynb*

La redéfinition des images n'est pas bien paramétrée. Malheureusement les résultats sont loin d'être excellents. Il faudra que je réadapte les filtres à appliquer sur mes images.

Je dois fignoler la modification des images et des paramètres du model.
