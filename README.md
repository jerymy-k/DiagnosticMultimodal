# DiagnosticMultimodal

## Choix techniques

### Modèle de classification : InceptionV3
- **Choix** : InceptionV3 (famille GoogLeNet/Inception), **validé par le formateur** à la place de GoogLeNet (InceptionV1)
- **Pourquoi** : disponible nativement dans Keras (`tf.keras.applications.InceptionV3`) et découpé en blocs nommés (`mixed0` … `mixed10`), ce qui permet de dégeler uniquement les dernières couches pour le fine-tuning. InceptionV1 n'existe que sur TensorFlow Hub, sous forme d'un bloc unique (gelé ou dégelé en entier), et impose Keras 2.
- **Poids** : pré-entraînés sur ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
- **Entrée** : images 299×299×3 (RGB)
- **Normalisation** : `preprocess_input` d'InceptionV3 → pixels dans [-1, 1]
- **Sortie** : vecteur de 2048 features

### Environnement
- Python 3.11, TensorFlow 2.21 (Keras 3) avec `tensorflow[and-cuda]`
- GPU NVIDIA en mode Hybrid (`supergfxctl -m Hybrid`). En mode Vfio, le GPU est réservé aux VM et inutilisable pour l'entraînement.
- `src/utils/gpu.py` : précharge `libcusolver` (bug de packaging TF 2.21) et active l'allocation mémoire progressive du GPU
