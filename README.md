# DiagnosticMultimodal
Laptop GPU in Hybrid mode (supergfxctl -m Hybrid). If the mode is Vfio, the GPU is reserved for VMs and unusable for training
## Choix techniques

### Modèle de classification : GoogLeNet (InceptionV1)
- **Source des poids** : TensorFlow Hub, `google/imagenet/inception_v1/feature_vector/5` (pré-entraîné sur ImageNet)
- **Entrée** : images 224×224×3 (RGB)
- **Normalisation** : pixels divisés par 255 → [0, 1]
- **Sortie** : vecteur de 1024 features
- **Limite** : le modèle Hub est un bloc unique (gelé ou dégelé en entier). InceptionV3 sera entraîné en comparaison pour un fine-tuning partiel des dernières couches.

### Environnement
- Python 3.11, TensorFlow 2.21 avec `tf-keras` (Keras 2, requis par TF Hub)
- `setuptools<81` : TF Hub dépend encore de `pkg_resources`
- GPU NVIDIA en mode Hybrid (`supergfxctl -m Hybrid`). En mode Vfio, le GPU est réservé aux VM et inutilisable.
- `src/utils/gpu.py` : précharge `libcusolver` (bug de packaging TF 2.21), active Keras 2 et l'allocation mémoire progressive
