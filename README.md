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




## Comparaison des runs et modèles retenus

Les runs sont suivis dans MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). La sélection se fait sur le jeu de **validation** ; le test sert uniquement à confirmer la généralisation.

### Classification (`brain-tumor-classification`)
Métrique principale : **F1 macro** en validation, avec le **recall** en complément (en médical, une tumeur manquée coûte plus cher qu'une fausse alerte).

| Run | Phase | Dégel à partir de | F1 val | Recall val | Accuracy test | Precision test | Recall test | F1 test |
|---|---|---|---|---|---|---|---|---|
| **inceptionv3-fine-tuning-mixed8** | fine-tuning | `mixed8` | **0.959** | **0.957** | **0.962** | 0.959 | **0.964** | **0.961** |
| inceptionv3-fine-tuning-mixed9 | fine-tuning | `mixed9` | 0.952 | 0.948 | 0.959 | 0.959 | 0.961 | 0.960 |
| inceptionv3-feature-extraction | feature extraction | — | 0.913 | 0.910 | 0.911 | 0.911 | 0.919 | 0.914 |

**Modèle retenu : fine-tuning à partir de `mixed8`** (`models/classifier.keras`)
- Meilleur F1 macro et meilleur recall en validation.
- Le fine-tuning apporte +4.6 points de F1 en validation par rapport à la feature extraction : les features ImageNet gelées ne suffisent pas pour des IRM, il faut adapter les derniers blocs.
- Dégeler un bloc de plus (`mixed8`, 11.1 M paramètres entraînables, contre 6.1 M pour `mixed9`) améliore encore légèrement les résultats.
- Pas de sur-apprentissage visible sur le test : F1 test 0.961, proche du F1 validation 0.959.

### Détection (`brain-tumor-detection`)
Métrique principale : **mAP@0.5:0.95** en validation (elle mesure aussi la précision de la boîte, pas seulement la détection).

| Run | Precision val | Recall val | mAP@0.5 val | mAP@0.5:0.95 val | Temps d'entraînement | Taille des poids |
|---|---|---|---|---|---|---|
| **yolov8s** | **0.962** | **0.943** | **0.972** | **0.775** | 22.5 min | 22 Mo |
| yolov8n | 0.946 | 0.936 | 0.966 | 0.771 | 13.5 min | 6.5 Mo |

Les deux runs utilisent la même configuration (`config/detection.yaml` : 50 epochs, imgsz 512, batch 16).

**Modèle retenu : YOLOv8s** (`models/detector.pt`)
- Meilleur sur toutes les métriques de validation, en particulier la precision (+1.6 point, donc moins de fausses boîtes).
- Le gain en mAP@0.5:0.95 reste modeste (+0.004), mais le coût est acceptable : environ 4 ms par image sur GPU, ce qui suffit pour l'API.
- Sur le test : precision 0.931, recall 0.906, mAP@0.5 0.944, mAP@0.5:0.95 0.732.
