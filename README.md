# DiagnosticMultimodal

Aide au diagnostic de tumeurs cérébrales sur IRM, avec deux modèles complémentaires :

- **Classification** (InceptionV3, transfer learning) : type de tumeur parmi *Glioma*, *Meningioma*, *No Tumor*, *Pituitary*.
- **Détection** (YOLOv8) : localisation de la tumeur avec une boîte englobante.

Le projet couvre toute la chaîne MLOps : préparation des données, entraînement, suivi des expériences avec **MLflow**, orchestration avec **Apache Airflow**, mise à disposition des modèles avec une API **FastAPI**, et interface utilisateur **Streamlit**.

## Architecture

```
                  ┌──────────────────── Airflow : brain_tumor_pipeline ────────────────────┐
 data/ (brut) ──► │ check_and_prepare_datasets → validate_data → train_classifier →        │
                  │ train_detector → evaluate_and_log_mlflow → save_models_and_artifacts   │
                  └────────────────────────────────┬───────────────────────────────────────┘
                                                   │ runs, métriques, Model Registry
                                                   ▼
                                     MLflow (mlflow.db + mlruns/)
                                                   │ models/ (champion)
                                                   ▼
                     Streamlit (app/) ──HTTP──► FastAPI (api/) ──► classifier.keras + detector.pt
```

## Structure du projet

```
├── airflow/
│   ├── dags/brain_tumor_pipeline.py   # DAG (6 tâches), appelle uniquement src/
│   ├── start_airflow.sh               # lance Airflow en mode standalone
│   └── requirements.txt               # Airflow (environnement séparé)
├── api/
│   ├── main.py                        # app FastAPI : /health, /predict/classification, /predict/detection
│   ├── schemas.py                     # modèles Pydantic des réponses
│   ├── config.py                      # chemins des modèles, seuils par défaut
│   └── services/                      # chargement des modèles, prétraitement, lecture des images
├── app/streamlit_app.py               # interface utilisateur (appelle l'API)
├── config/
│   ├── data.yaml                      # chemins, classes, split, augmentation
│   ├── classification.yaml            # hyperparamètres InceptionV3
│   ├── detection.yaml                 # hyperparamètres et augmentation YOLOv8
│   └── pipeline.yaml                  # candidats YOLO, dossiers, mode rapide
├── models/                            # modèles retenus (champions) et leurs métadonnées
├── notebooks/
│   ├── main.ipynb                     # exploration + classification
│   └── detection.ipynb                # préparation YOLO + détection
├── src/
│   ├── data/                          # préparation et validation des données
│   ├── classification/                # modèle, entraînement, évaluation, export
│   ├── detection/                     # entraînement, évaluation, export YOLOv8
│   ├── utils/                         # config, MLflow, GPU
│   └── pipeline.py                    # point d'entrée de chaque étape du pipeline
├── tests/                             # tests pytest de l'API
├── pyproject.toml / uv.lock           # dépendances du projet (uv)
└── requirements.txt                   # export des dépendances (pip)
```

## Installation

Prérequis : Python 3.11, [uv](https://docs.astral.sh/uv/), un GPU NVIDIA conseillé pour l'entraînement (l'API fonctionne sur CPU).

```bash
uv sync
```

Sans uv : `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.

**Airflow** est installé dans un environnement séparé (`airflow/.venv`) : ses dépendances entrent en conflit avec celles de TensorFlow et PyTorch. On utilise le fichier de contraintes officiel :

```bash
uv venv airflow/.venv --python 3.11
uv pip install --python airflow/.venv/bin/python -r airflow/requirements.txt \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.11.txt
```

### Données

Le dataset brut doit être placé dans `data/` (non versionné) :

```
data/
├── Train/<classe>/images/*.jpg  et  Train/<classe>/labels/*.txt   (format YOLO)
└── Val/<classe>/images/*.jpg    et  Val/<classe>/labels/*.txt
```

La préparation (étape `check_and_prepare_datasets`) génère `processed/` :
- `processed/split.csv` : split stratifié 70 / 15 / 15 (seed 42), commun aux deux modèles ;
- `processed/classification/{train,val,test}/<classe>/` : le train est équilibré par augmentation hors ligne (flou, bruit, miroir) ;
- `processed/yolo/{images,labels}/{train,valid,test}/` et `data.yaml`.

## Utilisation

### 1. Pipeline complet avec Airflow

```bash
airflow/start_airflow.sh
```

- UI : http://localhost:8080 (utilisateur `admin`, mot de passe dans `airflow/simple_auth_manager_passwords.json.generated`).
- DAG `brain_tumor_pipeline` : bouton *Trigger*. Paramètres :
  - `quick` : 1 epoch par modèle, pour tester le pipeline en quelques minutes ;
  - `force_prepare` : régénère `processed/` même s'il existe déjà ;
  - `force_save` : promeut les nouveaux modèles même s'ils ne battent pas le champion.
- Chaque tâche : 1 retry (délai 2 min), un timeout, un log d'échec (`on_failure_callback`). Les logs de `src/` apparaissent dans les logs de chaque tâche.

Les étapes peuvent aussi être lancées sans Airflow :

```bash
.venv/bin/python -m src.pipeline check_and_prepare_datasets
.venv/bin/python -m src.pipeline train_classifier --quick
```

Le DAG reste léger : chaque tâche appelle `src.pipeline` avec le Python du projet. Les parties TensorFlow et PyTorch d'une même étape tournent dans des processus séparés (voir *Environnement*).

**Promotion des modèles** : `save_models_and_artifacts` ne remplace un modèle de `models/` que s'il fait mieux sur le test (F1 macro pour la classification, mAP@0.5:0.95 pour la détection). Il est alors enregistré dans le Model Registry MLflow avec l'alias `champion`. Un entraînement rapide (`quick`) n'écrase donc jamais un bon modèle.

### 2. MLflow

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- Expériences `brain-tumor-classification` et `brain-tumor-detection` : paramètres, métriques par epoch et sur le test, courbes, matrices de confusion, poids.
- Model Registry : `brain-tumor-classifier` et `brain-tumor-detector`, alias `champion`, description (run, métriques, date).

### 3. API FastAPI

```bash
.venv/bin/uvicorn api.main:app
```

Documentation interactive : http://127.0.0.1:8000/docs

| Route | Description |
|---|---|
| `GET /health` | état de l'API et des modèles chargés |
| `POST /predict/classification` | image → `class`, `confidence`, `probabilities` |
| `POST /predict/detection` | image → `detections` (`class`, `confidence`, `bbox [x1, y1, x2, y2]`), `count` ; paramètres `conf`, `iou`, `annotated` (image annotée en base64) |

Les deux modèles sont chargés **une seule fois** au démarrage. Erreurs gérées : `415` (type de fichier), `400` (image illisible), `422` (paramètre invalide), `503` (modèle non chargé), `500` (erreur interne, trace dans les logs).

Variables d'environnement : `MODELS_DIR`, `DETECTION_CONF` (0.25), `DETECTION_IOU` (0.7).

### 4. Interface Streamlit

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Upload d'une IRM, choix du mode (classification ou détection), résultats (classe, confiance, graphique des probabilités ; image annotée, tableau des détections, seuil réglable). L'interface ne charge aucun modèle : tout passe par l'API (`API_URL`, par défaut `http://127.0.0.1:8000`). Les erreurs de l'API (injoignable, timeout, 400, 415, 503) sont affichées avec un message clair.

### 5. Tests

```bash
.venv/bin/pytest
```

14 tests de l'API : `/health`, classification, détection (dont image sans détection et image annotée), et tous les cas d'erreur (415, 400, 422, 503, 500).

## Résultats

### Classification (`brain-tumor-classification`)

Métrique principale : **F1 macro** en validation, avec le **recall** en complément (en médical, une tumeur manquée coûte plus cher qu'une fausse alerte). La sélection se fait sur la validation ; le test sert à confirmer la généralisation.

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
- Erreurs principales : faux positifs sur *Glioma* (bords irréguliers, difficiles à encadrer) et boîtes mal ajustées sur *Pituitary* (tumeurs petites).

## Choix techniques

### Données
- 5 249 images valides, 0 image corrompue. 2 images sans label et 2 labels sans image sont écartés.
- 272 fichiers de labels contiennent une boîte d'une autre classe que leur dossier (surtout *Pituitary*, sur la région de l'hypophyse). Ils sont gardés ; seules les 16 images sans aucune boîte de leur propre classe sont écartées pour YOLO.
- Le même split sert à la classification et à la détection, pour éviter qu'une image de test d'un modèle soit dans le train de l'autre.

### Modèle de classification : InceptionV3
- **Choix** : InceptionV3 (famille GoogLeNet/Inception), **validé par le formateur** à la place de GoogLeNet (InceptionV1).
- **Pourquoi** : disponible nativement dans Keras (`tf.keras.applications.InceptionV3`) et découpé en blocs nommés (`mixed0` … `mixed10`), ce qui permet de dégeler uniquement les dernières couches pour le fine-tuning. InceptionV1 n'existe que sur TensorFlow Hub, sous forme d'un bloc unique (gelé ou dégelé en entier), et impose Keras 2.
- **Poids** : pré-entraînés sur ImageNet (`include_top=False`, `pooling="avg"`), entrée 299×299×3.
- **Normalisation** : couche `Rescaling(1/127.5, offset=-1)` incluse dans le modèle (pixels dans [-1, 1]) : l'API envoie les pixels bruts.
- **Entraînement** : feature extraction (tête seule, lr 1e-3), puis fine-tuning à partir de `mixed9` ou `mixed8` (lr 1e-5, BatchNorm gelées), early stopping sur `val_loss`.

### Modèle de détection : YOLOv8
- Comparaison de `yolov8n` et `yolov8s` (pré-entraînés COCO), 50 epochs, imgsz 512.
- Augmentation adaptée aux IRM : pas de variation de teinte ni de saturation (images en niveaux de gris), légère rotation, translation, zoom, miroir horizontal, mosaic.

### Environnement
- Python 3.11, TensorFlow 2.21 (Keras 3) avec `tensorflow[and-cuda]`, PyTorch 2.14, Ultralytics 8.4.183, MLflow 3.17, Airflow 3.3.2.
- GPU NVIDIA en mode Hybrid (`supergfxctl -m Hybrid`). En mode Vfio, le GPU est réservé aux VM et inutilisable pour l'entraînement.
- `src/utils/gpu.py` : précharge `libcusolver` (bug de packaging TF 2.21) et active l'allocation mémoire progressive du GPU.
- `setup_gpu()` charge aussi Triton (utilisé par PyTorch) **avant** TensorFlow : dans un même processus, charger Triton après TensorFlow provoque un conflit de symboles LLVM et un crash (`free(): invalid pointer`).
- L'API tourne sur **CPU** (`CUDA_VISIBLE_DEVICES=-1` dans `api/config.py`) : TensorFlow (CUDA 12) et PyTorch (CUDA 13) ne peuvent pas partager le GPU dans le même processus. Le temps d'inférence reste d'environ 0.2 s par image, et l'API peut tourner sur une machine sans GPU.
- Pour la même raison, le pipeline lance les parties TensorFlow et PyTorch d'une étape dans des processus séparés (`src/pipeline.py`).

## Limites et pistes d'amélioration
- Les IRM saines sont annotées avec une boîte *No Tumor* : le détecteur renvoie donc cette boîte plutôt qu'une liste vide pour un cerveau sain.
- Une partie des labels contient des boîtes d'une autre classe ; un nettoyage manuel améliorerait sans doute la détection.
- Le modèle n'a pas été validé cliniquement : c'est un outil d'aide, pas un diagnostic.
