# Guide de démo

## Avant la démo

```bash
airflow/start_airflow.sh                                  # terminal 1 : Airflow (http://localhost:8080)
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # terminal 2 : MLflow (http://localhost:5000)
.venv/bin/uvicorn api.main:app                            # terminal 3 : API (http://127.0.0.1:8000/docs)
.venv/bin/streamlit run app/streamlit_app.py              # terminal 4 : interface (http://localhost:8501)
```

- Vérifier `/health` : `{"status": "ok", "models_loaded": {"classifier": true, "detector": true}}`.
- Préparer quelques images : `tests/data/glioma.jpg`, `tests/data/no_tumor.jpg`, une image de `processed/classification/test/`, un fichier `.txt` pour montrer l'erreur 415.
- Lancer `.venv/bin/pytest` une fois (14 tests, environ 10 s).

## Déroulé (10 min)

1. **Le problème** (1 min) : aider au diagnostic d'IRM cérébrales, deux besoins complémentaires : *quel type de tumeur* (classification) et *où* (détection).
2. **Les données** (1 min) : 5 249 images, 4 classes déséquilibrées, labels YOLO. Split stratifié 70/15/15 commun aux deux modèles, train équilibré par augmentation, 16 images écartées pour YOLO (aucune boîte de leur classe).
3. **Classification** (2 min) : InceptionV3, feature extraction puis fine-tuning à partir de `mixed8`. Montrer la comparaison des 3 runs dans MLflow et la matrice de confusion. F1 test 0.961.
4. **Détection** (2 min) : YOLOv8n contre YOLOv8s, YOLOv8s retenu (mAP@0.5:0.95 val 0.775). Montrer `results.png`, la matrice de confusion et l'analyse des erreurs (Glioma, Pituitary).
5. **MLflow** (1 min) : runs, paramètres, artefacts, Model Registry avec l'alias `champion`.
6. **Airflow** (1 min) : graphe du DAG, statuts des tâches, logs d'une tâche, paramètres `quick` / `force_prepare` / `force_save`.
7. **Application** (2 min) : Streamlit → classification puis détection d'une IRM, seuil de confiance, erreur avec un fichier `.txt`, API arrêtée (message « API injoignable »).

## Code review : points à savoir expliquer

| Question probable | Réponse |
|---|---|
| Pourquoi InceptionV3 et pas GoogLeNet ? | Validé par le formateur. Même famille, disponible dans Keras avec des blocs nommés, donc dégel partiel possible. InceptionV1 n'existe que sur TF Hub en un seul bloc. |
| Pourquoi un learning rate de 1e-5 en fine-tuning ? | Pour ajuster les features pré-entraînées sans les détruire. Les BatchNorm restent gelées pour garder leurs statistiques. |
| Pourquoi choisir sur la validation ? | Le test ne sert qu'à mesurer la généralisation ; choisir dessus le rendrait optimiste. |
| Pourquoi le même split pour les deux modèles ? | Pour qu'une image de test de l'un ne soit pas dans le train de l'autre. |
| Pourquoi l'API tourne sur CPU ? | TensorFlow (CUDA 12) et PyTorch (CUDA 13) ne peuvent pas partager le GPU dans un même processus. 0.2 s par image suffit. |
| Comment les modèles sont-ils chargés ? | Une seule fois au démarrage (`lifespan`), stockés dans `app.state.models`. |
| Pourquoi Airflow dans un environnement séparé ? | Ses dépendances entrent en conflit avec TensorFlow et PyTorch. Le DAG appelle `src.pipeline` avec le Python du projet. |
| Que se passe-t-il si un nouvel entraînement est moins bon ? | `save_models_and_artifacts` ne promeut un modèle que s'il bat le champion sur le test. |
| Comment le pipeline réagit à des données invalides ? | `validate_data` lève une exception : la tâche passe en rouge et les tâches suivantes ne tournent pas. |

## Mise en situation : modifications probables

| Demande | Où modifier | Vérifier |
|---|---|---|
| Changer le seuil de confiance par défaut | `api/config.py` (`DETECTION_CONF`) ou variable d'environnement | `/docs`, Streamlit |
| Ajouter un endpoint (ex. `/models` avec les run_id) | `api/main.py` + un schéma dans `api/schemas.py` | `/docs`, un test dans `tests/test_api.py` |
| Ajouter une métrique de classification | `src/classification/metrics.py` (`compute_metrics`) | `python -m src.pipeline train_classifier --quick` |
| Ajouter une augmentation | `config/data.yaml` (`augmentations`) + fonction dans `src/data/prepare.py` (`augment_train`) | `python -m src.pipeline check_and_prepare_datasets --force` |
| Changer un hyperparamètre | `config/classification.yaml` ou `config/detection.yaml` | DAG avec `quick` |
| Ajouter une tâche au DAG | fonction dans `src/`, entrée dans `STEPS` de `src/pipeline.py` et de `airflow/dags/brain_tumor_pipeline.py` | graphe dans l'UI Airflow |

Après chaque modification : `.venv/bin/pytest`.

## Provoquer un échec pour la démo

Renommer `data/` puis lancer le DAG avec `force_prepare` : `check_and_prepare_datasets` échoue (« Données brutes introuvables »), le retry est visible, la tâche passe en rouge et les suivantes ne tournent pas. Remettre le dossier ensuite.
