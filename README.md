# Workflow CI - Mursyid Dwi Wahidiyantoro

Repository ini memuat konfigurasi MLflow Project dan workflow CI untuk continuous integration dan automated retraining model machine learning Heart Disease.

## Struktur Direktori
```
Workflow-CI_Mursyid-Dwi-Wahidiyantoro/
├── .github/workflows/
│   └── ci.yml
└── MLProject/
    ├── MLproject
    ├── conda.yaml
    ├── modelling.py
    └── heart_preprocessing.csv
```

## Spesifikasi MLflow Project
- **Entry point**: `main`
- **Parameter**:
  - `n_estimators`: Jumlah decision trees (default: 100)
  - `max_depth`: Kedalaman pohon maksimal (default: 5)
- **Perintah Eksekusi**:
  ```bash
  mlflow run MLProject --env-manager local -P n_estimators=100 -P max_depth=5
  ```

## CI Trigger
Workflow CI akan memicu re-training model secara otomatis setiap kali ada perubahan pada branch `main` atau melalui trigger manual `workflow_dispatch`.
