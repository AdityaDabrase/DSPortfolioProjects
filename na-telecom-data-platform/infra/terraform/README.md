# Terraform — GCP infrastructure (optional)

Provision GCS landing bucket and BigQuery dataset for cloud mode.

## Prerequisites

- Terraform >= 1.5
- GCP project with billing enabled
- `gcloud auth application-default login`

## Usage

```bash
cd infra/terraform
terraform init
terraform plan -var="project_id=YOUR_PROJECT" -var="bucket_name=YOUR_UNIQUE_BUCKET"
terraform apply -var="project_id=YOUR_PROJECT" -var="bucket_name=YOUR_UNIQUE_BUCKET"
```

## Resources created

- `google_storage_bucket.na_telecom_raw` — landing zone for CRTC/FCC/subscription parquet
- `google_bigquery_dataset.na_telecom` — warehouse dataset for staging and marts

## Environment after apply

```bash
export GCP_PROJECT=YOUR_PROJECT
export GCS_BUCKET=YOUR_UNIQUE_BUCKET
export BQ_DATASET=na_telecom
export USE_GCS=1
export USE_BIGQUERY=1
```

Copy `main.tf.example` to `main.tf` before running (example provided to avoid requiring credentials in CI).
