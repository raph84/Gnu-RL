terraform {
  required_providers {
    google = {
      source = "hashicorp/google-beta"
      version = "3.46.0"
    }
  }
}

provider "google" {
  region  = "us-east4"
}

resource "google_service_account" "gnu-rl-agent" {
  account_id   = "gnu-rl-agent"
}

resource "google_storage_bucket_iam_binding" "gnu-rl-agent-bucket-ObjectCreator" {
  bucket = google_storage_bucket.gnu-rl-agent.name
  role = "roles/storage.objectCreator"
  members = [
    join(":", ["serviceAccount", google_service_account.gnu-rl-agent.email])
  ]
}

resource "google_storage_bucket_iam_binding" "gnu-rl-agent-bucket-ObjectViewer" {
  bucket = google_storage_bucket.gnu-rl-agent.name
  role = "roles/storage.objectViewer"
  members = [
    join(":", ["serviceAccount", google_service_account.gnu-rl-agent.email])
  ]
}

resource "google_storage_bucket" "gnu-rl-agent" {
  name          = "gnu-rl-agent"
  location      = "US-EAST4"

  uniform_bucket_level_access = true

  storage_class = "REGIONAL"

  versioning {
      enabled = true
  }

  labels = {
      project = "thermostat"
  }

}

resource "google_cloudbuild_trigger" "build-trigger" {
  name = "gnu-rl-api-push-trigger"
  description = "Push to gnu-rl-svc"

  github {
      owner = "raph84"
      name = "Gnu-RL"

      push {
          branch = "gnu-rl-svc"
      }
  }

  filename = "cloudbuild.yaml"
}