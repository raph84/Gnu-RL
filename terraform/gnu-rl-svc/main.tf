#
# Required environment variables : 
#   - GOOGLE_APPPLICATION_CREDENTIALS
#
#

locals {
  # Ids for multiple sets of EC2 instances, merged together
  project_id = "thermostat-292016"
}


terraform {
  required_providers {
    google = {
      # using beta for Cloud Build GitHub
      source = "hashicorp/google-beta"
      version = "3.46.0"
    }
  }
}

provider "google" {
  region  = "us-east4"
  project = local.project_id
}

resource "google_service_account" "gnu-rl-agent" {
  account_id   = "gnu-rl-agent"
}

resource "google_storage_bucket_iam_binding" "gnu-rl-agent-bucket-ObjectCreator" {
  bucket = google_storage_bucket.gnu-rl-agent-bucket.name
  role = "roles/storage.objectCreator"
  members = [
    join(":", ["serviceAccount", google_service_account.gnu-rl-agent.email])
  ]
}

resource "google_storage_bucket_iam_binding" "gnu-rl-agent-bucket-ObjectViewer" {
  bucket = google_storage_bucket.gnu-rl-agent-bucket.name
  role = "roles/storage.objectViewer"
  members = [
    join(":", ["serviceAccount", google_service_account.gnu-rl-agent.email])
  ]
}

resource "google_project_iam_member" "cloud-debuger" {
  project = local.project_id
  role    = "roles/clouddebugger.agent"
  member  = join(":", ["serviceAccount", google_service_account.gnu-rl-agent.email])
}

resource "google_storage_bucket" "gnu-rl-agent-bucket" {
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

  substitutions = {
    # used in Cloud Build to set Cloud Run identity
    _SERVICEACCOUNT = "gnu-rl-agent@${local.project_id}.iam.gserviceaccount.com"
  }

  github {
      owner = "raph84"
      name = "Gnu-RL"

      push {
          branch = "gnu-rl-svc"
      }
  }

  filename = "cloudbuild.yaml"
}

resource "google_cloud_run_service" "default" {
  location                   = "us-east4"
  name                       = "gnu-rl-agent"
  project                    = "thermostat-292016"
  template {
        
        spec {
            container_concurrency = 1
            service_account_name  = "gnu-rl-agent@thermostat-292016.iam.gserviceaccount.com"
            timeout_seconds       = 300

            containers {
                image   = "us.gcr.io/thermostat-292016/gnu-rl-agent:latest"

                env {
                    name  = "PROJECT_ID"
                    value = "thermostat-292016"
                }
                env {
                    name  = "AGENT_BUCKET"
                    value = "gnu-rl-agent"
                }

                env {
                    name  = "SAVE_AGENT"
                    #value = "--save_agent"
                    value = " "
                }
                ports {
                    container_port = 8080
                }

                resources {
                    limits   = {
                        "cpu"    = "1000m"
                        "memory" = "256Mi"
                    }
                    requests = {}
                }
            }
        }
    }
}

resource "google_cloud_run_service_iam_member" "member" {
  project = local.project_id
  service = google_cloud_run_service.default.name
  role = "roles/run.invoker"
  location = "us-east4"
  member = join(":", ["serviceAccount", "thermostat-agent@thermostat-292016.iam.gserviceaccount.com"])
  
}