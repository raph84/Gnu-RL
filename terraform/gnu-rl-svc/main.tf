#
# Required environment variables : 
#   - GOOGLE_APPPLICATION_CREDENTIALS
#
#

locals {
  # Ids for multiple sets of EC2 instances, merged together
  project_id = "thermostat-292016"
  api_name = "gnu-rl-agent"
}


terraform {
  required_providers {
    google = {
      # using beta for Cloud Build GitHub
      source = "hashicorp/google-beta"
      version = "3.46.0"
    }
    docker = {
      source = "terraform-providers/docker"
      version = "~> 2.7.2"
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

data "google_client_config" "default" {}

provider "docker" {
  registry_auth {
    address  = "gcr.io"
    username = "oauth2accesstoken"
    password = data.google_client_config.default.access_token
  }
  #host = "npipe:////.//pipe//docker_engine"
}

data "docker_registry_image" "gnu-rl-api-image" {
  name = "gcr.io/${local.project_id}/${local.api_name}"
}

data "google_container_registry_image" "gnu-rl-api-image-latest" {
  name    = local.api_name
  project = local.project_id
  digest  = data.docker_registry_image.gnu-rl-api-image.sha256_digest
}

output "image_url" {
  value = data.google_container_registry_image.gnu-rl-api-image-latest.image_url
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
                image   = data.google_container_registry_image.gnu-rl-api-image-latest.image_url

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
                    value = "--save_agent"
                    #value = " "
                }
                ports {
                    container_port = 8080
                }

                resources {
                    limits   = {
                        "cpu"    = "1000m"
                        "memory" = "512Mi"
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