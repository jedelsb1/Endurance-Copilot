# =============================================================================
# Endurance Copilot — Google Cloud Infrastructure as Code (Terraform)
# Infrastructure as Code
# Provisions Vertex AI Reasoning Engine / Agent Builder resources, Cloud Run,
# Secret Manager secrets, Cloud Storage Memory Bank, and Cloud Logging Sinks.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs for ADK & Vertex AI Agent Engine
resource "google_project_service" "agentops_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "secretmanager.googleapis.com",
    "run.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Dedicated Least-Privilege Service Account for the Endurance Copilot Agent
resource "google_service_account" "endurance_agent_sa" {
  account_id   = "endurance-copilot-agent-sa"
  display_name = "Endurance Copilot ADK Multi-Agent Service Account"
  description  = "Executes Vertex AI Reasoning Engine, Secret Manager access, and DLP PII redaction"
}

resource "google_project_iam_member" "agent_sa_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/secretmanager.secretAccessor",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.endurance_agent_sa.email}"
}

# 3. Google Cloud Secret Manager Resources (No hardcoded API keys)
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.agentops_apis]
}

resource "google_secret_manager_secret" "oura_personal_access_token" {
  secret_id = "oura-personal-access-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.agentops_apis]
}

resource "google_secret_manager_secret" "garmin_oauth_token" {
  secret_id = "garmin-oauth-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.agentops_apis]
}

# 4. Persistent Memory Bank & Telemetry Archive Bucket
resource "google_storage_bucket" "agent_memory_bank" {
  name                        = "${var.project_id}-endurance-memory-bank"
  location                    = var.region
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
}

# 5. Cloud Run v2 Service hosting the FastAPI & ADK Agent Orchestrator
resource "google_cloud_run_v2_service" "endurance_copilot_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.endurance_agent_sa.email

    containers {
      image = var.container_image

      ports {
        container_port = 7860
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "ENABLE_GCP_DLP_API"
        value = "1"
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "OURA_PERSONAL_ACCESS_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.oura_personal_access_token.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.agentops_apis,
    google_secret_manager_secret.gemini_api_key,
  ]
}
