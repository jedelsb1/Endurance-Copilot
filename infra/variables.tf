variable "project_id" {
  description = "Google Cloud Project ID hosting the Endurance Copilot Agent"
  type        = string
  default     = "endurance-copilot-prod"
}

variable "region" {
  description = "Google Cloud Region for Vertex AI Reasoning Engine and Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "endurance-copilot-agent"
}

variable "container_image" {
  description = "Artifact Registry container URI for Endurance Copilot"
  type        = string
  default     = "us-central1-docker.pkg.dev/endurance-copilot-prod/agent-repo/endurance-copilot:latest"
}
