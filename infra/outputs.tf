output "cloud_run_service_uri" {
  description = "Public HTTPS endpoint of the deployed Endurance Copilot ADK service"
  value       = google_cloud_run_v2_service.endurance_copilot_service.uri
}

output "agent_service_account_email" {
  description = "Least-privilege IAM Service Account bound to the ADK Agent"
  value       = google_service_account.endurance_agent_sa.email
}

output "memory_bank_bucket_name" {
  description = "Cloud Storage bucket persisting compacted ADK sessions and telemetry"
  value       = google_storage_bucket.agent_memory_bank.name
}
