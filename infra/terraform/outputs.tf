output "network" {
  value = docker_network.tracking.name
}

output "postgres_url" {
  value     = "postgresql://${var.postgres_user}:${var.postgres_password}@127.0.0.1:${var.postgres_port}/mlflow"
  sensitive = true
}

output "minio_endpoint" {
  value = "http://127.0.0.1:${var.minio_api_port}"
}

output "minio_console" {
  value = "http://127.0.0.1:${var.minio_console_port}"
}
