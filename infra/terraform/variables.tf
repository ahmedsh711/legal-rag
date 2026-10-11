variable "docker_host" {
  description = "Docker daemon address, e.g. npipe:////./pipe/docker_engine on Windows"
  type        = string
  default     = null
}

variable "name_prefix" {
  description = "Prefix for every container, network and volume"
  type        = string
  default     = "legal-rag-tf"
}

variable "postgres_image" {
  type    = string
  default = "postgres:17-alpine"
}

variable "minio_image" {
  type    = string
  default = "bitnamilegacy/minio:2025.7.23"
}

variable "postgres_user" {
  type    = string
  default = "legalrag"
}

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "minio_root_user" {
  type    = string
  default = "minio"
}

variable "minio_root_password" {
  type      = string
  sensitive = true

  validation {
    condition     = length(var.minio_root_password) >= 8
    error_message = "MinIO needs a root password of at least 8 characters."
  }
}

variable "minio_buckets" {
  type    = list(string)
  default = ["dvc", "mlflow", "langfuse"]
}

# host ports differ from docker/compose.yaml so both stacks can run side by side
variable "postgres_port" {
  type    = number
  default = 15432
}

variable "minio_api_port" {
  type    = number
  default = 19000
}

variable "minio_console_port" {
  type    = number
  default = 19001
}
