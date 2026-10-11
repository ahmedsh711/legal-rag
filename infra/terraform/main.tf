# Backing services for experiment tracking: Postgres (MLflow metadata) and MinIO (DVC remote,
# MLflow artifacts) on their own network, with named volumes that survive container replacement.

resource "docker_network" "tracking" {
  name = "${var.name_prefix}-net"
}

resource "docker_volume" "postgres_data" {
  name = "${var.name_prefix}-postgres-data"
}

resource "docker_volume" "minio_data" {
  name = "${var.name_prefix}-minio-data"
}

resource "docker_image" "postgres" {
  name         = var.postgres_image
  keep_locally = true
}

resource "docker_image" "minio" {
  name         = var.minio_image
  keep_locally = true
}

resource "docker_container" "postgres" {
  name    = "${var.name_prefix}-postgres"
  image   = docker_image.postgres.image_id
  restart = "unless-stopped"

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=mlflow",
  ]

  ports {
    internal = 5432
    external = var.postgres_port
    ip       = "127.0.0.1"
  }

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }

  networks_advanced {
    name    = docker_network.tracking.id
    aliases = ["postgres"]
  }

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U ${var.postgres_user}"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
}

resource "docker_container" "minio" {
  name    = "${var.name_prefix}-minio"
  image   = docker_image.minio.image_id
  restart = "unless-stopped"

  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}",
    "MINIO_DEFAULT_BUCKETS=${join(",", var.minio_buckets)}",
  ]

  ports {
    internal = 9000
    external = var.minio_api_port
    ip       = "127.0.0.1"
  }

  ports {
    internal = 9001
    external = var.minio_console_port
    ip       = "127.0.0.1"
  }

  volumes {
    volume_name    = docker_volume.minio_data.name
    container_path = "/bitnami/minio/data"
  }

  networks_advanced {
    name    = docker_network.tracking.id
    aliases = ["minio"]
  }

  healthcheck {
    test     = ["CMD", "curl", "-fs", "http://localhost:9000/minio/health/live"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
}
