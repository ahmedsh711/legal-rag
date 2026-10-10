# Tracking infrastructure (Terraform)

Postgres (MLflow metadata) and MinIO (DVC remote, MLflow artifacts) as Docker containers, managed with the [kreuzwerker/docker](https://registry.terraform.io/providers/kreuzwerker/docker/latest) provider. It is the infrastructure-as-code version of the `tracking` services in `docker/compose.yaml` and uses different host ports (15432, 19000, 19001), so both can run at the same time.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set the passwords
terraform init
terraform plan
terraform apply
terraform output minio_endpoint
terraform destroy
```

If the Terraform binary cannot run on the host, the official image works the same way:

```bash
docker run --rm -v "$PWD:/work" -w /work -v /var/run/docker.sock:/var/run/docker.sock \
  -e TF_VAR_postgres_password -e TF_VAR_minio_root_password hashicorp/terraform:1.16.5 apply
```

| Resource | Purpose |
|---|---|
| `docker_network.tracking` | private network; containers reach each other as `postgres` and `minio` |
| `docker_volume.postgres_data`, `docker_volume.minio_data` | data that survives container replacement |
| `docker_container.postgres`, `docker_container.minio` | the services, with health checks, bound to 127.0.0.1 only |

Passwords are input variables marked `sensitive`; the Postgres URL output is sensitive too (`terraform output -raw postgres_url`).

## State

State is kept locally in `terraform.tfstate`, which is git-ignored because it contains the passwords in plain text. A shared setup would use a remote backend with locking, for example an S3 bucket with DynamoDB locking or Terraform Cloud, so that two people cannot apply at the same time and the state is not lost with a laptop.
