terraform {
  backend "gcs" {
    bucket = "thermostat-tfstate"
    prefix = "env/prod"
  }
}