terraform {
  backend "gcs" {
    bucket = "thermostat-292016-tfstate"
    prefix = "env/prod"
  }
}