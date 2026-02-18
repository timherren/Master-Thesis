# Quick start script for TRAM-DAG Application
# Run this file in R or RStudio

cat("\n")
cat("========================================\n")
cat("  TRAM-DAG Continuous 3 Variables\n")
cat("  Linear DGP (LinearShift) Application\n")
cat("========================================\n")
cat("\n")

# Check and load required packages
required_packages <- c("shiny", "shinyFiles", "visNetwork", "reticulate", 
                       "igraph", "dplyr", "DT", "jsonlite", "httr")
missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  cat("Installing missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing)
}

# Load libraries
library(shiny)
library(shinyFiles)
library(visNetwork)
library(reticulate)
library(igraph)
library(dplyr)
library(DT)
library(jsonlite)
library(httr)

cat("✓ All R packages loaded\n")

# Configure Python
cat("Configuring Python environment...\n")
tryCatch({
  use_condaenv("tramdag", required = TRUE)
  cat("✓ Python environment configured\n")
}, error = function(e) {
  cat("⚠ Warning: Could not set conda environment. Using default Python.\n")
  cat("  Error:", conditionMessage(e), "\n")
  cat("  You may need to adjust the conda environment name in app.R\n")
})

# Test Python connection
cat("Testing Python connection...\n")
tryCatch({
  tramdag <- import("tramdag")
  cat("✓ Python tramdag module loaded successfully\n")
}, error = function(e) {
  cat("❌ Error loading Python tramdag module:\n")
  cat("  ", conditionMessage(e), "\n")
  cat("\nPlease ensure:\n")
  cat("  1. Python environment is correctly configured\n")
  cat("  2. tramdag package is installed: pip install tramdag\n")
  stop("Cannot proceed without Python tramdag module")
})

cat("\n")
cat("========================================\n")
cat("  Starting Shiny Application...\n")
cat("========================================\n")
cat("\n")
cat("The app will open in your browser automatically.\n")
cat("If not, navigate to: http://127.0.0.1:3838\n")
cat("\nPress Ctrl+C (or Esc in RStudio) to stop the app.\n")
cat("\n")
cat("NOTE: Experiment folders will be automatically created in your Downloads folder.\n")
cat("      All results and models will be saved there.\n")
cat("\n")

# Run the app
app_dir <- dirname(normalizePath("app.R", mustWork = FALSE))
if (file.exists(file.path(app_dir, "app.R"))) {
  original_wd <- getwd()
  setwd(app_dir)
  cat("Changed working directory to app directory:", app_dir, "\n")
  on.exit(setwd(original_wd), add = TRUE)
}

runApp("app.R", host = "127.0.0.1", port = 3838, launch.browser = TRUE)
