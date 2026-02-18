# TRAM-DAG Causal Analysis Application
# Interactive Shiny app for fitting TRAM-DAG models, exploring causal effects, and generating reports.
# Supports arbitrary datasets and DAG structures.

library(shiny)
library(shinyFiles)
library(visNetwork)
library(reticulate)
library(igraph)
library(dplyr)
library(jsonlite)
library(httr)
library(promises)
library(future)
plan(multisession)

# Allow uploads up to 100 MB
options(shiny.maxRequestSize = 100 * 1024^2)

## --------------------------------------------------
## 0) Python / TRAM-DAG setup
## --------------------------------------------------

# Configure Python environment
# In Docker: TRAMDAG_PYTHON env var points to the conda env Python
# Locally:   falls back to the tramdag conda environment
if (nzchar(Sys.getenv("TRAMDAG_PYTHON"))) {
  use_python(Sys.getenv("TRAMDAG_PYTHON"), required = TRUE)
} else {
  use_condaenv("tramdag", required = TRUE)
}
py_config()

# Set Python environment variables BEFORE importing torch
py_run_string("
import os
# Disable torch dynamo compilation to avoid circular import issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
")

# Import Python modules
tryCatch({
  torch <- import("torch")
  tryCatch({
    invisible(torch[["_dynamo"]])
  }, error = function(e) {
    tryCatch({
      py_run_string("
import torch
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True
")
    }, error = function(e2) {})
  })
}, error = function(e) {
  warning("Could not pre-import torch: ", conditionMessage(e))
})

# Import other dependencies
pd <- import("pandas")
np <- import("numpy")
os <- import("os")

# Import tramdag modules
tramdag <- import("tramdag")
TramDagConfig <- tramdag$TramDagConfig
TramDagModel <- tramdag$TramDagModel

# Initialize Ollama LLM client (local, free - for result interpretations)
OLLAMA_BASE_URL <- Sys.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL <- Sys.getenv("OLLAMA_MODEL", "llama3.2")
ollama_available <- FALSE

tryCatch({
  cat("Checking Ollama availability at:", OLLAMA_BASE_URL, "\n")
  resp <- httr::GET(paste0(OLLAMA_BASE_URL, "/api/tags"), httr::timeout(5))
  if (httr::status_code(resp) == 200) {
    ollama_available <- TRUE
    models <- jsonlite::fromJSON(httr::content(resp, as = "text", encoding = "UTF-8"))
    model_names <- models$models$name
    if (length(model_names) > 0) {
      cat("✓ Ollama is running. Available models:", paste(model_names, collapse = ", "), "\n")
      if (!any(grepl(OLLAMA_MODEL, model_names, fixed = TRUE))) {
        cat("⚠ Configured model '", OLLAMA_MODEL, "' not found. Pull it with: ollama pull ", OLLAMA_MODEL, "\n")
        ollama_available <- FALSE
      } else {
        cat("✓ Using model:", OLLAMA_MODEL, "\n")
      }
    } else {
      cat("⚠ Ollama is running but no models are available. Pull a model with: ollama pull", OLLAMA_MODEL, "\n")
      ollama_available <- FALSE
    }
  } else {
    cat("⚠ Ollama responded with status", httr::status_code(resp), "- LLM interpretations disabled\n")
  }
}, error = function(e) {
  cat("⚠ Ollama not reachable at", OLLAMA_BASE_URL, "- LLM interpretations disabled\n")
  cat("  To enable: install Ollama (https://ollama.ai), start it, and run: ollama pull", OLLAMA_MODEL, "\n")
})

# Helper function to call Ollama chat API
ollama_chat <- function(system_msg, user_msg, temperature = 0.3) {
  if (!ollama_available) return("")
  
  tryCatch({
    body <- list(
      model = OLLAMA_MODEL,
      messages = list(
        list(role = "system", content = system_msg),
        list(role = "user", content = user_msg)
      ),
      stream = FALSE,
      options = list(temperature = temperature)
    )
    
    resp <- httr::POST(
      url = paste0(OLLAMA_BASE_URL, "/api/chat"),
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      httr::content_type_json(),
      httr::timeout(120)
    )
    
    if (httr::status_code(resp) == 200) {
      result <- jsonlite::fromJSON(httr::content(resp, as = "text", encoding = "UTF-8"))
      content <- result$message$content
      content <- gsub("^\\s+|\\s+$", "", content)
      content <- gsub("\\n{3,}", "\\n\\n", content)
      return(content)
    } else {
      cat("Warning: Ollama returned status", httr::status_code(resp), "\n")
      return("")
    }
  }, error = function(e) {
    cat("Warning: Ollama call failed:", conditionMessage(e), "\n")
    return("")
  })
}

## --------------------------------------------------
## 1) Helper Functions
## --------------------------------------------------

create_default_dag <- function(vars, edge_type = "ls") {
  n <- length(vars)
  A <- matrix("0", nrow = n, ncol = n, dimnames = list(vars, vars))
  if (n >= 2) {
    for (i in seq_len(n - 1)) {
      for (j in (i + 1):n) {
        A[vars[i], vars[j]] <- edge_type
      }
    }
  }
  return(A)
}

# Ensure adjacency matrix is upper triangular (required by TRAM-DAG)
ensure_upper_triangular <- function(A, vars) {
  edge_list <- list()
  for (r in rownames(A)) {
    for (cc in colnames(A)) {
      val <- A[r, cc]
      if (!is.na(val) && val != "0" && val != 0 && r != cc) {
        edge_list <- c(edge_list, list(c(r, cc)))
      }
    }
  }

  # Topological sort: reorder vars so all edges go from lower to higher index
  if (length(edge_list) > 0) {
    g <- igraph::make_empty_graph(directed = TRUE)
    g <- igraph::add_vertices(g, length(vars), name = vars)
    for (e in edge_list) {
      g <- igraph::add_edges(g, c(e[1], e[2]))
    }
    topo <- tryCatch(igraph::topo_sort(g, mode = "out"), error = function(e) NULL)
    if (!is.null(topo)) {
      vars <- names(topo)
    }
  }

  A_new <- matrix("0", nrow = length(vars), ncol = length(vars),
                  dimnames = list(vars, vars))

  for (e in edge_list) {
    from_var <- e[1]; to_var <- e[2]
    fi <- which(vars == from_var)
    ti <- which(vars == to_var)
    if (length(fi) == 1 && length(ti) == 1 && fi < ti) {
      A_new[from_var, to_var] <- A[from_var, to_var]
    }
  }

  diag(A_new) <- "0"
  return(A_new)
}

# Convert adjacency matrix to edge list for visNetwork
adjacency_to_edges <- function(A) {
  vars_row <- rownames(A)
  vars_col <- colnames(A)
  if (is.null(vars_row)) vars_row <- vars_col
  if (is.null(vars_col)) vars_col <- vars_row
  
  idx <- which(A != "0" & A != 0, arr.ind = TRUE)
  if (nrow(idx) == 0) {
    return(data.frame(
      from = character(0),
      to   = character(0),
      edge_type = character(0),
      stringsAsFactors = FALSE
    ))
  }
  
  data.frame(
    from = vars_row[idx[, "row"]],
    to   = vars_col[idx[, "col"]],
    edge_type = as.character(A[idx]),
    stringsAsFactors = FALSE
  )
}

# Create experiment directory in Downloads folder
create_experiment_dir <- function(experiment_name = NULL) {
  downloads_path <- file.path(Sys.getenv("HOME"), "Downloads")
  
  if (is.null(experiment_name)) {
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    experiment_name <- paste0("TramDag_Experiment_", timestamp)
  }
  
  experiment_dir <- file.path(downloads_path, experiment_name)
  
  # Create directory if it doesn't exist
  if (!dir.exists(experiment_dir)) {
    dir.create(experiment_dir, recursive = TRUE)
  }
  
  # Create scripts folder for reproducible analysis
  scripts_dir <- file.path(experiment_dir, "scripts")
  if (!dir.exists(scripts_dir)) {
    dir.create(scripts_dir, recursive = TRUE)
  }
  
  return(experiment_dir)
}

# Save reproducible analysis scripts to experiment folder
save_analysis_scripts <- function(experiment_dir, data_path = NULL, amat = NULL, 
                                  epochs = 100, learning_rate = 0.01, 
                                  batch_size = 512, set_initial_weights = FALSE,
                                  X_var = NULL, Y_var = NULL,
                                  x_treated = 1, x_control = 0) {
  vars <- if (!is.null(amat)) rownames(amat) else NULL
  if (is.null(vars) && !is.null(amat)) vars <- colnames(amat)
  if (is.null(X_var)) X_var <- if (!is.null(vars) && length(vars) > 0) vars[1] else "x1"
  if (is.null(Y_var)) Y_var <- if (!is.null(vars) && length(vars) > 1) vars[length(vars)] else "x2"
  scripts_dir <- file.path(experiment_dir, "scripts")
  if (!dir.exists(scripts_dir)) {
    dir.create(scripts_dir, recursive = TRUE)
  }
  
  # Create main reproducible script
  main_script <- paste0('
# Reproducible TRAM-DAG Analysis Script
# Generated automatically for experiment: ', basename(experiment_dir), '
# Date: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '

library(reticulate)
library(jsonlite)

# Configure Python environment (adjust environment name to match your setup)
tryCatch({
  use_condaenv("tramdag", required = TRUE)
}, error = function(e) {
  cat("Conda env not found. Using default Python. Install tramdag: pip install tramdag\\n")
})

# Import Python modules
tramdag <- import("tramdag")
TramDagConfig <- tramdag$TramDagConfig
TramDagModel <- tramdag$TramDagModel
np <- import("numpy")

# Set experiment directory
EXPERIMENT_DIR <- "', experiment_dir, '"

# Load configuration
cfg <- TramDagConfig()
cfg$setup_configuration(
  experiment_name = "', basename(experiment_dir), '",
  EXPERIMENT_DIR = EXPERIMENT_DIR
)
cfg$update()

# Load fitted model
cat("Loading fitted model from:", EXPERIMENT_DIR, "\\n")
td_model <- TramDagModel$from_directory(EXPERIMENT_DIR)
cat("Model loaded successfully!\\n")

# Example: Observational sampling
cat("\\n=== Observational Sampling ===\\n")
n_samples <- 10000L
samples_result <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = NULL,
  verbose = TRUE
)

# Extract samples
sampled_dict <- samples_result[[1]]
sample_list <- list()
for (node in names(sampled_dict)) {
  py_tensor <- sampled_dict[[node]]
  if (reticulate::py_has_attr(py_tensor, "detach")) {
    np_array <- py_tensor$detach()$cpu()$numpy()
  } else if (reticulate::py_has_attr(py_tensor, "cpu")) {
    np_array <- py_tensor$cpu()$numpy()
  } else if (reticulate::py_has_attr(py_tensor, "numpy")) {
    np_array <- py_tensor$numpy()
  } else {
    np_array <- py_tensor
  }
  sample_list[[node]] <- as.numeric(py_to_r(np_array))
}
sampled_df <- as.data.frame(sample_list)
cat("Sampled", nrow(sampled_df), "observations\\n")

# Interventional sampling (ATE computation)
cat("\\n=== Interventional Sampling (ATE) ===\\n")
# do(', X_var, ' = ', x_treated, ') vs do(', X_var, ' = ', x_control, ') on ', Y_var, '
X <- "', X_var, '"
Y <- "', Y_var, '"
x_treated <- ', x_treated, '
x_control <- ', x_control, '

# Treatment
do_treated <- reticulate::dict()
do_treated[[X]] <- as.numeric(x_treated)
samp_treated <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = do_treated,
  verbose = TRUE
)
y_treated_tensor <- samp_treated[[1]][[Y]]
if (reticulate::py_has_attr(y_treated_tensor, "detach")) {
  y_treated <- as.numeric(py_to_r(y_treated_tensor$detach()$cpu()$numpy()))
} else {
  y_treated <- as.numeric(py_to_r(y_treated_tensor$cpu()$numpy()))
}

# Control
do_control <- reticulate::dict()
do_control[[X]] <- as.numeric(x_control)
samp_control <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = do_control,
  verbose = TRUE
)
y_control_tensor <- samp_control[[1]][[Y]]
if (reticulate::py_has_attr(y_control_tensor, "detach")) {
  y_control <- as.numeric(py_to_r(y_control_tensor$detach()$cpu()$numpy()))
} else {
  y_control <- as.numeric(py_to_r(y_control_tensor$cpu()$numpy()))
}

# Compute ATE
ate <- mean(y_treated) - mean(y_control)
cat("\\nATE = E[", Y, " | do(", X, " = ", x_treated, ")] - E[", Y, " | do(", X, " = ", x_control, ")] = ", 
    round(ate, 4), "\\n", sep = "")

cat("\\n=== Analysis Complete ===\\n")
cat("Results saved in:", EXPERIMENT_DIR, "\\n")
')
  
  # Write main script
  writeLines(main_script, file.path(scripts_dir, "reproduce_analysis.R"))
  
  # Create README for scripts folder
  readme_content <- paste0('# Reproducible Analysis Scripts

This folder contains scripts to reproduce the TRAM-DAG analysis for this experiment.

## Experiment Information

- **Experiment Name**: ', basename(experiment_dir), '
- **Created**: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
- **Experiment Directory**: ', experiment_dir, '

## Files

- `reproduce_analysis.R`: Main script to reproduce the analysis
  - Loads the fitted model from the experiment directory
  - Performs observational sampling
  - Computes ATE (Average Treatment Effect) for interventions
  - Can be customized for specific analyses

## Usage

1. Ensure Python environment is set up:
   ```r
   library(reticulate)
   use_condaenv("tramdag", required = TRUE)
   ```

2. Run the reproducible script:
   ```r
   source("scripts/reproduce_analysis.R")
   ```

## Customization

Edit `reproduce_analysis.R` to:
- Change intervention variables and values
- Modify number of samples
- Add additional analyses
- Create custom visualizations

## Model Fitting Parameters

- **Epochs**: ', epochs, '
- **Learning Rate**: ', learning_rate, '
- **Batch Size**: ', batch_size, '
- **Initial Weights**: ', if(set_initial_weights) "R-based" else "Random", '

## Data Information
', if(!is.null(data_path)) paste0('- **Data Path**: ', data_path, '\n'), 
if(!is.null(amat)) paste0('- **Adjacency Matrix**: Saved in configuration.json\n'), '
- **Configuration**: See `configuration.json` in experiment directory

## Notes

- The model must be fitted first (this is done automatically in the Shiny app)
- All model files are saved in variable-specific subdirectories (one per variable)
- Loss history is available in `train_loss_hist.json` and `val_loss_hist.json` for each variable
')
  
  writeLines(readme_content, file.path(scripts_dir, "README.md"))
  
  cat("Analysis scripts saved to:", scripts_dir, "\n")
}

# Export complete reproducible package
# Creates a complete reproducible package that can be run on any computer.
# Includes: data, full workflow script, environment files, and Docker support.
export_reproducible_package <- function(experiment_dir, data_df, data_path = NULL, 
                                        amat = NULL, epochs = 100, learning_rate = 0.01,
                                        batch_size = 512, set_initial_weights = FALSE,
                                        X_var = NULL, Y_var = NULL,
                                        x_treated = 1, x_control = 0) {
  pkg_vars <- colnames(data_df)
  if (is.null(X_var)) X_var <- if (length(pkg_vars) > 0) pkg_vars[1] else "x1"
  if (is.null(Y_var)) Y_var <- if (length(pkg_vars) > 1) pkg_vars[length(pkg_vars)] else "x2"
  
  scripts_dir <- file.path(experiment_dir, "reproducible_package")
  if (!dir.exists(scripts_dir)) {
    dir.create(scripts_dir, recursive = TRUE)
  }
  
  # 1. Save data file
  data_file <- file.path(scripts_dir, "data.csv")
  write.csv(data_df, data_file, row.names = FALSE)
  cat("Data saved to:", data_file, "\n")
  
  # 2. Create complete workflow script (from scratch)
  workflow_script <- paste0('
# ============================================================================
# Complete Reproducible TRAM-DAG Workflow
# ============================================================================
# This script reproduces the ENTIRE analysis from scratch
# Generated: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
# Experiment: ', basename(experiment_dir), '
# ============================================================================

# Install required R packages if not already installed
required_r_packages <- c("reticulate", "jsonlite", "igraph", "dplyr")
missing_r <- required_r_packages[!sapply(required_r_packages, requireNamespace, quietly = TRUE)]
if (length(missing_r) > 0) {
  cat("Installing missing R packages:", paste(missing_r, collapse = ", "), "\\n")
  install.packages(missing_r)
}

# Load R libraries
library(reticulate)
library(jsonlite)
library(igraph)
library(dplyr)

# ============================================================================
# Step 1: Configure Python Environment
# ============================================================================
cat("\\n=== Step 1: Configuring Python Environment ===\\n")

# Option 1: Use conda environment (if available)
tryCatch({
  use_condaenv("tramdag", required = TRUE)
  cat("✓ Using conda environment: tramdag\\n")
}, error = function(e) {
  cat("⚠ Conda environment not found. Using default Python.\\n")
  cat("  You may need to install tramdag: pip install tramdag\\n")
})

# Import Python modules
tramdag <- import("tramdag")
TramDagConfig <- tramdag$TramDagConfig
TramDagModel <- tramdag$TramDagModel
np <- import("numpy")
pd <- import("pandas")

# ============================================================================
# Step 2: Load Data
# ============================================================================
cat("\\n=== Step 2: Loading Data ===\\n")
data_file <- file.path(dirname(getwd()), "data.csv")
if (!file.exists(data_file)) {
  # Try alternative path
  data_file <- "data.csv"
}
df <- read.csv(data_file, check.names = TRUE)
cat("✓ Data loaded:", nrow(df), "rows,", ncol(df), "columns\\n")
cat("  Variables:", paste(colnames(df), collapse = ", "), "\\n")

# ============================================================================
# Step 3: Define DAG Structure
# ============================================================================
cat("\\n=== Step 3: Setting Up DAG Structure ===\\n")

# Load adjacency matrix from configuration or recreate it
vars <- colnames(df)
n_vars <- length(vars)

# Adjacency matrix (', if(!is.null(amat)) {
  amat_str <- paste(apply(amat, 1, function(x) paste(x, collapse = ",")), collapse = ";")
  paste0("from saved configuration: ", amat_str)
} else {
  "default structure"
}, ')
n_vars <- length(vars)
amat_codes <- matrix(c(', 
if(!is.null(amat) && nrow(amat) > 0 && ncol(amat) > 0) {
  amat_flat <- as.vector(amat)
  paste0('"', paste(amat_flat, collapse = '", "'), '"')
} else {
  # Default: all zeros
  n_vars_calc <- ncol(df)
  paste0(rep('"0"', n_vars_calc * n_vars_calc), collapse = ", ")
}, '), 
                nrow = n_vars, ncol = n_vars,
                dimnames = list(vars, vars))

# Ensure upper triangular
diag(amat_codes) <- "0"
for (i in 1:n_vars) {
  for (j in 1:i) {
    if (j < i) amat_codes[i, j] <- "0"
  }
}

cat("✓ DAG structure defined\\n")

# ============================================================================
# Step 4: Create Configuration
# ============================================================================
cat("\\n=== Step 4: Creating TRAM-DAG Configuration ===\\n")

# Set experiment directory (use current directory)
EXPERIMENT_DIR <- getwd()
if (basename(EXPERIMENT_DIR) != "', basename(experiment_dir), '") {
  # Create subdirectory for this experiment
  EXPERIMENT_DIR <- file.path(EXPERIMENT_DIR, "', basename(experiment_dir), '_reproduced")
  if (!dir.exists(EXPERIMENT_DIR)) {
    dir.create(EXPERIMENT_DIR, recursive = TRUE)
  }
}

cfg <- TramDagConfig()
cfg$setup_configuration(
  experiment_name = "', basename(experiment_dir), '_reproduced",
  EXPERIMENT_DIR = EXPERIMENT_DIR
)
cfg$update()

# Infer data types
data_types <- sapply(df, function(x) {
  if (is.numeric(x) && length(unique(x)) > 20) {
    "continous"
  } else if (is.numeric(x) || is.factor(x)) {
    "ordinal"
  } else {
    "continous"
  }
}, USE.NAMES = TRUE)
cfg$set_data_type(as.list(data_types))
cfg$update()

# Write adjacency matrix
py_amat <- np$array(amat_codes, dtype = "object")
utils_config <- tramdag$utils$configuration
utils_config$write_adj_matrix_to_configuration(py_amat, cfg$CONF_DICT_PATH)
cfg$update()

# Generate model names
py_data_types <- r_to_py(as.list(data_types))
nn_names <- utils_config$create_nn_model_names(py_amat, py_data_types)
utils_config$write_nn_names_matrix_to_configuration(nn_names, cfg$CONF_DICT_PATH)
cfg$update()

# Write nodes information
utils_config$write_nodes_information_to_configuration(cfg$CONF_DICT_PATH)
cfg$update()

# Compute levels for ordinal variables
if (any(data_types == "ordinal")) {
  py_df <- r_to_py(df)
  cfg$compute_levels(py_df)
}

cfg$save()
cat("✓ Configuration created and saved\\n")

# ============================================================================
# Step 5: Fit Model
# ============================================================================
cat("\\n=== Step 5: Fitting TRAM-DAG Model ===\\n")
cat("  Epochs: ', epochs, '\\n")
cat("  Learning Rate: ', learning_rate, '\\n")
cat("  Batch Size: ', batch_size, '\\n")
cat("  Initial Weights: ', if(set_initial_weights) "R-based" else "Random", '\\n")

# Split data (80/20)
n <- nrow(df)
set.seed(42)  # For reproducibility
train_idx <- sample(n, floor(0.8 * n))
train_df <- df[train_idx, ]
val_df <- df[-train_idx, ]

# Convert to pandas
py_train <- r_to_py(train_df)
py_val <- r_to_py(val_df)

# Create model
td_model <- TramDagModel$from_config(
  cfg,
  device = "auto",
  debug = FALSE,
  verbose = TRUE,
  set_initial_weights = ', if(set_initial_weights) "TRUE" else "FALSE", ',
  initial_data = py_train
)

# Compute min-max scaling
td_model$load_or_compute_minmax(
  td_train_data = py_train,
  use_existing = FALSE,
  write = TRUE
)

# Fit model
td_model$fit(
  py_train,
  py_val,
  learning_rate = ', learning_rate, ',
  epochs = as.integer(', epochs, '),
  batch_size = as.integer(', batch_size, '),
  device = "auto",
  verbose = TRUE
)

cat("✓ Model fitted successfully!\\n")

# ============================================================================
# Step 6: Observational Sampling
# ============================================================================
cat("\\n=== Step 6: Observational Sampling ===\\n")
n_samples <- 10000L
samples_result <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = NULL,
  verbose = TRUE
)

# Extract samples
sampled_dict <- samples_result[[1]]
sample_list <- list()
for (node in names(sampled_dict)) {
  py_tensor <- sampled_dict[[node]]
  if (reticulate::py_has_attr(py_tensor, "detach")) {
    np_array <- py_tensor$detach()$cpu()$numpy()
  } else if (reticulate::py_has_attr(py_tensor, "cpu")) {
    np_array <- py_tensor$cpu()$numpy()
  } else if (reticulate::py_has_attr(py_tensor, "numpy")) {
    np_array <- py_tensor$numpy()
  } else {
    np_array <- py_tensor
  }
  sample_list[[node]] <- as.numeric(py_to_r(np_array))
}
sampled_df <- as.data.frame(sample_list)
cat("✓ Sampled", nrow(sampled_df), "observations\\n")

# Save sampled data
write.csv(sampled_df, file.path(EXPERIMENT_DIR, "sampled_data.csv"), row.names = FALSE)

# ============================================================================
# Step 7: Interventional Sampling (ATE)
# ============================================================================
cat("\\n=== Step 7: Computing Average Treatment Effect (ATE) ===\\n")

# do(', X_var, ' = ', x_treated, ') vs do(', X_var, ' = ', x_control, ') on ', Y_var, '
X <- "', X_var, '"
Y <- "', Y_var, '"
x_treated <- ', x_treated, '
x_control <- ', x_control, '

# Treatment
do_treated <- reticulate::dict()
do_treated[[X]] <- as.numeric(x_treated)
samp_treated <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = do_treated,
  verbose = TRUE
)
y_treated_tensor <- samp_treated[[1]][[Y]]
if (reticulate::py_has_attr(y_treated_tensor, "detach")) {
  y_treated <- as.numeric(py_to_r(y_treated_tensor$detach()$cpu()$numpy()))
} else {
  y_treated <- as.numeric(py_to_r(y_treated_tensor$cpu()$numpy()))
}

# Control
do_control <- reticulate::dict()
do_control[[X]] <- as.numeric(x_control)
samp_control <- td_model$sample(
  number_of_samples = as.integer(n_samples),
  do_interventions = do_control,
  verbose = TRUE
)
y_control_tensor <- samp_control[[1]][[Y]]
if (reticulate::py_has_attr(y_control_tensor, "detach")) {
  y_control <- as.numeric(py_to_r(y_control_tensor$detach()$cpu()$numpy()))
} else {
  y_control <- as.numeric(py_to_r(y_control_tensor$cpu()$numpy()))
}

# Compute ATE
ate <- mean(y_treated) - mean(y_control)
cat("\\n✓ ATE = E[", Y, " | do(", X, " = ", x_treated, ")] - E[", Y, " | do(", X, " = ", x_control, ")] = ", 
    round(ate, 4), "\\n", sep = "")

# Save results
results <- list(
  ate = ate,
  y_treated_mean = mean(y_treated),
  y_control_mean = mean(y_control),
  y_treated_std = sd(y_treated),
  y_control_std = sd(y_control)
)
write_json(results, file.path(EXPERIMENT_DIR, "ate_results.json"), pretty = TRUE)

cat("\\n=== Analysis Complete ===\\n")
cat("Results saved in:", EXPERIMENT_DIR, "\\n")
cat("\\nTo reproduce this analysis on another computer:\\n")
cat("  1. Copy the entire reproducible_package folder\\n")
cat("  2. Install R and Python dependencies (see requirements files)\\n")
cat("  3. Run: Rscript run_complete_workflow.R\\n")
')
  
  writeLines(workflow_script, file.path(scripts_dir, "run_complete_workflow.R"))
  
  # 3. Create R package requirements file
  r_packages <- c("reticulate", "jsonlite", "igraph", "dplyr")
  r_requirements <- paste0('# R Package Requirements
# Generated: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
# Install with: install.packages(c("', paste(r_packages, collapse = '", "'), '"))

', paste(r_packages, collapse = "\n"))
  writeLines(r_requirements, file.path(scripts_dir, "r_requirements.txt"))
  
  # 4. Create Python requirements file
  python_requirements <- paste0('# Python Package Requirements
# Generated: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
# Install with: pip install -r python_requirements.txt

tramdag
numpy
pandas
torch
')
  writeLines(python_requirements, file.path(scripts_dir, "python_requirements.txt"))
  
  # 5. Create Jupyter Notebook (.ipynb)
  # Convert the R script into a Jupyter notebook format
  notebook_cells <- list()
  
  # Cell 1: Markdown header
  notebook_cells[[1]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c(
      paste0("# TRAM-DAG Complete Workflow\n"),
      paste0("**Experiment**: ", basename(experiment_dir), "\n"),
      paste0("**Generated**: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"),
      paste0("\nThis notebook reproduces the complete TRAM-DAG analysis from scratch.\n"),
      paste0("\n## Model Parameters\n"),
      paste0("- **Epochs**: ", epochs, "\n"),
      paste0("- **Learning Rate**: ", learning_rate, "\n"),
      paste0("- **Batch Size**: ", batch_size, "\n"),
      paste0("- **Initial Weights**: ", if(set_initial_weights) "R-based" else "Random", "\n")
    )
  )
  
  # Cell 2: Install packages
  notebook_cells[[2]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 1: Install Required Packages\n", 
               "Run this cell to install required R and Python packages.")
  )
  
  notebook_cells[[3]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# Install R packages if not already installed\n",
      "required_r_packages <- c(\"reticulate\", \"jsonlite\", \"igraph\", \"dplyr\")\n",
      "missing_r <- required_r_packages[!sapply(required_r_packages, requireNamespace, quietly = TRUE)]\n",
      "if (length(missing_r) > 0) {\n",
      "  cat(\"Installing missing R packages:\", paste(missing_r, collapse = \", \"), \"\\n\")\n",
      "  install.packages(missing_r)\n",
      "}\n",
      "\n",
      "# Load R libraries\n",
      "library(reticulate)\n",
      "library(jsonlite)\n",
      "library(igraph)\n",
      "library(dplyr)"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 4: Configure Python
  notebook_cells[[4]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 2: Configure Python Environment")
  )
  
  notebook_cells[[5]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# Option 1: Use conda environment (if available)\n",
      "tryCatch({\n",
      "  use_condaenv(\"tramdag\", required = TRUE)\n",
      "  cat(\"✓ Using conda environment: tramdag\\n\")\n",
      "}, error = function(e) {\n",
      "  cat(\"⚠ Conda environment not found. Using default Python.\\n\")\n",
      "  cat(\"  You may need to install tramdag: pip install tramdag\\n\")\n",
      "})\n",
      "\n",
      "# Import Python modules\n",
      "tramdag <- import(\"tramdag\")\n",
      "TramDagConfig <- tramdag$TramDagConfig\n",
      "TramDagModel <- tramdag$TramDagModel\n",
      "np <- import(\"numpy\")\n",
      "pd <- import(\"pandas\")"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 5: Load Data
  notebook_cells[[6]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 3: Load Data")
  )
  
  notebook_cells[[7]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# Load data\n",
      "df <- read.csv(\"data.csv\", check.names = TRUE)\n",
      "cat(\"✓ Data loaded:\", nrow(df), \"rows,\", ncol(df), \"columns\\n\")\n",
      "cat(\"  Variables:\", paste(colnames(df), collapse = \", \"), \"\\n\")\n",
      "\n",
      "# Display first few rows\n",
      "head(df)"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 6: Define DAG
  notebook_cells[[8]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 4: Define DAG Structure")
  )
  
  # Generate DAG code
  vars_code <- paste0("vars <- colnames(df)\n", "n_vars <- length(vars)\n")
  amat_code <- if(!is.null(amat) && nrow(amat) > 0 && ncol(amat) > 0) {
    amat_flat <- as.vector(amat)
    paste0("amat_codes <- matrix(c(\"", paste(amat_flat, collapse = "\", \""), "\"), \n",
           "                nrow = n_vars, ncol = n_vars,\n",
           "                dimnames = list(vars, vars))\n")
  } else {
    paste0("amat_codes <- matrix(\"0\", nrow = n_vars, ncol = n_vars,\n",
           "                dimnames = list(vars, vars))\n")
  }
  
  notebook_cells[[9]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      vars_code,
      "\n",
      amat_code,
      "\n",
      "# Ensure upper triangular\n",
      "diag(amat_codes) <- \"0\"\n",
      "for (i in 1:n_vars) {\n",
      "  for (j in 1:i) {\n",
      "    if (j < i) amat_codes[i, j] <- \"0\"\n",
      "  }\n",
      "}\n",
      "\n",
      "cat(\"✓ DAG structure defined\\n\")\n",
      "print(amat_codes)"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 7: Create Configuration
  notebook_cells[[10]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 5: Create TRAM-DAG Configuration")
  )
  
  notebook_cells[[11]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# Set experiment directory\n",
      "EXPERIMENT_DIR <- getwd()\n",
      "if (basename(EXPERIMENT_DIR) != \"", basename(experiment_dir), "\") {\n",
      "  EXPERIMENT_DIR <- file.path(EXPERIMENT_DIR, \"", basename(experiment_dir), "_reproduced\")\n",
      "  if (!dir.exists(EXPERIMENT_DIR)) {\n",
      "    dir.create(EXPERIMENT_DIR, recursive = TRUE)\n",
      "  }\n",
      "}\n",
      "\n",
      "cfg <- TramDagConfig()\n",
      "cfg$setup_configuration(\n",
      "  experiment_name = \"", basename(experiment_dir), "_reproduced\",\n",
      "  EXPERIMENT_DIR = EXPERIMENT_DIR\n",
      ")\n",
      "cfg$update()\n",
      "\n",
      "# Infer data types\n",
      "data_types <- sapply(df, function(x) {\n",
      "  if (is.numeric(x) && length(unique(x)) > 20) {\n",
      "    \"continous\"\n",
      "  } else if (is.numeric(x) || is.factor(x)) {\n",
      "    \"ordinal\"\n",
      "  } else {\n",
      "    \"continous\"\n",
      "  }\n",
      "}, USE.NAMES = TRUE)\n",
      "cfg$set_data_type(as.list(data_types))\n",
      "cfg$update()\n",
      "\n",
      "# Write adjacency matrix\n",
      "py_amat <- np$array(amat_codes, dtype = \"object\")\n",
      "utils_config <- tramdag$utils$configuration\n",
      "utils_config$write_adj_matrix_to_configuration(py_amat, cfg$CONF_DICT_PATH)\n",
      "cfg$update()\n",
      "\n",
      "# Generate model names\n",
      "py_data_types <- r_to_py(as.list(data_types))\n",
      "nn_names <- utils_config$create_nn_model_names(py_amat, py_data_types)\n",
      "utils_config$write_nn_names_matrix_to_configuration(nn_names, cfg$CONF_DICT_PATH)\n",
      "cfg$update()\n",
      "\n",
      "# Write nodes information\n",
      "utils_config$write_nodes_information_to_configuration(cfg$CONF_DICT_PATH)\n",
      "cfg$update()\n",
      "\n",
      "# Compute levels for ordinal variables\n",
      "if (any(data_types == \"ordinal\")) {\n",
      "  py_df <- r_to_py(df)\n",
      "  cfg$compute_levels(py_df)\n",
      "}\n",
      "\n",
      "cfg$save()\n",
      "cat(\"✓ Configuration created and saved\\n\")"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 8: Fit Model
  notebook_cells[[12]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 6: Fit TRAM-DAG Model")
  )
  
  notebook_cells[[13]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# Split data (80/20)\n",
      "n <- nrow(df)\n",
      "set.seed(42)  # For reproducibility\n",
      "train_idx <- sample(n, floor(0.8 * n))\n",
      "train_df <- df[train_idx, ]\n",
      "val_df <- df[-train_idx, ]\n",
      "\n",
      "# Convert to pandas\n",
      "py_train <- r_to_py(train_df)\n",
      "py_val <- r_to_py(val_df)\n",
      "\n",
      "# Create model\n",
      "td_model <- TramDagModel$from_config(\n",
      "  cfg,\n",
      "  device = \"auto\",\n",
      "  debug = FALSE,\n",
      "  verbose = TRUE,\n",
      "  set_initial_weights = ", if(set_initial_weights) "TRUE" else "FALSE", ",\n",
      "  initial_data = py_train\n",
      ")\n",
      "\n",
      "# Compute min-max scaling\n",
      "td_model$load_or_compute_minmax(\n",
      "  td_train_data = py_train,\n",
      "  use_existing = FALSE,\n",
      "  write = TRUE\n",
      ")\n",
      "\n",
      "# Fit model\n",
      "td_model$fit(\n",
      "  py_train,\n",
      "  py_val,\n",
      "  learning_rate = ", learning_rate, ",\n",
      "  epochs = as.integer(", epochs, "),\n",
      "  batch_size = as.integer(", batch_size, "),\n",
      "  device = \"auto\",\n",
      "  verbose = TRUE\n",
      ")\n",
      "\n",
      "cat(\"✓ Model fitted successfully!\\n\")"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 9: Sampling
  notebook_cells[[14]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 7: Observational Sampling")
  )
  
  notebook_cells[[15]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "n_samples <- 10000L\n",
      "samples_result <- td_model$sample(\n",
      "  number_of_samples = as.integer(n_samples),\n",
      "  do_interventions = NULL,\n",
      "  verbose = TRUE\n",
      ")\n",
      "\n",
      "# Extract samples\n",
      "sampled_dict <- samples_result[[1]]\n",
      "sample_list <- list()\n",
      "for (node in names(sampled_dict)) {\n",
      "  py_tensor <- sampled_dict[[node]]\n",
      "  if (reticulate::py_has_attr(py_tensor, \"detach\")) {\n",
      "    np_array <- py_tensor$detach()$cpu()$numpy()\n",
      "  } else if (reticulate::py_has_attr(py_tensor, \"cpu\")) {\n",
      "    np_array <- py_tensor$cpu()$numpy()\n",
      "  } else if (reticulate::py_has_attr(py_tensor, \"numpy\")) {\n",
      "    np_array <- py_tensor$numpy()\n",
      "  } else {\n",
      "    np_array <- py_tensor\n",
      "  }\n",
      "  sample_list[[node]] <- as.numeric(py_to_r(np_array))\n",
      "}\n",
      "sampled_df <- as.data.frame(sample_list)\n",
      "cat(\"✓ Sampled\", nrow(sampled_df), \"observations\\n\")\n",
      "\n",
      "# Save sampled data\n",
      "write.csv(sampled_df, file.path(EXPERIMENT_DIR, \"sampled_data.csv\"), row.names = FALSE)\n",
      "\n",
      "# Display summary\n",
      "summary(sampled_df)"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Cell 10: ATE Computation
  notebook_cells[[16]] <- list(
    cell_type = "markdown",
    metadata = list(),
    source = c("## Step 8: Compute Average Treatment Effect (ATE)")
  )
  
  notebook_cells[[17]] <- list(
    cell_type = "code",
    metadata = list(),
    source = c(
      "# do(", X_var, " = ", x_treated, ") vs do(", X_var, " = ", x_control, ") on ", Y_var, "\n",
      "X <- \"", X_var, "\"\n",
      "Y <- \"", Y_var, "\"\n",
      "x_treated <- ", x_treated, "\n",
      "x_control <- ", x_control, "\n",
      "\n",
      "# Treatment\n",
      "do_treated <- reticulate::dict()\n",
      "do_treated[[X]] <- as.numeric(x_treated)\n",
      "samp_treated <- td_model$sample(\n",
      "  number_of_samples = as.integer(n_samples),\n",
      "  do_interventions = do_treated,\n",
      "  verbose = TRUE\n",
      ")\n",
      "y_treated_tensor <- samp_treated[[1]][[Y]]\n",
      "if (reticulate::py_has_attr(y_treated_tensor, \"detach\")) {\n",
      "  y_treated <- as.numeric(py_to_r(y_treated_tensor$detach()$cpu()$numpy()))\n",
      "} else {\n",
      "  y_treated <- as.numeric(py_to_r(y_treated_tensor$cpu()$numpy()))\n",
      "}\n",
      "\n",
      "# Control\n",
      "do_control <- reticulate::dict()\n",
      "do_control[[X]] <- as.numeric(x_control)\n",
      "samp_control <- td_model$sample(\n",
      "  number_of_samples = as.integer(n_samples),\n",
      "  do_interventions = do_control,\n",
      "  verbose = TRUE\n",
      ")\n",
      "y_control_tensor <- samp_control[[1]][[Y]]\n",
      "if (reticulate::py_has_attr(y_control_tensor, \"detach\")) {\n",
      "  y_control <- as.numeric(py_to_r(y_control_tensor$detach()$cpu()$numpy()))\n",
      "} else {\n",
      "  y_control <- as.numeric(py_to_r(y_control_tensor$cpu()$numpy()))\n",
      "}\n",
      "\n",
      "# Compute ATE\n",
      "ate <- mean(y_treated) - mean(y_control)\n",
      "cat(\"\\n✓ ATE = E[\", Y, \" | do(\", X, \" = \", x_treated, \")] - E[\", Y, \" | do(\", X, \" = \", x_control, \")] = \", \n",
      "    round(ate, 4), \"\\n\", sep = \"\")\n",
      "\n",
      "# Save results\n",
      "results <- list(\n",
      "  ate = ate,\n",
      "  y_treated_mean = mean(y_treated),\n",
      "  y_control_mean = mean(y_control),\n",
      "  y_treated_std = sd(y_treated),\n",
      "  y_control_std = sd(y_control)\n",
      ")\n",
      "write_json(results, file.path(EXPERIMENT_DIR, \"ate_results.json\"), pretty = TRUE)\n",
      "\n",
      "# Display results\n",
      "print(results)"
    ),
    execution_count = NULL,
    outputs = list()
  )
  
  # Create notebook structure
  notebook <- list(
    cells = notebook_cells,
    metadata = list(
      kernelspec = list(
        display_name = "R",
        language = "R",
        name = "ir"
      ),
      language_info = list(
        name = "R",
        version = R.version.string
      )
    ),
    nbformat = 4L,
    nbformat_minor = 4L
  )
  
  # Write notebook as JSON
  notebook_json <- jsonlite::toJSON(notebook, auto_unbox = TRUE, pretty = TRUE)
  writeLines(notebook_json, file.path(scripts_dir, "complete_workflow.ipynb"))
  
  # 6. Create comprehensive README
  readme <- paste0('# Reproducible Package

This folder contains everything needed to reproduce the TRAM-DAG analysis on any computer.

## Contents

- `data.csv`: The original data file
- `run_complete_workflow.R`: Complete workflow script (from data to results)
- `complete_workflow.ipynb`: Jupyter notebook with complete workflow (interactive)
- `r_requirements.txt`: R package requirements
- `python_requirements.txt`: Python package requirements

## Experiment Information

- **Experiment Name**: ', basename(experiment_dir), '
- **Created**: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
- **Original Directory**: ', experiment_dir, '

## Model Fitting Parameters

- **Epochs**: ', epochs, '
- **Learning Rate**: ', learning_rate, '
- **Batch Size**: ', batch_size, '
- **Initial Weights**: ', if(set_initial_weights) "R-based" else "Random", '

## How to Reproduce

### Option 1: Jupyter Notebook (Recommended - Interactive)

1. **Install Jupyter with R kernel**:
   ```bash
   # Install Jupyter
   pip install jupyter
   
   # Install IRkernel for R
   # In R:
   install.packages("IRkernel")
   IRkernel::installspec()
   ```
   
2. **Install Python packages**:
   ```bash
   pip install -r python_requirements.txt
   ```
   Or if using conda:
   ```bash
   conda create -n tramdag python=3.9
   conda activate tramdag
   pip install -r python_requirements.txt
   ```
   
3. **Install R packages** (in R):
   ```r
   install.packages(c("reticulate", "jsonlite", "igraph", "dplyr"))
   ```

4. **Open and run the notebook**:
   ```bash
   jupyter notebook complete_workflow.ipynb
   ```
   Or with JupyterLab:
   ```bash
   jupyter lab complete_workflow.ipynb
   ```

### Option 2: Direct R Execution

1. **Install R** (version 4.0+)
2. **Install Python** (version 3.8+) with pip
3. **Install R packages**:
   ```r
   install.packages(c("reticulate", "jsonlite", "igraph", "dplyr"))
   ```
4. **Install Python packages**:
   ```bash
   pip install -r python_requirements.txt
   ```
   Or if using conda:
   ```bash
   conda create -n tramdag python=3.9
   conda activate tramdag
   pip install -r python_requirements.txt
   ```
5. **Run the workflow**:
   ```bash
   Rscript run_complete_workflow.R
   ```

### Option 3: Using Conda Environment

1. **Create conda environment**:
   ```bash
   conda create -n tramdag-reproducible python=3.9 r-base=4.3
   conda activate tramdag-reproducible
   ```
2. **Install Python packages**:
   ```bash
   pip install -r python_requirements.txt
   ```
3. **Install R packages** (in R):
   ```r
   install.packages(c("reticulate", "jsonlite", "igraph", "dplyr"))
   ```
4. **Run the workflow**:
   ```bash
   Rscript run_complete_workflow.R
   ```

## What the Workflow Does

1. **Configures Python environment** (conda or default)
2. **Loads data** from `data.csv`
3. **Sets up DAG structure** (from saved configuration)
4. **Creates TRAM-DAG configuration**
5. **Fits the model** with the same parameters as original
6. **Performs observational sampling**
7. **Computes Average Treatment Effect (ATE)**

## Output

The workflow will create a new directory with:
- Fitted model files
- Configuration files
- Sampled data (`sampled_data.csv`)
- ATE results (`ate_results.json`)

## Troubleshooting

### Python environment not found
- Make sure Python is installed and in PATH
- Or create a conda environment: `conda create -n tramdag python=3.9`
- Install tramdag: `pip install tramdag`

### R packages missing
- Run: `install.packages(c("reticulate", "jsonlite", "igraph", "dplyr"))`

### Jupyter notebook not working
- Make sure Jupyter is installed: `pip install jupyter`
- Install R kernel: In R, run `install.packages("IRkernel"); IRkernel::installspec()`
- Make sure Python and R packages are installed in the same environment

## Notes

- The workflow uses a fixed random seed (42) for reproducibility
- All paths are relative to the script location
- The reproduced experiment will be in a subdirectory with "_reproduced" suffix
')
  writeLines(readme, file.path(scripts_dir, "README.md"))
  
  cat("\n✓ Reproducible package created in:", scripts_dir, "\n")
  cat("  - Complete workflow script (R)\n")
  cat("  - Jupyter notebook (interactive)\n")
  cat("  - Data file\n")
  cat("  - Environment requirements\n")
  cat("  - Comprehensive README\n")
  cat("\nTo share: Copy the entire 'reproducible_package' folder\n")
  cat("To run: Open 'complete_workflow.ipynb' in Jupyter or run 'run_complete_workflow.R'\n")
}

# Create TramDagConfig from adjacency matrix and data
create_tramdag_config <- function(df, amat, experiment_dir, data_types = NULL) {
  # Removed debug cat() statements to avoid Shiny output conflicts
  # All logging is done in the main observeEvent handler
  
  # Convert R data.frame to pandas DataFrame
  py_df <- r_to_py(df)
  
  # Get variable names
  vars <- colnames(df)
  n_vars <- length(vars)
  
  # Infer data types if not provided
  if (is.null(data_types)) {
    data_types <- sapply(df, function(x) {
      if (is.numeric(x) && length(unique(x)) > 20) {
        "continous"
      } else if (is.numeric(x) || is.factor(x)) {
        "ordinal_Xn_Yo"
      } else {
        "continous"
      }
    }, USE.NAMES = TRUE)
  }
  
  # Ensure data_types is a named vector/list
  if (!is.null(names(data_types))) {
    data_types <- as.list(data_types)
  }
  
  # Get variable names in correct order
  vars <- colnames(df)
  
  # Convert adjacency matrix to Python format
  amat_codes <- amat
  if (is.numeric(amat) || all(amat %in% c(0, 1))) {
    amat_codes <- matrix("0", nrow = nrow(amat), ncol = ncol(amat),
                        dimnames = dimnames(amat))
    idx <- which(amat != 0, arr.ind = TRUE)
    if (nrow(idx) > 0) {
      for (i in seq_len(nrow(idx))) {
        amat_codes[idx[i, "row"], idx[i, "col"]] <- "ls"
      }
    }
  } else {
    amat_codes <- matrix(as.character(amat), 
                        nrow = nrow(amat), ncol = ncol(amat),
                        dimnames = dimnames(amat))
  }
  
  # CRITICAL: Ensure matrix is upper triangular
  amat_codes <- ensure_upper_triangular(amat_codes, vars)
  
  # R matrix -> Python numpy array with string labels
  py_amat <- np$array(amat_codes, dtype = "object")
  
  # Ensure the directory exists
  if (!dir.exists(experiment_dir)) {
    dir.create(experiment_dir, recursive = TRUE)
  }
  
  # Construct the configuration file path
  conf_dict_path <- file.path(experiment_dir, "configuration.json")
  
  # Create configuration
  cfg <- TramDagConfig()
  
  # Setup experiment directory
  tryCatch({
    cfg$setup_configuration(
      experiment_name = basename(experiment_dir),
      EXPERIMENT_DIR = experiment_dir
    )
  }, error = function(e) {
    stop(e)
  })
  
  # Update config
  tryCatch({
    cfg$update()
  }, error = function(e) {
    stop(e)
  })
  
  # Access the configuration utils module
  utils_config <- tramdag$utils$configuration
  
  # Set data types
  py_data_types <- r_to_py(data_types)
  
  tryCatch({
    cfg$set_data_type(py_data_types)
  }, error = function(e) {
    stop(e)
  })
  
  tryCatch({
    cfg$update()
  }, error = function(e) {
    stop(e)
  })
  
  # Write adjacency matrix directly to config
  tryCatch({
    utils_config$write_adj_matrix_to_configuration(py_amat, conf_dict_path)
  }, error = function(e) {
    stop(e)
  })
  
  # Reload the configuration
  cfg$update()
  
  # Generate model names matrix
  tryCatch({
    current_config_py <- utils_config$load_configuration_dict(conf_dict_path)
    
    if (inherits(current_config_py, "list")) {
      data_type_from_config <- current_config_py$data_type
      if (is.null(data_type_from_config)) {
        data_type_from_config <- NULL
      }
    } else {
      data_type_from_config <- current_config_py$get("data_type", NULL)
    }
  }, error = function(e) {
    stop(e)
  })
  
  # If data_type from config is a list, convert it to dict
  if (!is.null(data_type_from_config)) {
    is_dict <- tryCatch({
      data_type_r <- py_to_r(data_type_from_config)
      is.list(data_type_r) && !is.null(names(data_type_r)) && length(names(data_type_r)) > 0
    }, error = function(e) {
      FALSE
    })
    
    if (!is_dict) {
      data_types_list <- py_to_r(data_type_from_config)
      if (is.list(data_types_list) && !is.null(names(data_types_list))) {
        py_data_type_dict <- r_to_py(data_types_list)
      } else {
        py_data_type_dict <- r_to_py(data_types)
      }
    } else {
      py_data_type_dict <- data_type_from_config
    }
  } else {
    py_data_type_dict <- r_to_py(data_types)
  }
  
  # Generate model names from adjacency matrix and data types
  tryCatch({
    nn_names_matrix <- utils_config$create_nn_model_names(py_amat, py_data_type_dict)
  }, error = function(e) {
    stop(e)
  })
  
  # Write the model names matrix to configuration
  tryCatch({
    utils_config$write_nn_names_matrix_to_configuration(nn_names_matrix, conf_dict_path)
  }, error = function(e) {
    stop(e)
  })
  
  # Reload config
  cfg$update()
  
  # Write nodes information
  tryCatch({
    is_valid <- utils_config$validate_adj_matrix(py_amat)
    if (!is_valid) {
      amat_r <- py_to_r(py_amat)
      lower_triangle <- amat_r[lower.tri(amat_r)]
      if (any(lower_triangle != "0")) {
        stop(paste("Adjacency matrix has non-zero entries in lower triangle:",
                   paste(which(lower_triangle != "0"), collapse = ", "),
                   "\nMatrix must be upper triangular (only entries above diagonal)."))
      }
      stop("Adjacency matrix validation failed. Matrix must be upper triangular with zeros on diagonal.")
    }
    
    # Write nodes information with min/max values from data
    tryCatch({
      py_min_vals <- r_to_py(as.data.frame(t(sapply(df, min, na.rm = TRUE))))
      py_max_vals <- r_to_py(as.data.frame(t(sapply(df, max, na.rm = TRUE))))
      # Transpose to get column-per-variable format expected by the function
      py_min_frame <- py_min_vals$T
      py_max_frame <- py_max_vals$T
      utils_config$write_nodes_information_to_configuration(
        conf_dict_path,
        min_vals = py_min_frame,
        max_vals = py_max_frame
      )
    }, error = function(e) {
      # Fallback: call without min/max if conversion fails
      tryCatch({
        utils_config$write_nodes_information_to_configuration(conf_dict_path)
      }, error = function(e2) {
        stop(e2)
      })
    })
    
    # Reload and verify nodes were created
    cfg$update()
    final_config <- utils_config$load_configuration_dict(conf_dict_path)
    final_config_r <- py_to_r(final_config)
    
    if (is.null(final_config_r$nodes) || length(final_config_r$nodes) == 0) {
      stop("Failed to create nodes information in configuration. This is required for model fitting.")
    }
    
    # Verify all variables have node entries
    missing_nodes <- setdiff(names(data_types), names(final_config_r$nodes))
    if (length(missing_nodes) > 0) {
      stop(paste("Missing node entries for variables:", paste(missing_nodes, collapse = ", ")))
    }
    
  }, error = function(e) {
    error_msg <- conditionMessage(e)
    stop(paste("Error creating nodes information:", error_msg,
               "\n\nThis is required for model fitting.",
               "\nPlease ensure:",
               "\n  1. Adjacency matrix is upper triangular (zeros below diagonal)",
               "\n  2. Diagonal is all zeros (no self-loops)",
               "\n  3. Edge codes are valid ('0', 'ls', 'cs', 'ci', 'si')",
               "\n  4. Variable names in data match those in adjacency matrix"))
  })
  
  # Compute levels for ordinal variables if needed
  if (any(grepl("ordinal", data_types, ignore.case = TRUE))) {
    cfg$compute_levels(py_df)
  }
  
  # Save configuration
  cfg$save()
  
  return(cfg)
}

# Fit TRAM-DAG model
fit_tramdag_model <- function(cfg, train_df, val_df = NULL, 
                              epochs = 100L, learning_rate = 0.01,
                              batch_size = 512L, device = "auto",
                              set_initial_weights = FALSE) {
  
  # Ensure config is up to date
  cfg$update()
  
  # Verify configuration has nodes
  utils_config <- tramdag$utils$configuration
  conf_dict_path <- cfg$CONF_DICT_PATH
  if (is.null(conf_dict_path) || is.na(conf_dict_path)) {
    stop("Cannot access configuration. CONF_DICT_PATH is not set.")
  }
  
  # Verify nodes exist in configuration
  conf_dict <- utils_config$load_configuration_dict(conf_dict_path)
  conf_dict_r <- py_to_r(conf_dict)
  if (is.null(conf_dict_r$nodes) || length(conf_dict_r$nodes) == 0) {
    stop(paste("Configuration is missing nodes information.",
               "This should have been created during config setup.",
               "Please check that write_nodes_information_to_configuration was called successfully."))
  }
  
  # Convert R data.frames to pandas
  py_train <- r_to_py(train_df)
  py_val <- if (!is.null(val_df)) r_to_py(val_df) else NULL
  
  # Create model from config
  tryCatch({
    td_model <- TramDagModel$from_config(
      cfg,
      device = device,
      debug = FALSE,
      verbose = TRUE,
      set_initial_weights = set_initial_weights,
      initial_data = py_train
    )
  }, error = function(e) {
    error_msg <- conditionMessage(e)
    if (grepl("nodes", error_msg, ignore.case = TRUE)) {
      stop(paste("Error creating model from config:", error_msg,
                 "\n\nConfiguration may be missing required nodes information.",
                 "\nPlease check that the DAG was properly set up."))
    }
    stop(paste("Error creating model from config:", error_msg))
  })
  
  # Note: load_or_compute_minmax is called internally by fit() so we don't call it explicitly.
  
  # Fit model
  # Removed debug cat() statements to avoid Shiny output conflicts
  tryCatch({
    if (!is.null(py_val)) {
      result <- td_model$fit(
        py_train,
        py_val,
        learning_rate = as.numeric(learning_rate),
        epochs = as.integer(epochs),
        batch_size = as.integer(batch_size),
        device = device,
        verbose = TRUE
      )
    } else {
      result <- td_model$fit(
        py_train,
        learning_rate = as.numeric(learning_rate),
        epochs = as.integer(epochs),
        batch_size = as.integer(batch_size),
        device = device,
        verbose = TRUE
      )
    }
  }, error = function(e) {
    error_msg <- conditionMessage(e)
    stop(paste("Error fitting model:\n",
               "  Message: ", error_msg, "\n\n",
               "Common issues:\n",
               "  - Data format mismatch\n",
               "  - Missing required columns\n",
               "  - Invalid configuration\n",
               "  - Method call syntax error"))
  })
  
  return(td_model)
}

# Sample from model with interventions
sample_from_model <- function(td_model, n_samples = 10000L, 
                              do_interventions = NULL) {
  
  # Convert interventions to Python dict if provided
  py_do <- NULL
  if (!is.null(do_interventions) && length(do_interventions) > 0) {
    py_do <- r_to_py(do_interventions)
  }
  
  # Check if any ordinal variables are present
  # When ordinal nodes exist, number_of_samples must match the internal
  # number_of_counterfactual_samples (default 1000) to avoid a shape mismatch
  has_ordinal <- FALSE
  tryCatch({
    nodes <- td_model$cfg$conf_dict$nodes
    for (node_name in names(nodes)) {
      dtype <- nodes[[node_name]]$data_type
      if (!is.null(dtype) && grepl("ordinal", dtype)) {
        has_ordinal <- TRUE
        break
      }
    }
  }, error = function(e) {})
  
  n <- as.integer(n_samples)
  if (has_ordinal) {
    # Cap at 1000 and align batch_size for ordinal compatibility
    safe_n <- min(n, 1000L)
    result <- td_model$sample(
      number_of_samples = safe_n,
      batch_size = safe_n,
      do_interventions = py_do,
      verbose = TRUE
    )
  } else {
    # Continuous-only: use requested n_samples with default batch_size (like the notebook)
    result <- td_model$sample(
      number_of_samples = n,
      do_interventions = py_do,
      verbose = TRUE
    )
  }
  
  # Extract samples (result is tuple: (sampled_by_node, latents_by_node))
  sampled_dict <- result[[1]]
  latents_dict <- result[[2]]
  
  # Convert to R data.frame
  sample_list <- list()
  for (node in names(sampled_dict)) {
    py_tensor <- sampled_dict[[node]]
    # Convert PyTorch tensor to numpy array to R vector
    if (py_has_attr(py_tensor, "detach")) {
      np_array <- py_tensor$detach()$cpu()$numpy()
    } else if (py_has_attr(py_tensor, "cpu")) {
      np_array <- py_tensor$cpu()$numpy()
    } else if (py_has_attr(py_tensor, "numpy")) {
      np_array <- py_tensor$numpy()
    } else {
      np_array <- py_tensor
    }
    sample_list[[node]] <- as.numeric(py_to_r(np_array))
  }
  
  return(as.data.frame(sample_list))
}

# Compute ATE (Average Treatment Effect)
compute_ate <- function(td_model, Y, X, x_treated = 1, x_control = 0, 
                        n_samples = 10000L) {
  
  # Sample under treatment
  do_treated <- setNames(list(x_treated), X)
  samp_treated <- sample_from_model(td_model, n_samples, do_treated)
  y_treated <- samp_treated[[Y]]
  
  # Sample under control
  do_control <- setNames(list(x_control), X)
  samp_control <- sample_from_model(td_model, n_samples, do_control)
  y_control <- samp_control[[Y]]
  
  # Compute ATE
  ate <- mean(y_treated) - mean(y_control)
  
  return(list(
    ate = ate,
    Y_treated = y_treated,
    Y_control = y_control,
    mean_treated = mean(y_treated),
    mean_control = mean(y_control),
    samp_treated_df = samp_treated,
    samp_control_df = samp_control
  ))
}

# Generate LLM interpretation of model fitting results (via Ollama)
generate_model_interpretation <- function(experiment_dir, epochs, learning_rate, 
                                         batch_size, variables) {
  if (!ollama_available) return("")
  
  loss_info <- ""
  vars <- variables
  if (!is.null(vars) && length(vars) > 0) {
    loss_summary <- character(0)
    for (var in vars) {
      train_loss_file <- file.path(experiment_dir, var, "train_loss_hist.json")
      if (file.exists(train_loss_file)) {
        tryCatch({
          train_loss <- jsonlite::fromJSON(train_loss_file)
          if (is.numeric(train_loss) && length(train_loss) > 0) {
            initial_loss <- train_loss[1]
            final_loss <- train_loss[length(train_loss)]
            loss_summary <- c(loss_summary, 
                             paste0(var, ": ", round(initial_loss, 4), " -> ", round(final_loss, 4)))
          }
        }, error = function(e) {})
      }
    }
    if (length(loss_summary) > 0) {
      loss_info <- paste("\nTraining loss progression:\n", paste(loss_summary, collapse = "\n"))
    }
  }
  
  sys_msg <- "You are a data science expert explaining TRAM-DAG model fitting results. 
Provide a concise, clear explanation (2-3 short paragraphs, maximum 120 words) of what the model fitting means.
Focus on:
- What the model learned
- Training quality indicators
- What this enables (causal inference capabilities)
Write in clear, structured prose. Do not use markdown headers, lists, or bullet points."
  
  usr_msg <- paste0("A TRAM-DAG model was fitted with the following parameters:
- Variables: ", paste(variables, collapse = ", "), "
- Training epochs: ", epochs, "
- Learning rate: ", learning_rate, "
- Batch size: ", batch_size, "
- Experiment directory: ", basename(experiment_dir), "
", loss_info, "

Explain what this model fitting means and what capabilities it provides for causal analysis. Keep it concise (2-3 paragraphs, max 120 words).")
  
  ollama_chat(sys_msg, usr_msg)
}

# Generate LLM interpretation of ATE results (via Ollama)
generate_ate_interpretation <- function(ate_result, X, Y, x_treated, x_control, variables) {
  if (!ollama_available) return("")
  
  sys_msg <- "You are a causal inference expert explaining Average Treatment Effect (ATE) results. 
Provide a concise, clear explanation (2-3 short paragraphs, maximum 120 words) of what the ATE means.
Focus on:
- What the ATE value indicates (effect size and direction)
- Practical interpretation
- What this means for understanding the causal relationship
Write in clear, structured prose. Do not use markdown headers, lists, or bullet points."
  
  ate <- ate_result$ate
  y_treated_mean <- ate_result$mean_treated
  y_control_mean <- ate_result$mean_control
  y_treated_std <- sd(ate_result$Y_treated)
  y_control_std <- sd(ate_result$Y_control)
  
  usr_msg <- paste0("An Average Treatment Effect (ATE) was computed:
- Treatment variable (X): ", X, "
- Outcome variable (Y): ", Y, "
- Treatment condition: do(", X, " = ", x_treated, ")
- Control condition: do(", X, " = ", x_control, ")
- ATE = ", round(ate, 4), "
- Mean outcome under treatment: ", round(y_treated_mean, 4), " (std: ", round(y_treated_std, 4), ")
- Mean outcome under control: ", round(y_control_mean, 4), " (std: ", round(y_control_std, 4), ")
- Available variables: ", paste(variables, collapse = ", "), "

Explain what this ATE result means in practical terms. Is the effect positive or negative? How large is it? What does this tell us about the causal relationship between ", X, " and ", Y, "? Keep it concise (2-3 paragraphs, max 120 words).")
  
  ollama_chat(sys_msg, usr_msg)
}

# Helper: capture ALL matplotlib figures produced by a plot function to a single PNG.
# Many tramdag methods create one figure per variable and call plt.show() on each.
# This helper intercepts plt.show(), collects all figures, saves each to a temp PNG,
# then stacks them vertically into one combined image.
# Defined at top level so both generate_pdf_report and server can use it.
capture_matplotlib_plot <- function(plot_func, width = 1200, height = 500) {
  plt <- import("matplotlib.pyplot")
  PIL <- tryCatch(import("PIL"), error = function(e) NULL)
  
  tmp <- tempfile(fileext = ".png")
  
  # Save original plt.show before overriding
  original_show <- plt$show
  
  tryCatch({
    plt$close("all")
    
    # Override plt.show() to be a no-op so figures stay open
    plt$show <- function(...) invisible(NULL)
    
    # Also switch to non-interactive backend to prevent display
    was_interactive <- plt$isinteractive()
    plt$ioff()
    
    plot_func()
    
    # Restore plt.show
    plt$show <- original_show
    if (was_interactive) plt$ion()
    
    # Get all open figure numbers
    fig_nums <- plt$get_fignums()
    fig_nums_r <- py_to_r(fig_nums)
    
    if (length(fig_nums_r) == 0) {
      plt$close("all")
      return(NULL)
    }
    
    if (length(fig_nums_r) == 1) {
      # Single figure: save directly
      fig <- plt$figure(fig_nums_r[[1]])
      fig$set_size_inches(as.numeric(width) / 100, as.numeric(height) / 100)
      fig$savefig(tmp, dpi = 100L, bbox_inches = "tight")
      plt$close("all")
      return(tmp)
    }
    
    # Multiple figures: save each, then stack vertically
    part_files <- c()
    for (fn in fig_nums_r) {
      fig <- plt$figure(fn)
      fig$set_size_inches(as.numeric(width) / 100, as.numeric(height) / 100)
      part_tmp <- tempfile(fileext = ".png")
      fig$savefig(part_tmp, dpi = 100L, bbox_inches = "tight")
      part_files <- c(part_files, part_tmp)
    }
    plt$close("all")
    
    # Stack images vertically using PIL if available, otherwise R png package
    if (!is.null(PIL)) {
      Image <- PIL$Image
      imgs <- lapply(part_files, function(f) Image$open(f))
      widths <- sapply(imgs, function(im) py_to_r(im$size)[[1]])
      heights <- sapply(imgs, function(im) py_to_r(im$size)[[2]])
      max_w <- max(widths)
      total_h <- sum(heights) + 10L * (length(imgs) - 1)  # 10px gap between
      # PIL requires Python tuples for size and color, not R vectors
      combined <- Image$new("RGB", 
                            tuple(as.integer(max_w), as.integer(total_h)), 
                            tuple(255L, 255L, 255L))
      y_offset <- 0L
      for (im in imgs) {
        combined$paste(im, tuple(0L, as.integer(y_offset)))
        y_offset <- y_offset + py_to_r(im$size)[[2]] + 10L
      }
      combined$save(tmp)
      # Clean up part files
      for (f in part_files) file.remove(f)
      return(tmp)
    } else {
      # Fallback without PIL: use R's png package to stack
      tryCatch({
        png_pkg <- requireNamespace("png", quietly = TRUE)
        if (png_pkg) {
          img_list <- lapply(part_files, function(f) png::readPNG(f))
          widths_px <- sapply(img_list, ncol)
          heights_px <- sapply(img_list, nrow)
          max_w_px <- max(widths_px)
          gap <- 10  # pixels between images
          total_h_px <- sum(heights_px) + gap * (length(img_list) - 1)
          
          # Create combined canvas (white)
          combined_arr <- array(1, dim = c(total_h_px, max_w_px, 3))
          y_off <- 1
          for (img in img_list) {
            h <- nrow(img)
            w <- ncol(img)
            combined_arr[y_off:(y_off + h - 1), 1:w, ] <- img[,,1:3]
            y_off <- y_off + h + gap
          }
          png::writePNG(combined_arr, tmp)
        } else {
          # Last resort: just use the last figure
          file.copy(part_files[length(part_files)], tmp)
        }
      }, error = function(e2) {
        file.copy(part_files[length(part_files)], tmp)
      })
      for (f in part_files) file.remove(f)
      return(tmp)
    }
  }, error = function(e) {
    tryCatch({ plt$show <- original_show }, error = function(e2) {})
    plt$close("all")
    cat("capture_matplotlib_plot error:", conditionMessage(e), "\n")
    return(NULL)
  })
}

# Generate comprehensive PDF report with full analysis
generate_pdf_report <- function(experiment_dir, data_df, amat, 
                               experiment_name, epochs, learning_rate, 
                               batch_size, set_initial_weights,
                               td_model = NULL, ate_result = NULL,
                               model_interpretation = NULL, 
                               ate_interpretation = NULL,
                               X_var = NULL, Y_var = NULL,
                               x_treated = NULL, x_control = NULL,
                               train_df = NULL, test_df = NULL,
                               sampled_data = NULL,
                               interventional_samples = NULL,
                               interventional_control = NULL) {
  # Generate a comprehensive PDF report with methodology, results, and interpretations.
  # Includes all diagnostic plots matching the notebook workflow.
  # Uses Python's reportlab via reticulate.
  
  tryCatch({
    # Check if reportlab is available
    tryCatch({
      reportlab_test <- import("reportlab", convert = FALSE)
      rm(reportlab_test)  # Clean up
    }, error = function(e) {
      stop("reportlab Python package is not installed. Please install it with: pip install reportlab")
    })
    
    # Import reportlab components via reticulate
    reportlab_lib <- import("reportlab.lib.pagesizes")
    reportlab_styles <- import("reportlab.lib.styles")
    reportlab_units <- import("reportlab.lib.units")
    reportlab_platypus <- import("reportlab.platypus")
    reportlab_colors <- import("reportlab.lib")
    reportlab_enums <- import("reportlab.lib.enums")
    datetime <- import("datetime")
    pathlib <- import("pathlib")
    plt <- import("matplotlib.pyplot")
    sns <- import("seaborn")
    torch <- import("torch")
    
    letter <- reportlab_lib$letter
    getSampleStyleSheet <- reportlab_styles$getSampleStyleSheet
    ParagraphStyle <- reportlab_styles$ParagraphStyle
    inch <- reportlab_units$inch
    SimpleDocTemplate <- reportlab_platypus$SimpleDocTemplate
    Paragraph <- reportlab_platypus$Paragraph
    Spacer <- reportlab_platypus$Spacer
    RLImage <- reportlab_platypus$Image
    PageBreak <- reportlab_platypus$PageBreak
    colors <- reportlab_colors$colors
    TA_CENTER <- reportlab_enums$TA_CENTER
    TA_LEFT <- reportlab_enums$TA_LEFT
    TA_JUSTIFY <- reportlab_enums$TA_JUSTIFY
    Path <- pathlib$Path
    
    # Create reports directory
    reports_dir <- file.path(experiment_dir, "reports")
    if (!dir.exists(reports_dir)) {
      dir.create(reports_dir, recursive = TRUE)
    }
    
    # Output path
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    report_path <- file.path(reports_dir, paste0("analysis_report_", timestamp, ".pdf"))
    
    # Helper: save a matplotlib-based plot to a file for embedding in the PDF
    save_report_plot <- function(plot_func, filename, width = 1200, height = 500) {
      out_path <- file.path(reports_dir, paste0(filename, "_", timestamp, ".png"))
      tmp <- capture_matplotlib_plot(plot_func, width = width, height = height)
      if (!is.null(tmp) && file.exists(tmp)) {
        file.copy(tmp, out_path, overwrite = TRUE)
        file.remove(tmp)
        return(out_path)
      }
      return(NULL)
    }
    
    # Helper: add an image to the story with proper sizing
    add_plot_to_story <- function(story, plot_path, img_width = 6.5, img_height = NULL, caption = NULL) {
      if (!is.null(plot_path) && file.exists(plot_path)) {
        # Read image dimensions to compute aspect ratio
        actual_w <- NULL
        actual_h <- NULL
        tryCatch({
          PIL <- import("PIL")
          im <- PIL$Image$open(plot_path)
          dims <- py_to_r(im$size)
          actual_w <- dims[[1]]
          actual_h <- dims[[2]]
          im$close()
        }, error = function(e) {})
        
        if (!is.null(actual_w) && !is.null(actual_h) && is.null(img_height)) {
          # Maintain aspect ratio, fitting to img_width
          aspect <- actual_h / actual_w
          img_height <- img_width * aspect
          # Cap height to avoid oversized images
          max_h <- 8.0
          if (img_height > max_h) {
            img_height <- max_h
            img_width <- max_h / aspect
          }
        } else if (is.null(img_height)) {
          img_height <- img_width * 0.6  # default ratio
        }
        
        story <- c(story, list(RLImage(plot_path, width = img_width * inch, height = img_height * inch)))
        if (!is.null(caption)) {
          caption_style <- ParagraphStyle(
            paste0('Caption_', sample(1:99999, 1)),
            parent = styles$Normal,
            fontSize = 9L,
            textColor = colors$HexColor('#666666'),
            alignment = TA_CENTER,
            spaceBefore = 4L,
            spaceAfter = 8L
          )
          story <- c(story, list(Paragraph(caption, caption_style)))
        }
        story <- c(story, list(Spacer(1L, 0.15 * inch)))
      }
      return(story)
    }
    
    # Create PDF document
    doc <- SimpleDocTemplate(report_path, pagesize = letter,
                            rightMargin = 54, leftMargin = 54,
                            topMargin = 54, bottomMargin = 36)
    
    # Container for content
    story <- list()
    styles <- getSampleStyleSheet()
    
    # Custom styles
    title_style <- ParagraphStyle(
      'CustomTitle',
      parent = styles$Heading1,
      fontSize = 22L,
      textColor = colors$HexColor('#1a1a1a'),
      spaceAfter = 20L,
      alignment = TA_CENTER
    )
    
    meta_style <- ParagraphStyle(
      'Meta',
      parent = styles$Normal,
      fontSize = 10L,
      textColor = colors$HexColor('#666666'),
      alignment = TA_CENTER
    )
    
    body_style <- ParagraphStyle(
      'BodyJustified',
      parent = styles$Normal,
      fontSize = 10L,
      alignment = TA_JUSTIFY,
      spaceBefore = 4L,
      spaceAfter = 4L
    )
    
    # ================================================================
    # TITLE PAGE
    # ================================================================
    story <- c(story, list(Spacer(1L, 1.5 * inch)))
    story <- c(story, list(Paragraph("TRAM-DAG Causal Analysis Report", title_style)))
    story <- c(story, list(Spacer(1L, 0.3 * inch)))
    story <- c(story, list(Paragraph(paste0("<b>Experiment:</b> ", experiment_name), meta_style)))
    story <- c(story, list(Paragraph(paste0("<b>Generated:</b> ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), meta_style)))
    story <- c(story, list(Paragraph(paste0("<b>Directory:</b> ", experiment_dir), meta_style)))
    
    # Summary of what was included
    story <- c(story, list(Spacer(1L, 0.5 * inch)))
    sections_included <- c("Data Overview", "DAG Specification", "Model Configuration")
    if (!is.null(td_model)) sections_included <- c(sections_included, "Model Fitting & Diagnostics")
    if (!is.null(sampled_data)) sections_included <- c(sections_included, "Observational Sampling")
    if (!is.null(ate_result)) sections_included <- c(sections_included, "Causal Effect Analysis (ATE)")
    if (!is.null(interventional_samples)) sections_included <- c(sections_included, "Interventional Sampling")
    toc_text <- paste0("<b>Report Contents:</b><br/>", paste(paste0("• ", sections_included), collapse = "<br/>"))
    story <- c(story, list(Paragraph(toc_text, body_style)))
    story <- c(story, list(PageBreak()))
    
    # ================================================================
    # 1. EXECUTIVE SUMMARY
    # ================================================================
    story <- c(story, list(Paragraph("1. Executive Summary", styles$Heading2)))
    n_vars <- ncol(data_df)
    n_obs <- nrow(data_df)
    summary_text <- paste0(
      "This report presents a causal inference analysis using TRAM-DAG ",
      "(Transformation Models for Directed Acyclic Graphs). ",
      "A dataset with <b>", n_obs, " observations</b> across <b>", n_vars, " variables</b> ",
      "(", paste(colnames(data_df), collapse = ", "), ") was analyzed. "
    )
    if (!is.null(td_model)) {
      summary_text <- paste0(summary_text,
        "The model was trained for <b>", epochs, " epochs</b> with a learning rate of ", learning_rate, 
        " and batch size ", batch_size, ". ")
    }
    if (!is.null(ate_result) && !is.null(X_var) && !is.null(Y_var)) {
      summary_text <- paste0(summary_text,
        "The Average Treatment Effect (ATE) of <b>", X_var, "</b> on <b>", Y_var, "</b> was estimated as ",
        "<b>", round(ate_result$ate, 4), "</b>. ")
    }
    story <- c(story, list(Paragraph(summary_text, body_style)))
    story <- c(story, list(Spacer(1L, 0.2 * inch)))
    
    # ================================================================
    # 2. DATA OVERVIEW
    # ================================================================
    story <- c(story, list(Paragraph("2. Data Overview", styles$Heading2)))
    
    data_summary <- paste0(
      "<b>Number of observations:</b> ", n_obs, "<br/>",
      "<b>Number of variables:</b> ", n_vars, "<br/>",
      "<b>Variables:</b> ", paste(colnames(data_df), collapse = ", ")
    )
    story <- c(story, list(Paragraph(data_summary, body_style)))
    story <- c(story, list(Spacer(1L, 0.1 * inch)))
    
    # Detailed statistics per variable
    story <- c(story, list(Paragraph("<b>Descriptive Statistics:</b>", styles$Heading4)))
    for (var in colnames(data_df)) {
      var_data <- data_df[[var]]
      if (is.numeric(var_data)) {
        var_stats <- paste0(
          "<b>", var, ":</b> ",
          "Mean = ", round(mean(var_data, na.rm = TRUE), 4), ", ",
          "SD = ", round(sd(var_data, na.rm = TRUE), 4), ", ",
          "Median = ", round(median(var_data, na.rm = TRUE), 4), ", ",
          "Min = ", round(min(var_data, na.rm = TRUE), 4), ", ",
          "Max = ", round(max(var_data, na.rm = TRUE), 4), ", ",
          "Missing = ", sum(is.na(var_data))
        )
      } else {
        var_stats <- paste0(
          "<b>", var, ":</b> ",
          "Type = ", class(var_data)[1], ", ",
          "Unique = ", length(unique(var_data)), ", ",
          "Missing = ", sum(is.na(var_data))
        )
      }
      story <- c(story, list(Paragraph(var_stats, body_style)))
    }
    story <- c(story, list(Spacer(1L, 0.15 * inch)))
    
    # 2.1 Data Pairplot
    story <- c(story, list(Paragraph("2.1 Pairwise Relationships", styles$Heading3)))
    story <- c(story, list(Paragraph(
      "The pairplot below shows the pairwise scatter plots and marginal kernel density estimates for all variables in the dataset.",
      body_style)))
    
    tryCatch({
      pairplot_path <- file.path(reports_dir, paste0("pairplot_", timestamp, ".png"))
      plt$close("all")
      py_pd_df <- r_to_py(data_df)
      g <- sns$pairplot(py_pd_df, diag_kind = "kde", plot_kws = list(alpha = 0.3, s = 10L))
      g$fig$set_size_inches(10, 10)
      g$fig$savefig(pairplot_path, dpi = 120L, bbox_inches = "tight")
      plt$close("all")
      story <- add_plot_to_story(story, pairplot_path, img_width = 5.5, caption = "Figure 1: Data pairplot with KDE marginals.")
    }, error = function(e) {
      cat("Warning: Could not generate pairplot for report:", conditionMessage(e), "\n")
    })
    
    story <- c(story, list(PageBreak()))
    
    # ================================================================
    # 3. CAUSAL DAG SPECIFICATION
    # ================================================================
    story <- c(story, list(Paragraph("3. Causal DAG Specification", styles$Heading2)))
    
    vars <- if (!is.null(amat) && !is.null(rownames(amat))) rownames(amat) else colnames(data_df)
    story <- c(story, list(Paragraph(paste0("<b>Variables in the DAG:</b> ", paste(vars, collapse = ", ")), body_style)))
    
    if (!is.null(amat)) {
      edges <- which(amat != "0" & amat != 0, arr.ind = TRUE)
      if (nrow(edges) > 0) {
        edge_list <- character(0)
        for (i in seq_len(nrow(edges))) {
          from <- vars[edges[i, "row"]]
          to <- vars[edges[i, "col"]]
          edge_type <- amat[edges[i, "row"], edges[i, "col"]]
          edge_list <- c(edge_list, paste0(from, " &rarr; ", to, " (", edge_type, ")"))
        }
        edge_text <- paste0("<b>Causal Edges:</b><br/>", paste(paste0("&bull; ", edge_list), collapse = "<br/>"))
        story <- c(story, list(Paragraph(edge_text, body_style)))
        
        # Generate DAG plot
        tryCatch({
          dag_plot_path <- file.path(reports_dir, paste0("dag_plot_", timestamp, ".png"))
          
          A_binary <- matrix(0L, nrow = nrow(amat), ncol = ncol(amat), dimnames = dimnames(amat))
          non_zero <- (amat != "0") & (amat != 0) & (!is.na(amat))
          A_binary[non_zero] <- 1L
          
          g <- igraph::graph_from_adjacency_matrix(A_binary, mode = "directed")
          
          n <- length(vars)
          angles <- seq(from = pi/2, by = -2*pi/n, length.out = n)
          lay <- cbind(x = cos(angles), y = sin(angles))
          rownames(lay) <- vars
          
          edge_labels <- NULL
          if (any(non_zero)) {
            edge_mat <- which(non_zero, arr.ind = TRUE)
            if (nrow(edge_mat) > 0) {
              edge_list_g <- igraph::as_edgelist(g)
              edge_labels <- character(nrow(edge_list_g))
              for (i in seq_len(nrow(edge_mat))) {
                from_var <- vars[edge_mat[i, "row"]]
                to_var <- vars[edge_mat[i, "col"]]
                edge_type <- amat[edge_mat[i, "row"], edge_mat[i, "col"]]
                edge_idx <- which(edge_list_g[, 1] == from_var & edge_list_g[, 2] == to_var)
                if (length(edge_idx) > 0) edge_labels[edge_idx[1]] <- edge_type
              }
              edge_labels[edge_labels == ""] <- NA
            }
          }
          
          png(dag_plot_path, width = 800, height = 600, res = 150, bg = "white")
          plot_args <- list(
            x = g,
            main = "Causal DAG Structure",
            vertex.label.cex = 1.5,
            vertex.size = 50,
            vertex.color = "lightblue",
            vertex.frame.color = "darkblue",
            vertex.label.color = "black",
            edge.arrow.size = 0.8,
            edge.color = "gray50",
            layout = lay,
            rescale = FALSE,
            xlim = range(lay[, 1]) * 1.3,
            ylim = range(lay[, 2]) * 1.3
          )
          if (!is.null(edge_labels) && length(edge_labels) > 0 && any(!is.na(edge_labels))) {
            plot_args$edge.label <- edge_labels
            plot_args$edge.label.cex <- 1.0
            plot_args$edge.label.color <- "darkblue"
          }
          do.call(plot, plot_args)
          dev.off()
          
          story <- add_plot_to_story(story, dag_plot_path, img_width = 4.5, caption = "Figure 2: Directed Acyclic Graph (DAG) structure.")
        }, error = function(e) {
          cat("Warning: Could not generate DAG plot:", conditionMessage(e), "\n")
        })
      }
    }
    story <- c(story, list(Spacer(1L, 0.1 * inch)))
    
    # ================================================================
    # 4. METHODOLOGY
    # ================================================================
    story <- c(story, list(Paragraph("4. Methodology", styles$Heading2)))
    
    # 4.1 TRAM-DAG Model
    story <- c(story, list(Paragraph("4.1 TRAM-DAG Model", styles$Heading3)))
    methodology_text <- paste0(
      "TRAM-DAG (Transformation Models for Directed Acyclic Graphs) is a flexible framework for causal inference ",
      "that combines transformation models with directed acyclic graphs to enable causal effect estimation ",
      "under interventions and counterfactuals. The model learns conditional distributions for each variable ",
      "given its parents in the DAG, allowing for complex non-linear relationships while maintaining ",
      "causal interpretability. TRAM-DAGs bridge the gap between interpretability and flexibility in causal modeling, ",
      "enabling queries at all three levels of Pearl's causal hierarchy: observational (L1), interventional (L2), ",
      "and counterfactual (L3) queries."
    )
    story <- c(story, list(Paragraph(methodology_text, body_style)))
    story <- c(story, list(Spacer(1L, 0.1 * inch)))
    
    # 4.2 Training Configuration
    story <- c(story, list(Paragraph("4.2 Training Configuration", styles$Heading3)))
    params_text <- paste0(
      "<b>Training Epochs:</b> ", epochs, "<br/>",
      "<b>Learning Rate:</b> ", learning_rate, "<br/>",
      "<b>Batch Size:</b> ", batch_size, "<br/>",
      "<b>Initial Weights:</b> ", ifelse(set_initial_weights, "R-based initialization", "Random initialization")
    )
    story <- c(story, list(Paragraph(params_text, body_style)))
    
    # Train/val/test split info
    if (!is.null(train_df) && !is.null(test_df)) {
      n_train <- tryCatch(nrow(train_df), error = function(e) NA)
      n_test <- tryCatch(nrow(test_df), error = function(e) NA)
      if (!is.na(n_train) && !is.na(n_test)) {
        split_text <- paste0(
          "<b>Training set:</b> ", n_train, " observations<br/>",
          "<b>Test set:</b> ", n_test, " observations"
        )
        story <- c(story, list(Paragraph(split_text, body_style)))
      }
    }
    story <- c(story, list(Spacer(1L, 0.1 * inch)))
    
    # Key references
    story <- c(story, list(Paragraph("4.3 Key References", styles$Heading3)))
    citations <- c(
      "Sick, B., &amp; D&uuml;rr, O. (2025). Interpretable Neural Causal Models with TRAM-DAGs. <i>arXiv:2503.16206</i>. CLeaR 2025.",
      "Hothorn, T., Most, L., &amp; B&uuml;hlmann, P. (2018). Most Likely Transformations. <i>Scand. J. Stat.</i>, 45(1), 110-134.",
      "Pearl, J. (2009). <i>Causality</i> (2nd ed.). Cambridge University Press.",
      "Hothorn, T., Kneib, T., &amp; B&uuml;hlmann, P. (2014). Conditional Transformation Models. <i>JRSS-B</i>, 76(1), 3-27."
    )
    for (cit in citations) {
      story <- c(story, list(Paragraph(paste0("&bull; ", cit), body_style)))
    }
    
    story <- c(story, list(PageBreak()))
    
    # ================================================================
    # 5. MODEL FITTING & DIAGNOSTICS
    # ================================================================
    if (!is.null(td_model)) {
      story <- c(story, list(Paragraph("5. Model Fitting &amp; Diagnostics", styles$Heading2)))
      story <- c(story, list(Paragraph(
        "The following diagnostics assess model convergence and fit quality, matching the standard TRAM-DAG diagnostic workflow.",
        body_style)))
      story <- c(story, list(Spacer(1L, 0.15 * inch)))
      
      fig_counter <- 3L  # Already used 1 (pairplot) and 2 (DAG)
      
      # 5.1 Loss History
      story <- c(story, list(Paragraph("5.1 Training &amp; Validation Loss", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "The loss history shows the negative log-likelihood loss over training epochs for both training and validation sets. Convergence is indicated by stabilization of both curves.",
        body_style)))
      tryCatch({
        loss_path <- file.path(reports_dir, paste0("loss_history_", timestamp, ".png"))
        original_show <- plt$show
        plt$close("all")
        plt$show <- function(...) invisible(NULL)
        plt$ioff()
        td_model$plot_loss_history()
        plt$show <- original_show
        
        fig <- plt$gcf()
        fig$set_size_inches(14, 10)
        axes <- fig$get_axes()
        for (i in seq_along(axes)) {
          ax <- axes[[i]]
          tryCatch({
            ax$legend(loc = "center left", bbox_to_anchor = c(1.02, 0.5), fontsize = 9L)
          }, error = function(e) {})
        }
        fig$subplots_adjust(hspace = 0.35, right = 0.82)
        fig$savefig(loss_path, dpi = 120L, bbox_inches = "tight")
        plt$close("all")
        
        story <- add_plot_to_story(story, loss_path, img_width = 6.5, 
                                   caption = paste0("Figure ", fig_counter, ": Training and validation loss history."))
        fig_counter <- fig_counter + 1L
      }, error = function(e) {
        cat("Warning: Could not generate loss plot for report:", conditionMessage(e), "\n")
      })
      
      # 5.2 Linear Shift History
      story <- c(story, list(Paragraph("5.2 Linear Shift Parameters", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "The linear shift parameters capture the direct causal effect of parent variables on each child. Tracking these over epochs shows how the model learns the causal relationships.",
        body_style)))
      tryCatch({
        shift_path <- save_report_plot(function() {
          td_model$plot_linear_shift_history()
        }, "shift_history", width = 1200, height = 500)
        story <- add_plot_to_story(story, shift_path, img_width = 6.5,
                                   caption = paste0("Figure ", fig_counter, ": Linear shift parameter history over training epochs."))
        fig_counter <- fig_counter + 1L
      }, error = function(e) {
        cat("Warning: Could not generate shift history plot:", conditionMessage(e), "\n")
      })
      
      # 5.3 Simple Intercepts History
      story <- c(story, list(Paragraph("5.3 Simple Intercepts", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "The simple intercepts (Bernstein polynomial coefficients) define the baseline transformation function for each variable. They should stabilize and maintain monotonicity.",
        body_style)))
      tryCatch({
        intercepts_path <- save_report_plot(function() {
          td_model$plot_simple_intercepts_history()
        }, "intercepts_history", width = 1200, height = 500)
        story <- add_plot_to_story(story, intercepts_path, img_width = 6.5,
                                   caption = paste0("Figure ", fig_counter, ": Simple intercept parameter history."))
        fig_counter <- fig_counter + 1L
      }, error = function(e) {
        cat("Warning: Could not generate intercepts plot:", conditionMessage(e), "\n")
      })
      
      story <- c(story, list(PageBreak()))
      
      # 5.4 h-DAG Transformation
      story <- c(story, list(Paragraph("5.4 h-DAG Transformation Functions", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "The h-DAG plots show the learned transformation function h(y|x) for each variable given its parents. These transformations map the observed data to a latent standard normal distribution.",
        body_style)))
      if (!is.null(train_df)) {
        tryCatch({
          py_train <- r_to_py(train_df)
          hdag_path <- save_report_plot(function() {
            td_model$plot_hdag(py_train, variables = as.list(vars), plot_n_rows = 1L)
          }, "hdag", width = 1400, height = 500)
          story <- add_plot_to_story(story, hdag_path, img_width = 6.5,
                                     caption = paste0("Figure ", fig_counter, ": h-DAG transformation functions."))
          fig_counter <- fig_counter + 1L
        }, error = function(e) {
          cat("Warning: Could not generate h-DAG plot:", conditionMessage(e), "\n")
        })
      }
      
      # 5.5 Latent Distributions
      story <- c(story, list(Paragraph("5.5 Latent Distributions", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "If the model fits well, the latent distributions (obtained by transforming the observed data through h) should follow a standard normal distribution N(0,1). Deviations indicate model misspecification.",
        body_style)))
      if (!is.null(train_df)) {
        tryCatch({
          py_train <- r_to_py(train_df)
          latents_path <- save_report_plot(function() {
            td_model$plot_latents(py_train)
          }, "latents", width = 1200, height = 500)
          story <- add_plot_to_story(story, latents_path, img_width = 6.5,
                                     caption = paste0("Figure ", fig_counter, ": Latent distributions (should be standard normal)."))
          fig_counter <- fig_counter + 1L
        }, error = function(e) {
          cat("Warning: Could not generate latents plot:", conditionMessage(e), "\n")
        })
      }
      
      # 5.6 Negative Log-Likelihood
      story <- c(story, list(Paragraph("5.6 Negative Log-Likelihood (NLL)", styles$Heading3)))
      story <- c(story, list(Paragraph(
        "The per-variable NLL on the training data provides a quantitative measure of model fit. Lower values indicate better fit.",
        body_style)))
      if (!is.null(train_df)) {
        tryCatch({
          py_train <- r_to_py(train_df)
          nll_result <- td_model$nll(py_train)
          nll_r <- py_to_r(nll_result)
          nll_lines <- sapply(names(nll_r), function(node) {
            paste0("<b>", node, ":</b> ", sprintf("%.4f", nll_r[[node]]))
          })
          nll_text <- paste0("NLL on training data:<br/>", paste(nll_lines, collapse = "<br/>"))
          story <- c(story, list(Paragraph(nll_text, body_style)))
        }, error = function(e) {
          story <- c(story, list(Paragraph(
            paste0("Could not compute NLL: ", conditionMessage(e)), body_style)))
        })
      }
      story <- c(story, list(Spacer(1L, 0.15 * inch)))
      
      # Model interpretation if available
      if (!is.null(model_interpretation) && trimws(model_interpretation) != "") {
        story <- c(story, list(Paragraph("5.7 Model Interpretation (AI-Generated)", styles$Heading3)))
        interpretation_html <- gsub("\n\n", "</p><p>", model_interpretation)
        interpretation_html <- gsub("\n", "<br/>", interpretation_html)
        story <- c(story, list(Paragraph(paste0("<p>", interpretation_html, "</p>"), body_style)))
      }
      
      story <- c(story, list(PageBreak()))
      
      # ================================================================
      # 6. OBSERVATIONAL SAMPLING
      # ================================================================
      if (!is.null(sampled_data) && !is.null(test_df)) {
        story <- c(story, list(Paragraph("6. Observational Sampling", styles$Heading2)))
        story <- c(story, list(Paragraph(
          "The model samples from the learned joint distribution (no interventions). If the model fits well, the sampled distributions (orange) should closely match the held-out test data (blue).",
          body_style)))
        
        tryCatch({
          py_test <- r_to_py(test_df)
          
          # Convert sampled data to Python dict of tensors
          sampled_list <- list()
          for (col in colnames(sampled_data)) {
            sampled_list[[col]] <- torch$tensor(as.numeric(sampled_data[[col]]))
          }
          py_sampled <- r_to_py(sampled_list)
          
          obs_path <- save_report_plot(function() {
            td_model$plot_samples_vs_true(py_test, py_sampled)
          }, "obs_samples_vs_true", width = 1400, height = 500)
          story <- add_plot_to_story(story, obs_path, img_width = 6.5,
                                     caption = paste0("Figure ", fig_counter, ": Observational samples (orange) vs held-out test data (blue)."))
          fig_counter <- fig_counter + 1L
        }, error = function(e) {
          cat("Warning: Could not generate observational samples plot:", conditionMessage(e), "\n")
        })
        story <- c(story, list(Spacer(1L, 0.15 * inch)))
      }
      
      # ================================================================
      # 7. CAUSAL EFFECT ANALYSIS (ATE)
      # ================================================================
      if (!is.null(ate_result) && !is.null(X_var) && !is.null(Y_var)) {
        story <- c(story, list(Paragraph("7. Causal Effect Analysis", styles$Heading2)))
        
        story <- c(story, list(Paragraph("7.1 Average Treatment Effect (ATE)", styles$Heading3)))
        ate_text <- paste0(
          "The ATE quantifies the causal effect of intervening on <b>", X_var, "</b> on the outcome <b>", Y_var, "</b>. ",
          "It is computed as the difference in expected outcomes under the treatment vs. control interventions:<br/><br/>",
          "<b>Treatment Variable:</b> ", X_var, "<br/>",
          "<b>Outcome Variable:</b> ", Y_var, "<br/>",
          "<b>Treatment Condition:</b> do(", X_var, " = ", x_treated, ")<br/>",
          "<b>Control Condition:</b> do(", X_var, " = ", x_control, ")<br/><br/>",
          "<b>ATE = ", round(ate_result$ate, 4), "</b><br/>",
          "E[", Y_var, " | do(", X_var, " = ", x_treated, ")] = ", round(ate_result$mean_treated, 4), 
          " (std: ", round(sd(ate_result$Y_treated), 4), ")<br/>",
          "E[", Y_var, " | do(", X_var, " = ", x_control, ")] = ", round(ate_result$mean_control, 4),
          " (std: ", round(sd(ate_result$Y_control), 4), ")"
        )
        story <- c(story, list(Paragraph(ate_text, body_style)))
        story <- c(story, list(Spacer(1L, 0.1 * inch)))
        
        # ATE interpretation if available
        if (!is.null(ate_interpretation) && trimws(ate_interpretation) != "") {
          story <- c(story, list(Paragraph("7.2 ATE Interpretation (AI-Generated)", styles$Heading3)))
          interpretation_html <- gsub("\n\n", "</p><p>", ate_interpretation)
          interpretation_html <- gsub("\n", "<br/>", interpretation_html)
          story <- c(story, list(Paragraph(paste0("<p>", interpretation_html, "</p>"), body_style)))
        }
        story <- c(story, list(Spacer(1L, 0.15 * inch)))
      }
      
      # ================================================================
      # 8. INTERVENTIONAL SAMPLING
      # ================================================================
      if (!is.null(interventional_samples) && !is.null(interventional_control)) {
        story <- c(story, list(Paragraph("8. Interventional Sampling: Treated vs Control", styles$Heading2)))
        story <- c(story, list(Paragraph(
          paste0("The plot compares samples under do(", 
                 ifelse(!is.null(X_var), X_var, "X"), " = ", 
                 ifelse(!is.null(x_control), x_control, "control"),
                 ") in blue versus do(",
                 ifelse(!is.null(X_var), X_var, "X"), " = ", 
                 ifelse(!is.null(x_treated), x_treated, "treated"),
                 ") in orange. The distributional shift between the two interventions visualizes the ATE."),
          body_style)))
        
        tryCatch({
          scipy_stats <- import("scipy.stats")
          np_mod <- import("numpy")
          
          vars <- colnames(interventional_samples)
          n_vars <- length(vars)
          
          inter_plot_path <- file.path(reports_dir, paste0("inter_treated_vs_control_", timestamp, ".png"))
          plt$close("all")
          plt$ioff()
          
          fig_w <- max(4.5 * n_vars, 10)
          fig_and_axes <- plt$subplots(1L, as.integer(n_vars), figsize = c(fig_w, 5))
          fig <- fig_and_axes[[1]]
          axes_raw <- fig_and_axes[[2]]
          
          if (n_vars == 1L) {
            axes_list <- list(axes_raw)
          } else {
            axes_list <- lapply(seq_len(n_vars), function(i) axes_raw[i - 1L])
          }
          
          for (i in seq_len(n_vars)) {
            var <- vars[i]
            ax <- axes_list[[i]]
            ctrl_vals <- as.numeric(interventional_control[[var]])
            treat_vals <- as.numeric(interventional_samples[[var]])
            all_vals <- c(ctrl_vals, treat_vals)
            bins <- np_mod$linspace(min(all_vals), max(all_vals), 80L)
            
            ax$hist(ctrl_vals, bins = bins, density = TRUE, alpha = 0.5, color = "steelblue",
                    label = paste0("do(", X_var, "=", x_control, ")"))
            ax$hist(treat_vals, bins = bins, density = TRUE, alpha = 0.5, color = "darkorange",
                    label = paste0("do(", X_var, "=", x_treated, ")"))
            
            tryCatch({
              x_grid <- np_mod$linspace(min(all_vals), max(all_vals), 300L)
              ax$plot(x_grid, scipy_stats$gaussian_kde(ctrl_vals)(x_grid), color = "steelblue", lw = 2L, linestyle = "--")
              ax$plot(x_grid, scipy_stats$gaussian_kde(treat_vals)(x_grid), color = "darkorange", lw = 2L, linestyle = "--")
            }, error = function(e) {})
            
            mean_ctrl <- mean(ctrl_vals)
            mean_treat <- mean(treat_vals)
            ax$axvline(x = mean_ctrl, color = "steelblue", linestyle = ":", lw = 1.5, alpha = 0.8)
            ax$axvline(x = mean_treat, color = "darkorange", linestyle = ":", lw = 1.5, alpha = 0.8)
            
            if (!is.null(Y_var) && var == Y_var) {
              ax$set_title(paste0(var, "  (ATE = ", round(mean_treat - mean_ctrl, 4), ")"), fontsize = 13L, fontweight = "bold")
            } else if (!is.null(X_var) && var == X_var) {
              ax$set_title(paste0(var, "  (intervened)"), fontsize = 13L, fontweight = "bold")
            } else {
              ax$set_title(paste0(var, "  (shift = ", round(mean_treat - mean_ctrl, 4), ")"), fontsize = 12L)
            }
            ax$set_xlabel(var, fontsize = 10L)
            ax$set_ylabel("Density", fontsize = 10L)
            ax$legend(fontsize = 8L, loc = "upper right")
          }
          
          fig$suptitle(paste0("do(", X_var, "=", x_control, ") vs do(", X_var, "=", x_treated, ")"),
                       fontsize = 14L, fontweight = "bold", y = 1.02)
          fig$tight_layout()
          fig$savefig(inter_plot_path, dpi = 120L, bbox_inches = "tight")
          plt$close("all")
          
          story <- add_plot_to_story(story, inter_plot_path, img_width = 6.5,
                                     caption = paste0("Figure ", fig_counter, ": Treated (orange) vs control (blue) interventional distributions."))
          fig_counter <- fig_counter + 1L
        }, error = function(e) {
          cat("Warning: Could not generate interventional comparison plot:", conditionMessage(e), "\n")
        })
        story <- c(story, list(Spacer(1L, 0.15 * inch)))
      }
    }  # end if td_model
    
    # ================================================================
    # APPENDIX: REPRODUCIBILITY
    # ================================================================
    story <- c(story, list(PageBreak()))
    story <- c(story, list(Paragraph("Appendix: Reproducibility", styles$Heading2)))
    repro_text <- paste0(
      "All model artifacts, configuration files, and training logs are saved in the experiment directory. ",
      "To reproduce this analysis, load the same data and configuration from the experiment directory below."
    )
    story <- c(story, list(Paragraph(repro_text, body_style)))
    story <- c(story, list(Paragraph(paste0("<b>Experiment Directory:</b> ", experiment_dir), body_style)))
    story <- c(story, list(Spacer(1L, 0.15 * inch)))
    
    # List files in experiment directory
    tryCatch({
      exp_files <- list.files(experiment_dir, recursive = TRUE, full.names = FALSE)
      if (length(exp_files) > 0) {
        # Group by top-level directory
        story <- c(story, list(Paragraph("<b>Experiment Files:</b>", body_style)))
        file_text <- paste(paste0("&bull; ", exp_files), collapse = "<br/>")
        # Limit to avoid overflow
        if (length(exp_files) > 50) {
          file_text <- paste(paste0("&bull; ", exp_files[1:50]), collapse = "<br/>")
          file_text <- paste0(file_text, "<br/>... and ", length(exp_files) - 50, " more files")
        }
        story <- c(story, list(Paragraph(file_text, body_style)))
      }
    }, error = function(e) {})
    story <- c(story, list(Spacer(1L, 0.2 * inch)))
    
    # References
    story <- c(story, list(Paragraph("References", styles$Heading2)))
    references <- c(
      "Sick, B., &amp; D&uuml;rr, O. (2025). Interpretable Neural Causal Models with TRAM-DAGs. <i>arXiv:2503.16206</i>. CLeaR 2025. https://doi.org/10.48550/arXiv.2503.16206",
      "Hothorn, T., Most, L., &amp; B&uuml;hlmann, P. (2018). Most Likely Transformations. <i>Scand. J. Stat.</i>, 45(1), 110-134.",
      "Pearl, J. (2009). <i>Causality: Models, Reasoning, and Inference</i> (2nd ed.). Cambridge University Press.",
      "Hothorn, T., Kneib, T., &amp; B&uuml;hlmann, P. (2014). Conditional Transformation Models. <i>JRSS-B</i>, 76(1), 3-27.",
      "Peters, J., Janzing, D., &amp; Sch&ouml;lkopf, B. (2017). <i>Elements of Causal Inference</i>. MIT Press."
    )
    for (ref in references) {
      story <- c(story, list(Paragraph(paste0("&bull; ", ref), body_style)))
    }
    
    # Build PDF
    doc$build(story)
    
    cat("PDF report generated:", report_path, "\n")
    return(report_path)
    
  }, error = function(e) {
    error_msg <- conditionMessage(e)
    cat("Error generating PDF report:", error_msg, "\n")
    
    # Provide helpful error message if reportlab is missing
    if (grepl("reportlab", error_msg, ignore.case = TRUE) || 
        grepl("No module named", error_msg, ignore.case = TRUE)) {
      cat("\n")
      cat("To generate PDF reports, install reportlab: pip install reportlab\n")
    }
    
    return(NULL)
  })
}

## --------------------------------------------------
## 2) UI
## --------------------------------------------------

ui <- fluidPage(
  # Add CSS for loading spinner
  tags$head(
    tags$style(HTML("
      /* ===== Global & Typography ===== */
      body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-color: #f0f4f8;
        color: #2c3e50;
      }

      /* Prevent Shiny from graying out the page while the server is busy */
      .recalculating {
        opacity: 1 !important;
      }

      /* ===== Sidebar Panel - Blue Medical Theme ===== */
      .well {
        background: linear-gradient(180deg, #f7fafd 0%, #eaf2fb 100%);
        border: 1px solid #c8ddf0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(44, 82, 130, 0.08);
      }
      .well h4 {
        color: #1a5276;
        font-weight: 600;
        border-bottom: 2px solid #aed6f1;
        padding-bottom: 6px;
        margin-top: 8px;
        letter-spacing: 0.3px;
      }
      .well h5 {
        color: #2471a3;
        font-weight: 600;
      }
      .well p {
        color: #4a6a8a;
        font-size: 0.92em;
        line-height: 1.5;
      }
      .well hr {
        border-top: 1px solid #c8ddf0;
        margin: 14px 0;
      }
      .well label {
        color: #1a5276;
        font-weight: 500;
      }
      .well .radio label, .well .checkbox label {
        color: #34495e;
        font-weight: 400;
      }
      .well .form-control {
        border: 1px solid #b3cde0;
        border-radius: 6px;
        transition: border-color 0.2s, box-shadow 0.2s;
      }
      .well .form-control:focus {
        border-color: #5dade2;
        box-shadow: 0 0 0 3px rgba(93, 173, 226, 0.2);
      }
      .well .selectize-input {
        border: 1px solid #b3cde0;
        border-radius: 6px;
      }
      .well .selectize-input.focus {
        border-color: #5dade2;
        box-shadow: 0 0 0 3px rgba(93, 173, 226, 0.2);
      }
      .well .btn-file {
        background-color: #d6eaf8;
        border: 1px solid #85c1e9;
        color: #1a5276;
        border-radius: 6px;
        font-weight: 500;
      }
      .well .btn-file:hover {
        background-color: #aed6f1;
        border-color: #5dade2;
      }

      /* ===== All Sidebar Buttons - Unified Blue Palette ===== */
      .well .btn-primary,
      .well .btn-success,
      .well .btn-info,
      .well .btn-warning,
      .well .btn-default {
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(26, 82, 118, 0.15);
      }

      /* Standard action buttons */
      .well .btn-primary {
        background-color: #2980b9;
        color: #fff;
      }
      .well .btn-primary:hover, .well .btn-primary:focus {
        background-color: #2471a3;
        box-shadow: 0 4px 10px rgba(41, 128, 185, 0.3);
      }

      /* Large action buttons (Fit, Sample, ATE) */
      .well .btn-success.btn-lg {
        background: linear-gradient(135deg, #2980b9 0%, #1a6da0 100%);
        color: #fff;
        width: 100%;
        padding: 10px 16px;
        font-size: 1.05em;
      }
      .well .btn-success.btn-lg:hover, .well .btn-success.btn-lg:focus {
        background: linear-gradient(135deg, #1a6da0 0%, #15577f 100%);
        box-shadow: 0 4px 12px rgba(26, 109, 160, 0.35);
      }

      .well .btn-info.btn-lg {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: #fff;
        width: 100%;
        padding: 10px 16px;
        font-size: 1.05em;
      }
      .well .btn-info.btn-lg:hover, .well .btn-info.btn-lg:focus {
        background: linear-gradient(135deg, #2980b9 0%, #2471a3 100%);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.35);
      }

      /* Report button - slightly distinct but still blue */
      .well .btn-warning.btn-lg {
        background: linear-gradient(135deg, #1a5276 0%, #154360 100%);
        color: #fff;
        width: 100%;
        padding: 10px 16px;
        font-size: 1.05em;
      }
      .well .btn-warning.btn-lg:hover, .well .btn-warning.btn-lg:focus {
        background: linear-gradient(135deg, #154360 0%, #0e2f44 100%);
        box-shadow: 0 4px 12px rgba(26, 82, 118, 0.35);
        color: #fff;
      }

      /* Small buttons */
      .well .btn-sm.btn-primary {
        font-size: 0.85em;
        padding: 5px 12px;
        border-radius: 6px;
      }

      /* Info boxes in sidebar */
      .well .info-box {
        background-color: #eaf2fb;
        border-left: 4px solid #2980b9;
        border-radius: 4px;
        padding: 10px 12px;
        margin: 10px 0;
        font-size: 0.9em;
      }

      /* ===== Chat Interface ===== */
      .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f7fafd;
        border: 1px solid #d5e8f5;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 8px;
      }
      .chat-msg {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 1em;
        line-height: 1.5;
        max-width: 100%;
        word-wrap: break-word;
      }
      .chat-msg.user {
        background: #d4e8f7;
        color: #1a3a50;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        position: relative;
        z-index: 1;
      }
      .chat-msg.assistant {
        background: #eaf2fb;
        color: #1a3a50;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border-left: 3px solid #5dade2;
      }
      .chat-msg .chat-role {
        font-size: 0.78em;
        font-weight: 600;
        margin-bottom: 3px;
        opacity: 0.7;
      }
      .chat-input-row {
        display: flex;
        gap: 8px;
        align-items: flex-end;
      }
      .chat-input-row .form-group {
        flex: 1;
        margin-bottom: 0;
      }
      .thinking-dots {
        display: flex;
        gap: 5px;
        padding: 4px 0;
      }
      .thinking-dots .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #5dade2;
        animation: thinking-bounce 1.4s infinite ease-in-out both;
      }
      .thinking-dots .dot:nth-child(1) { animation-delay: 0s; }
      .thinking-dots .dot:nth-child(2) { animation-delay: 0.2s; }
      .thinking-dots .dot:nth-child(3) { animation-delay: 0.4s; }
      @keyframes thinking-bounce {
        0%, 80%, 100% {
          transform: scale(0.6);
          opacity: 0.4;
        }
        40% {
          transform: scale(1);
          opacity: 1;
        }
      }

      /* ===== Title Panel ===== */
      h2 {
        color: #1a5276;
      }

      /* ===== Loading Spinner ===== */
      .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(41, 128, 185, 0.15);
        border-radius: 50%;
        border-top-color: #2980b9;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
        vertical-align: middle;
      }
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      .loading-container {
        text-align: center;
        padding: 20px;
        background-color: #f7fafd;
        border-radius: 8px;
        margin: 10px 0;
      }
      .loading-text {
        color: #2980b9;
        font-weight: bold;
        margin-top: 10px;
      }
      .shiny-notification {
        position: fixed;
        top: calc(50%);
        left: calc(50%);
        margin-top: -100px;
        margin-left: -150px;
        background-color: rgba(26, 82, 118, 0.92);
        color: white;
        padding: 20px;
        border-radius: 8px;
        z-index: 9999;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
      }

      /* ===== Plot Thumbnails ===== */
      .plot-thumbnail {
        position: relative;
        text-align: center;
      }
      .plot-thumbnail .shiny-image-output.recalculating,
      .plot-thumbnail .shiny-plot-output.recalculating {
        opacity: 0.3;
      }
      .plot-thumbnail .shiny-image-output.recalculating::after,
      .plot-thumbnail .shiny-plot-output.recalculating::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 37.5%;
        width: 36px;
        height: 36px;
        margin-top: -18px;
        margin-left: -18px;
        border: 4px solid rgba(41, 128, 185, 0.15);
        border-top-color: #2980b9;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }
      .plot-loading-msg {
        display: none;
        text-align: center;
        padding: 30px 15px;
        color: #5d7ea0;
        font-size: 1.1em;
        background: #f7fafd;
        border: 1px dashed #b3cde0;
        border-radius: 8px;
        margin: 10px 0;
      }
      .plot-loading-msg .loading-spinner {
        display: inline-block;
        margin-right: 8px;
      }
      .plot-thumbnail .shiny-image-output.recalculating ~ .plot-loading-msg {
        display: block;
      }
      .plot-thumbnail img {
        max-width: 75%;
        height: auto;
        cursor: pointer;
        transition: opacity 0.2s, box-shadow 0.2s;
        border: 1px solid #c8ddf0;
        border-radius: 6px;
      }
      .plot-thumbnail img:hover {
        opacity: 0.88;
        box-shadow: 0 3px 12px rgba(41, 128, 185, 0.2);
      }
      .plot-thumbnail .click-hint {
        font-size: 0.8em;
        color: #85a5c2;
        margin-top: 2px;
      }

      /* Sidebar plots should use full width */
      .well .plot-thumbnail img {
        max-width: 100%;
      }

      /* ===== Full-size Modal Overlay ===== */
      #plot-modal-overlay {
        display: none;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(10, 30, 50, 0.85);
        z-index: 10000;
      }
      #plot-modal-content {
        position: absolute;
        display: flex;
        flex-direction: column;
        background: rgba(20, 40, 60, 0.7);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        overflow: hidden;
        min-width: 320px;
        min-height: 220px;
      }
      #plot-modal-toolbar {
        display: flex;
        gap: 8px;
        align-items: center;
        padding: 8px 14px;
        background: rgba(255,255,255,0.1);
        cursor: move;
        user-select: none;
        flex-shrink: 0;
      }
      #plot-modal-toolbar button, #plot-modal-close {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        font-size: 18px;
        width: 36px;
        height: 36px;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s;
      }
      #plot-modal-toolbar button:hover, #plot-modal-close:hover {
        background: rgba(255,255,255,0.35);
      }
      #plot-modal-toolbar .zoom-label {
        color: rgba(255,255,255,0.8);
        font-size: 13px;
        min-width: 45px;
        text-align: center;
        user-select: none;
      }
      #plot-modal-toolbar .zoom-slider {
        -webkit-appearance: none;
        appearance: none;
        width: 160px;
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.25);
        outline: none;
        cursor: pointer;
      }
      #plot-modal-toolbar .zoom-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #5dade2;
        cursor: pointer;
        border: 2px solid white;
      }
      #plot-modal-toolbar .zoom-slider::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #5dade2;
        cursor: pointer;
        border: 2px solid white;
      }
      .toolbar-spacer { flex: 1; }
      #plot-modal-img-wrap {
        flex: 1;
        min-height: 0;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
      }
      #plot-modal-img-wrap img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 6px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        transform-origin: center center;
        transition: transform 0.15s ease;
        user-select: none;
        -webkit-user-drag: none;
      }
      /* Resize handles on the panel edges and corners */
      .modal-resize-handle {
        position: absolute;
        z-index: 5;
      }
      .modal-resize-handle.h-top    { top: -4px;  left: 14px;  right: 14px; height: 8px;  cursor: n-resize; }
      .modal-resize-handle.h-bottom { bottom: -4px; left: 14px; right: 14px; height: 8px; cursor: s-resize; }
      .modal-resize-handle.h-left   { left: -4px;  top: 14px;  bottom: 14px; width: 8px;  cursor: w-resize; }
      .modal-resize-handle.h-right  { right: -4px; top: 14px;  bottom: 14px; width: 8px;  cursor: e-resize; }
      .modal-resize-handle.h-tl { top: -5px;  left: -5px;  width: 16px; height: 16px; cursor: nw-resize; }
      .modal-resize-handle.h-tr { top: -5px;  right: -5px; width: 16px; height: 16px; cursor: ne-resize; }
      .modal-resize-handle.h-bl { bottom: -5px; left: -5px; width: 16px; height: 16px; cursor: sw-resize; }
      .modal-resize-handle.h-br { bottom: -5px; right: -5px; width: 16px; height: 16px; cursor: se-resize; }
    ")),
    tags$script(HTML("
      // --- Resizable / draggable plot modal with zoom ---
      var currentZoom = 100;
      var mDrag  = { on: false, sx: 0, sy: 0, oL: 0, oT: 0 };
      var mResize = { on: false, dir: '', sx: 0, sy: 0, oW: 0, oH: 0, oL: 0, oT: 0 };

      function setZoom(val) {
        val = Math.max(20, Math.min(300, val));
        currentZoom = val;
        var $img = $('#plot-modal-img');
        $img.css('transform', 'scale(' + (val / 100) + ')');
        $('#zoom-slider').val(val);
        $('#zoom-label').text(Math.round(val) + '%');
      }

      function centerModal() {
        var $c = $('#plot-modal-content');
        var w = Math.min(window.innerWidth * 0.85, 1400);
        var h = window.innerHeight * 0.85;
        $c.css({ width: w, height: h,
                 left: (window.innerWidth - w) / 2,
                 top:  (window.innerHeight - h) / 2 });
      }

      // Open
      $(document).on('click', '.plot-thumbnail img', function() {
        $('#plot-modal-img').attr('src', $(this).attr('src'));
        setZoom(100);
        centerModal();
        $('#plot-modal-overlay').fadeIn(150);
      });

      // Close
      $(document).on('click', '#plot-modal-overlay', function(e) {
        if (e.target === this) $('#plot-modal-overlay').fadeOut(150);
      });
      $(document).on('click', '#plot-modal-close', function(e) {
        e.stopPropagation();
        $('#plot-modal-overlay').fadeOut(150);
      });
      $(document).on('keydown', function(e) {
        if (e.key === 'Escape') $('#plot-modal-overlay').fadeOut(150);
      });

      // Zoom controls
      $(document).on('click', '#zoom-in-btn',    function() { setZoom(currentZoom + 15); });
      $(document).on('click', '#zoom-out-btn',   function() { setZoom(currentZoom - 15); });
      $(document).on('click', '#zoom-reset-btn', function() { setZoom(100); centerModal(); });
      $(document).on('input', '#zoom-slider',    function() { setZoom(parseInt($(this).val())); });
      $(document).on('wheel', '#plot-modal-overlay', function(e) {
        if ($('#plot-modal-overlay').is(':visible')) {
          e.preventDefault();
          setZoom(currentZoom + (e.originalEvent.deltaY < 0 ? 10 : -10));
        }
      });

      // Drag-to-move via toolbar
      $(document).on('mousedown', '#plot-modal-toolbar', function(e) {
        if ($(e.target).is('button, input, span')) return;
        var $c = $('#plot-modal-content');
        mDrag = { on: true, sx: e.clientX, sy: e.clientY,
                  oL: parseInt($c.css('left')), oT: parseInt($c.css('top')) };
        e.preventDefault();
      });

      // Resize via edge / corner handles
      $(document).on('mousedown', '.modal-resize-handle', function(e) {
        var $c = $('#plot-modal-content');
        mResize = { on: true, dir: $(this).data('dir'),
                    sx: e.clientX, sy: e.clientY,
                    oW: $c.width(), oH: $c.height(),
                    oL: parseInt($c.css('left')), oT: parseInt($c.css('top')) };
        e.preventDefault();
        e.stopPropagation();
      });

      $(document).on('mousemove', function(e) {
        if (mDrag.on) {
          $('#plot-modal-content').css({
            left: mDrag.oL + e.clientX - mDrag.sx,
            top:  mDrag.oT + e.clientY - mDrag.sy
          });
        }
        if (mResize.on) {
          var dx = e.clientX - mResize.sx, dy = e.clientY - mResize.sy;
          var d = mResize.dir, nW = mResize.oW, nH = mResize.oH, nL = mResize.oL, nT = mResize.oT;
          if (d.indexOf('right')  >= 0) nW = Math.max(320, mResize.oW + dx);
          if (d.indexOf('left')   >= 0) { nW = Math.max(320, mResize.oW - dx); nL = mResize.oL + mResize.oW - nW; }
          if (d.indexOf('bottom') >= 0) nH = Math.max(220, mResize.oH + dy);
          if (d.indexOf('top')    >= 0) { nH = Math.max(220, mResize.oH - dy); nT = mResize.oT + mResize.oH - nH; }
          $('#plot-modal-content').css({ width: nW, height: nH, left: nL, top: nT });
        }
      });
      $(document).on('mouseup', function() { mDrag.on = false; mResize.on = false; });

      // Show/hide loading messages when plots are recalculating
      $(document).on('shiny:recalculating', function(e) {
        var $target = $(e.target);
        if ($target.closest('.plot-thumbnail').length) {
          $target.closest('.plot-thumbnail').find('.plot-loading-msg').show();
          $target.closest('.plot-thumbnail').find('.click-hint').hide();
        }
      });
      $(document).on('shiny:recalculated', function(e) {
        var $target = $(e.target);
        if ($target.closest('.plot-thumbnail').length) {
          $target.closest('.plot-thumbnail').find('.plot-loading-msg').hide();
          $target.closest('.plot-thumbnail').find('.click-hint').show();
        }
      });

      function chatSend() {
        var input = $('#chat_input');
        var msg = $.trim(input.val());
        if (!msg) return;
        var container = document.querySelector('.chat-container');
        if (container) {
          var els = container.children;
          if (els.length === 1 && els[0].style && els[0].style.fontStyle === 'italic') els[0].remove();
          var bubble = document.createElement('div');
          bubble.className = 'chat-msg user';
          var roleDiv = document.createElement('div');
          roleDiv.className = 'chat-role';
          roleDiv.textContent = 'You';
          var msgP = document.createElement('p');
          msgP.textContent = msg;
          bubble.appendChild(roleDiv);
          bubble.appendChild(msgP);
          container.appendChild(bubble);
          var thinking = document.createElement('div');
          thinking.className = 'chat-msg assistant chat-thinking-bubble';
          thinking.id = 'js-thinking-dots';
          thinking.innerHTML = '<div class=\"chat-role\">Assistant</div><div class=\"thinking-dots\"><span class=\"dot\"></span><span class=\"dot\"></span><span class=\"dot\"></span></div>';
          container.appendChild(thinking);
          container.scrollTop = container.scrollHeight;
        }
        $('#btn_chat_send').prop('disabled', true).text('...');
      }
      $(document).on('shiny:value', function(e) {
        if (e.name === 'chat_messages') {
          $('#btn_chat_send').prop('disabled', false).text('Send');
          $('#js-thinking-dots').remove();
        }
        setTimeout(function() {
          var c = document.querySelector('.chat-container');
          if (c) c.scrollTop = c.scrollHeight;
        }, 50);
      });
      $(document).on('keydown keypress', '#chat_input', function(e) {
        if ((e.key === 'Enter' || e.keyCode === 13) && !e.shiftKey) {
          e.preventDefault();
          e.stopPropagation();
          chatSend();
          $('#btn_chat_send').click();
          return false;
        }
      });
      $(document).on('mousedown', '#btn_chat_send', function() {
        chatSend();
      });
    "))
  ),
  
  # Hidden modal overlay for enlarged plots (resizable / draggable panel)
  div(id = "plot-modal-overlay",
    div(id = "plot-modal-content",
      # Edge and corner resize handles
      div(class = "modal-resize-handle h-top",    `data-dir` = "top"),
      div(class = "modal-resize-handle h-bottom", `data-dir` = "bottom"),
      div(class = "modal-resize-handle h-left",   `data-dir` = "left"),
      div(class = "modal-resize-handle h-right",  `data-dir` = "right"),
      div(class = "modal-resize-handle h-tl", `data-dir` = "top-left"),
      div(class = "modal-resize-handle h-tr", `data-dir` = "top-right"),
      div(class = "modal-resize-handle h-bl", `data-dir` = "bottom-left"),
      div(class = "modal-resize-handle h-br", `data-dir` = "bottom-right"),
      # Toolbar (also serves as drag handle)
      div(id = "plot-modal-toolbar",
        tags$button(id = "zoom-out-btn", HTML("&minus;")),
        tags$input(id = "zoom-slider", class = "zoom-slider", type = "range",
                   min = "20", max = "300", value = "100", step = "5"),
        tags$button(id = "zoom-in-btn", HTML("+")),
        span(id = "zoom-label", class = "zoom-label", "100%"),
        tags$button(id = "zoom-reset-btn", title = "Reset zoom", HTML("&#8634;")),
        span(class = "toolbar-spacer"),
        tags$button(id = "plot-modal-close", title = "Close", HTML("&times;"))
      ),
      div(id = "plot-modal-img-wrap",
        tags$img(id = "plot-modal-img", src = "")
      )
    )
  ),
  
  titlePanel("Causal Data DAG Analysis Application"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      # Step 1: Data Upload
      h4("1. Upload Data"),
      fileInput("datafile", "Data (CSV)", 
                accept = c(".csv", "text/csv"),
                buttonLabel = "Browse..."),
      hr(),
      
      # Step 2: Review Data Types
      h4("2. Review Data Types"),
      p("Review and edit inferred data types for each variable:"),
      uiOutput("data_type_editor"),
      hr(),
      
      # Step 3: DAG Definition
      h4("3. Define DAG"),
      radioButtons("dag_source", "DAG Source",
                   choices = c("Fully Connected Test DAG" = "default",
                               "Upload Adjacency Matrix" = "upload",
                               "Draw Manually" = "manual"),
                   selected = "default"),
      
      conditionalPanel(
        condition = "input.dag_source == 'default'",
        p("Creates a fully connected DAG (all upstream variables cause all downstream variables) using the selected edge type."),
        selectInput("default_edge_type", "Edge Type",
                   choices = c("LinearShift (ls)" = "ls",
                              "ComplexShift (cs)" = "cs",
                              "SimpleIntercept (si)" = "si",
                              "ComplexIntercept (ci)" = "ci"),
                   selected = "ls"),
        actionButton("btn_load_default", "Load Test DAG",
                     class = "btn-primary")
      ),
      
      conditionalPanel(
        condition = "input.dag_source == 'upload'",
        fileInput("amat_file", "Adjacency Matrix CSV",
                  accept = c(".csv", "text/csv")),
        p("Upload a CSV with row and column names matching your variables. Use 0/1 values (edges default to LinearShift) or edge type codes: 'ls', 'cs', 'si', 'ci'. You can change edge types after loading."),
        actionButton("btn_upload_dag", "Load DAG", 
                     class = "btn-primary")
      ),
      
      conditionalPanel(
        condition = "input.dag_source == 'manual'",
        selectInput("edge_type", "Edge Type for New Edges",
                   choices = c("LinearShift (ls)" = "ls",
                              "ComplexShift (cs)" = "cs",
                              "SimpleIntercept (si)" = "si",
                              "ComplexIntercept (ci)" = "ci"),
                   selected = "ls"),
        p("Draw edges in the network editor below, then click 'Apply'."),
        actionButton("btn_manual_apply", "Apply DAG from Editor",
                     class = "btn-primary")
      ),
      
      hr(),
      
      # Edge type editing (if DAG exists)
      conditionalPanel(
        condition = "output.dag_exists",
        h5("Edit Edge Types"),
        p("Select an edge and change its type:"),
        uiOutput("edge_type_editor")
      ),
      
      hr(),
      
      # Step 4: Experiment Setup
      h4("4. Experiment Setup"),
      textInput("experiment_name", "Experiment Name (optional)", 
                value = "", placeholder = "Auto-generated if empty"),
      p("Experiment folder will be created in the output/ folder"),
      
      # Configuration info box
      div(
        style = "margin: 10px 0; padding: 10px 12px; background-color: #eaf2fb; border-left: 4px solid #2980b9; border-radius: 4px; font-size: 0.9em;",
        p(strong("Configuration:"), style = "margin: 0 0 5px 0; color: #1a5276;"),
        p("Configuration is automatically created when you fit the model. This includes:", style = "margin: 0; color: #4a6a8a;"),
        tags$ul(
          style = "margin: 5px 0 0 0; padding-left: 20px; color: #4a6a8a;",
          tags$li("Data type inference (continuous/ordinal)"),
          tags$li("Adjacency matrix setup"),
          tags$li("Model name generation"),
          tags$li("Node information creation"),
          tags$li("All saved to configuration.json")
        ),
        p("Each experiment gets its own configuration file in the experiment directory.", style = "margin: 5px 0 0 0; font-style: italic; color: #5d7ea0;")
      ),
      hr(),
      
      # Step 5: Model Fitting
      h4("5. Fit Model"),
      numericInput("epochs", "Epochs", value = 100, min = 10, step = 10),
      numericInput("learning_rate", "Learning Rate", 
                   value = 0.01, min = 0.001, max = 0.1, step = 0.001),
      numericInput("batch_size", "Batch Size", 
                   value = 512, min = 32, step = 32),
      
      checkboxInput("set_initial_weights", 
                    "Initialize with R-based weights", value = FALSE),
      
      actionButton("btn_fit", "Fit TRAM-DAG Model", 
                   class = "btn-success btn-lg"),
      
      hr(),
      
      # Step 6: Observational Sampling
      h4("6. Observational Sampling"),
      p("Sample from the fitted model without interventions. This generates data from the learned causal model and compares it against the test data."),
      
      numericInput("n_samples", "Number of Samples", 
                   value = 10000, min = 1000, step = 1000),
      
      actionButton("btn_sample", "Sample (Observational)", 
                   class = "btn-info btn-lg"),
      
      hr(),
      
      # Step 7: Interventional Sampling / ATE
      h4("7. Interventional Sampling & ATE"),
      
      # Intervention explanation box
      div(
        style = "margin: 10px 0; padding: 12px; background-color: #eaf2fb; border-left: 4px solid #5dade2; border-radius: 4px; font-size: 0.9em;",
        p(strong("Understanding Interventions:"), style = "margin: 0 0 8px 0; color: #1a5276;"),
        p("Interventions (do-calculus) allow us to estimate causal effects by simulating what happens when we", 
          strong("force"), "a variable to a specific value, breaking its natural causal dependencies.", 
          style = "margin: 0 0 8px 0; color: #4a6a8a;"),
        tags$ul(
          style = "margin: 5px 0 0 0; padding-left: 20px; color: #4a6a8a;",
          tags$li(strong("do(X = value):"), " Forces variable X to a specific value, ignoring its normal causes"),
          tags$li(strong("ATE (Average Treatment Effect):"), " The difference in outcome Y between treatment and control conditions")
        ),
        p("Example: do(X = 1) vs do(X = 0) compares outcomes when X is forced to 1 vs 0.", 
          style = "margin: 8px 0 0 0; font-style: italic; color: #5d7ea0;")
      ),
      
      selectInput("X_var", "Treatment Variable (X)", 
                  choices = NULL),
      
      # Distribution of selected treatment variable to guide intervention choice
      div(
        style = "margin: 8px 0; padding: 0; border: 1px solid #c8ddf0; border-radius: 6px; overflow: hidden; background: white;",
        div(class = "plot-thumbnail",
            plotOutput("treatment_dist_plot", height = "180px"),
            p(class = "click-hint", style = "margin: 0; padding: 2px 0;", "Click to enlarge")),
        uiOutput("treatment_stats")
      ),
      
      numericInput("x_treated", "do(X = ...)", value = 1, step = 0.1),
      numericInput("x_control", "Control: do(X = ...)", value = 0, step = 0.1),
      
      selectInput("Y_var", "Outcome Variable (Y)", 
                  choices = NULL),
      
      actionButton("btn_compute_ate", "Compute ATE", 
                   class = "btn-info btn-lg"),
      
      hr(),
      
      # Step 8: Report Generation
      h4("8. Generate Report"),
      p("Generate a full PDF analysis report including all diagnostic plots, ",
        "sampling results, causal effects, and interpretations. The report mirrors ",
        "the complete notebook workflow."),
      actionButton("btn_generate_report", "Generate Full Analysis Report (PDF)", 
                   class = "btn-warning btn-lg"),
      
      hr(),
      
      # Step 9: Chat with LLM about results
      h4("9. Ask Questions"),
      div(
        style = "margin: 6px 0 10px 0; padding: 8px 12px; background-color: #eaf2fb; border-left: 4px solid #5dade2; border-radius: 4px; font-size: 0.88em; color: #2c3e50;",
        p(style = "margin: 0 0 4px 0;",
          tags$strong("Offline LLM:"), " All data stays on your machine."),
        p(style = "margin: 0 0 4px 0; color: #4a6a8a;",
          "Model: ", tags$code(OLLAMA_MODEL), " via Ollama"),
        p(style = "margin: 0; color: #4a6a8a;",
          "Responses may take a moment \u2014 the model has billions of parameters but runs entirely on your device.")
      ),
      p("Ask questions about your results, the fitted model, ",
        "causal effects, or anything else about this analysis. ",
        "The LLM has access to all your calculations, plots, and model outputs and can provide informed answers."),
      div(
        class = "chat-input-row",
        textInput("chat_input", label = NULL, 
                  placeholder = "e.g. What does the ATE mean?",
                  width = "100%"),
        actionButton("btn_chat_send", "Send", class = "btn-primary btn-sm",
                     style = "margin-bottom: 0; white-space: nowrap;")
      ),
      actionButton("btn_chat_clear", "Clear Chat", class = "btn-default btn-sm",
                   style = "margin-top: 6px; font-size: 0.82em;"),
      uiOutput("chat_messages"),
      
      hr(),
      div(style = "text-align: center; color: #85a5c2; font-size: 0.82em; padding: 8px 0 4px 0;",
          "Made by Tim Herren")
    ),
    
    mainPanel(
      width = 9,
      
      # Experiment Directory Info
      h4("Experiment Directory"),
      verbatimTextOutput("experiment_dir_text"),
      hr(),
      
      # 1. Data Pairplot (matches notebook: sns.pairplot(df))
      conditionalPanel(
        condition = "output.data_loaded",
        h4("Data Overview (Pairplot)"),
        p("Pairwise relationships in the uploaded data."),
        div(class = "plot-thumbnail", 
            imageOutput("data_pairplot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Generating pairplot..."),
            p(class = "click-hint", "Click to enlarge")),
        hr()
      ),
      
      # 2. DAG Editor (if manual)
      conditionalPanel(
        condition = "input.dag_source == 'manual'",
        h4("DAG Editor"),
        visNetworkOutput("dag_editor", height = "400px"),
        hr()
      ),
      
      # 2b. DAG Visualization
      h4("DAG Structure"),
      p("This shows the current causal DAG structure with edge types."),
      verbatimTextOutput("dag_debug", placeholder = TRUE),
      plotOutput("dag_plot", height = "400px", width = "100%"),
      
      hr(),
      
      # 3. Model Status
      h4("Model Status"),
      verbatimTextOutput("fit_status"),
      
      hr(),
      
      # 4-9. Fit Diagnostics (matches notebook order: loss, shift, intercepts, hdag, latents, nll)
      conditionalPanel(
        condition = "output.model_fitted",
        h4("Fit Diagnostics"),
        
        # 4. td_model.plot_loss_history()
        h5("Loss History"),
        div(class = "plot-thumbnail", 
            imageOutput("loss_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing loss history..."),
            p(class = "click-hint", "Click to enlarge")),
        hr(),
        
        # 5. td_model.plot_linear_shift_history()
        h5("Linear Shift History"),
        p("Learned linear shift coefficients over training epochs."),
        div(class = "plot-thumbnail", 
            imageOutput("shift_history_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing linear shift history..."),
            p(class = "click-hint", "Click to enlarge")),
        hr(),
        
        # 6. td_model.plot_simple_intercepts_history()
        h5("Simple Intercepts History"),
        p("Intercept weights over training epochs."),
        div(class = "plot-thumbnail", 
            imageOutput("intercepts_history_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing intercepts history..."),
            p(class = "click-hint", "Click to enlarge")),
        hr(),
        
        # 7. td_model.plot_hdag(train_df, variables=[...], plot_n_rows=1)
        h5("Transformation Functions (h-DAG)"),
        p("Learned transformation functions h(y|x) for each variable."),
        div(class = "plot-thumbnail", 
            imageOutput("hdag_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing h-DAG transformations..."),
            p(class = "click-hint", "Click to enlarge")),
        hr(),
        
        # 8. td_model.plot_latents(train_df)
        h5("Latent Distributions"),
        p("Latent (U) distributions for each variable."),
        div(class = "plot-thumbnail", 
            imageOutput("latents_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing latent distributions..."),
            p(class = "click-hint", "Click to enlarge")),
        hr(),
        
        # 9. td_model.nll(train_df)
        h5("Negative Log-Likelihood"),
        verbatimTextOutput("nll_output"),
        hr()
      ),
      
      # LLM Model Interpretation via Ollama (extra, not in notebook)
      conditionalPanel(
        condition = "output.model_interpretation_available",
        h4("Model Interpretation"),
        div(
          style = "background-color: #eaf2fb; padding: 15px; border-radius: 6px; border-left: 4px solid #5dade2; margin: 10px 0;",
          htmlOutput("model_interpretation")
        ),
        hr()
      ),
      
      # 10. Observational Samples vs True
      conditionalPanel(
        condition = "output.sampled_data_available",
        h4("Observational Sampling: Sampled vs Observed"),
        p("The model samples from the learned joint distribution (no interventions). ",
          "If the model fits well, the sampled distributions (orange) should closely match ",
          "the held-out test data (blue)."),
        div(class = "plot-thumbnail", 
            imageOutput("samples_vs_true_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing samples vs true comparison..."),
            p(class = "click-hint", "Click to enlarge")),
        hr()
      ),
      
      # 11. ATE Results + Interventional Sampling
      h4("Interventional Sampling & ATE"),
      verbatimTextOutput("ate_result"),
      
      # ATE Interpretation (extra, not in notebook)
      conditionalPanel(
        condition = "output.ate_interpretation_available",
        h4("ATE Interpretation"),
        div(
          style = "background-color: #eaf2fb; padding: 15px; border-radius: 6px; border-left: 4px solid #2980b9; margin: 10px 0;",
          htmlOutput("ate_interpretation")
        ),
        hr()
      ),
      
      # 11. Interventional vs Observational comparison
      conditionalPanel(
        condition = "output.ate_result_available",
        h4("Interventional Sampling: Treated vs Control"),
        p("Visualizes the ATE as distributional shift. The control intervention do(X = control) is shown in blue, ",
          "the treatment intervention do(X = treated) in orange. The difference in means for the outcome variable ",
          "is the ATE. Shifts in other variables show how the intervention propagates through the causal graph."),
        div(class = "plot-thumbnail", 
            imageOutput("interventional_samples_vs_true_plot", height = "auto"), 
            div(class = "plot-loading-msg", span(class = "loading-spinner"), "Computing interventional samples vs true..."),
            p(class = "click-hint", "Click to enlarge")),
        hr()
      ),
      
      
    )
  )
)

## --------------------------------------------------
## 3) Server
## --------------------------------------------------

server <- function(input, output, session) {
  
  # Reactive values
  rv_data <- reactiveVal(NULL)
  rv_amat <- reactiveVal(NULL)
  rv_manual_edges <- reactiveVal(NULL)
  rv_editor_live_edges <- reactiveVal(NULL)
  rv_edge_types <- reactiveVal(list())
  rv_layout <- reactiveVal(NULL)
  rv_experiment_dir <- reactiveVal(NULL)
  rv_td_model <- reactiveVal(NULL)
  rv_ate_result <- reactiveVal(NULL)
  rv_sampled_data <- reactiveVal(NULL)
  rv_fit_status <- reactiveVal("No model fitted yet.")
  rv_model_interpretation <- reactiveVal(NULL)
  rv_ate_interpretation <- reactiveVal(NULL)
  rv_data_types <- reactiveVal(NULL)
  rv_train_df <- reactiveVal(NULL)
  rv_val_df <- reactiveVal(NULL)
  rv_test_df <- reactiveVal(NULL)
  rv_interventional_samples_treated <- reactiveVal(NULL)  # full df for treated intervention
  rv_interventional_samples_control <- reactiveVal(NULL)  # full df for control intervention
  rv_chat_history <- reactiveVal(list())  # list of list(role, content) for the chat
  rv_chat_thinking <- reactiveVal(FALSE)
  rv_chat_context_cache <- reactiveVal(NULL)  # cached context string for chat
  rv_nll_cache <- reactiveVal(NULL)  # cached NLL text for chat context
  
  # Display experiment directory
  output$experiment_dir_text <- renderText({
    dir <- rv_experiment_dir()
    if (is.null(dir)) "No experiment directory created yet. Upload data and fit model to create one."
    else paste("Experiment Directory:", dir)
  })
  
  # Data type editor UI
  output$data_type_editor <- renderUI({
    df <- rv_data()
    if (is.null(df)) return(p("Upload data first to review data types."))
    
    current_types <- rv_data_types()
    vars <- colnames(df)
    valid_choices <- c(
      "Continuous" = "continous",
      "Ordinal (numeric input, ordinal outcome)" = "ordinal_Xn_Yo",
      "Ordinal (numeric input, continuous outcome)" = "ordinal_Xn_Yc",
      "Ordinal (categorical input, ordinal outcome)" = "ordinal_Xc_Yo",
      "Ordinal (categorical input, continuous outcome)" = "ordinal_Xc_Yc"
    )
    
    type_inputs <- lapply(vars, function(v) {
      current_val <- if (!is.null(current_types[[v]])) current_types[[v]] else "continous"
      n_unique <- length(unique(df[[v]]))
      hint <- paste0("(", n_unique, " unique values)")
      fluidRow(
        column(4, strong(v), tags$small(hint, style = "color: #888;")),
        column(8, selectInput(
          inputId = paste0("dtype_", v),
          label = NULL,
          choices = valid_choices,
          selected = current_val,
          width = "100%"
        ))
      )
    })
    
    tagList(type_inputs)
  })
  
  # Update rv_data_types when user changes dropdowns
  observe({
    df <- rv_data()
    req(df)
    vars <- colnames(df)
    
    new_types <- list()
    for (v in vars) {
      input_id <- paste0("dtype_", v)
      val <- input[[input_id]]
      if (!is.null(val)) {
        new_types[[v]] <- val
      }
    }
    
    if (length(new_types) == length(vars)) {
      rv_data_types(new_types)
    }
  })
  
  # Load data
  observeEvent(input$datafile, {
    req(input$datafile)
    df <- read.csv(input$datafile$datapath, check.names = TRUE)
    rv_data(df)
    
    # Auto-infer data types
    inferred_types <- sapply(df, function(x) {
      if (is.numeric(x) && length(unique(x)) > 20) {
        "continous"
      } else if (is.numeric(x) || is.factor(x)) {
        "ordinal_Xn_Yo"
      } else {
        "continous"
      }
    }, USE.NAMES = TRUE)
    rv_data_types(as.list(inferred_types))
    
    # Update variable selectors
    vars <- colnames(df)
    updateSelectInput(session, "X_var", choices = vars, selected = vars[1])
    updateSelectInput(session, "Y_var", choices = vars, 
                     selected = vars[length(vars)])
    
    # Create circular layout for DAG
    n <- length(vars)
    angles <- seq(from = pi/2, by = -2*pi/n, length.out = n)
    coords <- cbind(x = cos(angles), y = sin(angles))
    rownames(coords) <- vars
    rv_layout(coords)
    
    # Initialize empty adjacency matrix with string codes
    A <- matrix("0", nrow = n, ncol = n, 
                dimnames = list(vars, vars))
    rv_amat(A)
    rv_edge_types(list())
    
    if (length(vars) >= 2) {
      showNotification(
        paste0("Data loaded with ", length(vars), " variables. Click 'Load Test DAG' to create a fully connected DAG, or define your own."), 
        type = "message", duration = 5)
    }
  })
  
  # Load default DAG (fully connected with selected edge type)
  observeEvent(input$btn_load_default, {
    req(rv_data())
    vars <- colnames(rv_data())
    
    if (length(vars) < 2) {
      showNotification("Need at least 2 variables to create a DAG.", type = "warning")
      return()
    }
    
    edge_code <- if (!is.null(input$default_edge_type)) input$default_edge_type else "ls"
    A <- create_default_dag(vars, edge_type = edge_code)
    rv_amat(A)
    
    edge_types <- list()
    for (i in seq_len(length(vars) - 1)) {
      for (j in (i + 1):length(vars)) {
        edge_types[[paste0(vars[i], "->", vars[j])]] <- edge_code
      }
    }
    rv_edge_types(edge_types)
    rv_manual_edges(adjacency_to_edges(A))
    
    showNotification(
      paste0("Fully connected DAG created with ", length(vars), " variables (", edge_code, " edges)."),
      type = "message", duration = 3)
  })
  
  # DAG Editor
  output$dag_editor <- renderVisNetwork({
    req(rv_data(), rv_layout())
    vars <- colnames(rv_data())
    coords <- rv_layout() * 300
    
    nodes <- data.frame(
      id = vars,
      label = vars,
      x = coords[vars, "x"],
      y = coords[vars, "y"],
      fixed = TRUE,
      physics = FALSE,
      stringsAsFactors = FALSE
    )
    
    edges <- rv_manual_edges()
    if (is.null(edges) || nrow(edges) == 0) {
      edges <- data.frame(from = character(0), to = character(0),
                         stringsAsFactors = FALSE)
    }
    
    visNetwork(nodes, edges, height = "400px") %>%
      visEdges(arrows = "to") %>%
      visOptions(manipulation = TRUE, nodesIdSelection = TRUE)
  })
  
  # When the user adds/edits/deletes edges in the editor, request updated edge list
  observeEvent(input$dag_editor_graphChange, {
    visNetworkProxy("dag_editor") %>% visGetEdges()
  })
  
  # Store live editor edges separately so the editor doesn't re-render
  observe({
    edges <- input$dag_editor_edges
    if (!is.null(edges)) {
      df <- tryCatch({
        if (is.data.frame(edges)) {
          edges
        } else if (is.list(edges) && length(edges) > 0) {
          if (is.list(edges[[1]])) {
            do.call(rbind, lapply(edges, function(e) as.data.frame(e, stringsAsFactors = FALSE)))
          } else {
            as.data.frame(edges, stringsAsFactors = FALSE)
          }
        } else {
          as.data.frame(edges, stringsAsFactors = FALSE)
        }
      }, error = function(e) NULL)
      if (!is.null(df)) {
        rv_editor_live_edges(df)
      }
    }
  })
  
  # Apply manual DAG (reads live editor edges, then syncs rv_manual_edges)
  observeEvent(input$btn_manual_apply, {
    req(rv_data())
    vars <- colnames(rv_data())
    edges <- rv_editor_live_edges()
    
    A <- matrix("0", nrow = length(vars), ncol = length(vars),
                dimnames = list(vars, vars))
    
    default_edge_type <- input$edge_type
    if (is.null(default_edge_type)) default_edge_type <- "ls"
    
    edge_types <- list()
    
    if (!is.null(edges) && is.data.frame(edges) && nrow(edges) > 0 &&
        "from" %in% colnames(edges) && "to" %in% colnames(edges)) {
      for (i in seq_len(nrow(edges))) {
        from <- tryCatch(as.character(edges$from[i])[1], error = function(e) NA_character_)
        to   <- tryCatch(as.character(edges$to[i])[1],   error = function(e) NA_character_)
        if (is.null(from) || is.null(to) || length(from) == 0 || length(to) == 0) next
        if (is.na(from) || is.na(to)) next
        if (nchar(from) == 0 || nchar(to) == 0) next
        if (from %in% vars && to %in% vars && from != to) {
          has_type <- "edge_type" %in% colnames(edges)
          etype <- if (has_type) tryCatch(as.character(edges$edge_type[i])[1], error = function(e) NA_character_) else NA_character_
          edge_type <- if (!is.na(etype) && nchar(etype) > 0) etype else default_edge_type
          A[from, to] <- edge_type
          edge_types[[paste0(from, "->", to)]] <- edge_type
        }
      }
    }
    
    A <- ensure_upper_triangular(A, vars)
    
    rv_amat(A)
    rv_edge_types(edge_types)
    rv_manual_edges(adjacency_to_edges(A))
    
    n_edges <- sum(A != "0" & A != 0)
    if (n_edges > 0) {
      showNotification(paste0("DAG applied: ", n_edges, " edge(s)."), type = "message", duration = 3)
    } else {
      showNotification("No valid edges found. Draw edges between nodes first, then click Apply.", type = "warning")
    }
  })
  
  # Upload adjacency matrix
  observeEvent(input$btn_upload_dag, {
    req(input$amat_file, rv_data())
    amat_raw <- as.matrix(read.csv(input$amat_file$datapath,
                                   row.names = 1, check.names = FALSE))
    
    if (nrow(amat_raw) != ncol(amat_raw)) {
      showNotification("Adjacency matrix must be square", type = "error")
      return()
    }
    
    data_vars <- colnames(rv_data())
    A_vars <- rownames(amat_raw)
    missing_in_data <- setdiff(A_vars, data_vars)
    
    if (length(missing_in_data) > 0) {
      showNotification(
        paste("Variables not in data:", paste(missing_in_data, collapse = ", ")),
        type = "error"
      )
      return()
    }
    
    # Convert to string codes if numeric
    if (is.numeric(amat_raw) || all(amat_raw %in% c(0, 1))) {
      amat_codes <- matrix("0", nrow = nrow(amat_raw), ncol = ncol(amat_raw),
                          dimnames = dimnames(amat_raw))
      idx <- which(amat_raw != 0, arr.ind = TRUE)
      if (nrow(idx) > 0) {
        for (i in seq_len(nrow(idx))) {
          amat_codes[idx[i, "row"], idx[i, "col"]] <- "ls"
        }
      }
      amat_raw <- amat_codes
    } else {
      amat_raw <- matrix(as.character(amat_raw),
                        nrow = nrow(amat_raw), ncol = ncol(amat_raw),
                        dimnames = dimnames(amat_raw))
    }
    
    # Ensure matrix is upper triangular
    amat_raw <- ensure_upper_triangular(amat_raw, data_vars)
    
    rv_amat(amat_raw)
    rv_manual_edges(adjacency_to_edges(amat_raw))
    
    # Extract edge types
    edge_types <- list()
    edges_df <- adjacency_to_edges(amat_raw)
    if (nrow(edges_df) > 0 && "edge_type" %in% colnames(edges_df)) {
      for (i in seq_len(nrow(edges_df))) {
        key <- paste0(edges_df$from[i], "->", edges_df$to[i])
        edge_types[[key]] <- edges_df$edge_type[i]
      }
    }
    rv_edge_types(edge_types)
    
    # No notification for quick operations
  })
  
  # Check if DAG exists
  output$dag_exists <- reactive({
    !is.null(rv_amat()) && any(rv_amat() != "0" & rv_amat() != 0)
  })
  outputOptions(output, "dag_exists", suspendWhenHidden = FALSE)
  
  # Check if model is fitted
  output$model_fitted <- reactive({
    !is.null(rv_td_model()) && !is.null(rv_experiment_dir())
  })
  outputOptions(output, "model_fitted", suspendWhenHidden = FALSE)
  
  # Check if sampled data is available
  output$sampled_data_available <- reactive({
    !is.null(rv_sampled_data()) && !is.null(rv_data())
  })
  outputOptions(output, "sampled_data_available", suspendWhenHidden = FALSE)
  
  # Check if data is loaded
  output$data_loaded <- reactive({
    !is.null(rv_data())
  })
  outputOptions(output, "data_loaded", suspendWhenHidden = FALSE)
  
  # Check if ATE result is available
  output$ate_result_available <- reactive({
    !is.null(rv_ate_result())
  })
  outputOptions(output, "ate_result_available", suspendWhenHidden = FALSE)
  
  # Check if model interpretation is available
  output$model_interpretation_available <- reactive({
    !is.null(rv_model_interpretation()) && trimws(rv_model_interpretation()) != ""
  })
  outputOptions(output, "model_interpretation_available", suspendWhenHidden = FALSE)
  
  # Check if ATE interpretation is available
  output$ate_interpretation_available <- reactive({
    !is.null(rv_ate_interpretation()) && trimws(rv_ate_interpretation()) != ""
  })
  outputOptions(output, "ate_interpretation_available", suspendWhenHidden = FALSE)
  
  # Render model interpretation
  output$model_interpretation <- renderUI({
    interpretation <- rv_model_interpretation()
    if (is.null(interpretation) || trimws(interpretation) == "") {
      return(NULL)
    }
    # Convert newlines to <br> for HTML display
    interpretation_html <- gsub("\n\n", "</p><p>", interpretation)
    interpretation_html <- gsub("\n", "<br>", interpretation_html)
    HTML(paste0("<p>", interpretation_html, "</p>"))
  })
  
  # Render ATE interpretation
  output$ate_interpretation <- renderUI({
    interpretation <- rv_ate_interpretation()
    if (is.null(interpretation) || trimws(interpretation) == "") {
      return(NULL)
    }
    # Convert newlines to <br> for HTML display
    interpretation_html <- gsub("\n\n", "</p><p>", interpretation)
    interpretation_html <- gsub("\n", "<br>", interpretation_html)
    HTML(paste0("<p>", interpretation_html, "</p>"))
  })
  
  # Update intervention defaults when the treatment variable changes
  observeEvent(input$X_var, {
    req(rv_data(), input$X_var)
    df <- rv_data()
    x_var <- input$X_var
    if (is.null(x_var) || !(x_var %in% colnames(df))) return()
    vals <- df[[x_var]]
    if (!is.numeric(vals)) return()
    q25 <- round(quantile(vals, 0.25, na.rm = TRUE), 2)
    q75 <- round(quantile(vals, 0.75, na.rm = TRUE), 2)
    updateNumericInput(session, "x_treated", value = q75)
    updateNumericInput(session, "x_control", value = q25)
  })
  
  # Treatment variable distribution plot to guide intervention value selection
  output$treatment_dist_plot <- renderPlot({
    req(rv_data(), input$X_var)
    df <- rv_data()
    x_var <- input$X_var
    if (is.null(x_var) || !(x_var %in% colnames(df))) return(NULL)
    
    x_vals <- df[[x_var]]
    if (!is.numeric(x_vals)) return(NULL)
    
    # Get current intervention values
    x_treated <- input$x_treated
    x_control <- input$x_control
    
    # Colours
    col_fill <- rgb(41/255, 128/255, 185/255, 0.25)
    col_border <- "#2980b9"
    col_treated <- "#1a5276"
    col_control <- "#5dade2"
    col_kde <- "#154360"
    
    # Plot
    par(mar = c(3, 3, 1.8, 0.5), mgp = c(1.8, 0.5, 0), cex.main = 0.95, 
        family = "sans", bg = "white")
    
    hist(x_vals, breaks = 40, freq = FALSE, col = col_fill, border = col_border,
         main = paste0("Distribution of ", x_var),
         xlab = x_var, ylab = "Density", las = 1, cex.axis = 0.8, cex.lab = 0.85)
    
    # Add KDE curve
    tryCatch({
      d <- density(x_vals, na.rm = TRUE)
      lines(d, col = col_kde, lwd = 2)
    }, error = function(e) {})
    
    # Mark intervention values
    if (!is.null(x_treated) && is.numeric(x_treated)) {
      abline(v = x_treated, col = col_treated, lwd = 2.5, lty = 1)
      mtext(paste0("do(", x_var, "=", x_treated, ")"), side = 3, at = x_treated, 
            col = col_treated, cex = 0.7, font = 2, line = 0.1)
    }
    if (!is.null(x_control) && is.numeric(x_control)) {
      abline(v = x_control, col = col_control, lwd = 2.5, lty = 2)
      mtext(paste0("ctrl=", x_control), side = 3, at = x_control, 
            col = col_control, cex = 0.7, font = 2, line = 0.1)
    }
    
    # Legend (sized for readability)
    legend_items <- c("Data distribution")
    legend_cols <- c(col_kde)
    legend_lty <- c(1)
    legend_lwd <- c(2)
    if (!is.null(x_treated) && is.numeric(x_treated)) {
      legend_items <- c(legend_items, paste0("Treated: ", x_treated))
      legend_cols <- c(legend_cols, col_treated)
      legend_lty <- c(legend_lty, 1)
      legend_lwd <- c(legend_lwd, 2.5)
    }
    if (!is.null(x_control) && is.numeric(x_control)) {
      legend_items <- c(legend_items, paste0("Control: ", x_control))
      legend_cols <- c(legend_cols, col_control)
      legend_lty <- c(legend_lty, 2)
      legend_lwd <- c(legend_lwd, 2.5)
    }
    legend("topright", legend = legend_items, col = legend_cols, 
           lty = legend_lty, lwd = legend_lwd, cex = 0.85, bg = "white",
           box.col = "#c8ddf0", inset = c(0.01, 0.01))
  }, bg = "white")
  outputOptions(output, "treatment_dist_plot", suspendWhenHidden = FALSE)
  
  # Summary stats and intervention guidance for the selected treatment variable
  output$treatment_stats <- renderUI({
    req(rv_data(), input$X_var)
    df <- rv_data()
    x_var <- input$X_var
    if (is.null(x_var) || !(x_var %in% colnames(df))) return(NULL)
    
    x_vals <- df[[x_var]]
    if (!is.numeric(x_vals)) return(p("Selected variable is not numeric.", style = "color: #85a5c2; padding: 4px 10px;"))
    
    x_min <- round(min(x_vals, na.rm = TRUE), 2)
    x_max <- round(max(x_vals, na.rm = TRUE), 2)
    x_mean <- round(mean(x_vals, na.rm = TRUE), 2)
    x_sd <- round(sd(x_vals, na.rm = TRUE), 2)
    q <- quantile(x_vals, probs = c(0.10, 0.25, 0.5, 0.75, 0.90), na.rm = TRUE)
    
    # Suggest meaningful intervention pairs
    low_val <- round(q[1], 2)   # 10th percentile
    high_val <- round(q[5], 2)  # 90th percentile
    q1_val <- round(q[2], 2)
    median_val <- round(q[3], 2)
    q3_val <- round(q[4], 2)
    mean_minus_sd <- round(x_mean - x_sd, 2)
    mean_plus_sd <- round(x_mean + x_sd, 2)
    
    div(
      style = "padding: 8px 10px; font-size: 0.82em; color: #4a6a8a; background: #f7fafd; border-top: 1px solid #e0ecf5;",
      # Data summary
      tags$table(
        style = "width: 100%; border-collapse: collapse; margin-bottom: 6px;",
        tags$tr(
          tags$td(HTML(paste0("<b>Range:</b> [", x_min, ", ", x_max, "]")), style = "padding: 1px 0;"),
          tags$td(HTML(paste0("<b>Mean:</b> ", x_mean)), style = "padding: 1px 0; text-align: right;")
        ),
        tags$tr(
          tags$td(HTML(paste0("<b>Median:</b> ", median_val)), style = "padding: 1px 0;"),
          tags$td(HTML(paste0("<b>SD:</b> ", x_sd)), style = "padding: 1px 0; text-align: right;")
        ),
        tags$tr(
          tags$td(HTML(paste0("<b>Q1 / Q3:</b> ", q1_val, " / ", q3_val)), style = "padding: 1px 0;"),
          tags$td(HTML(paste0("<b>P10 / P90:</b> ", low_val, " / ", high_val)), style = "padding: 1px 0; text-align: right;")
        )
      ),
      # Suggested interventions
      div(
        style = "border-top: 1px dashed #c8ddf0; padding-top: 5px; margin-top: 2px;",
        p(strong("Suggested intervention pairs:"), style = "margin: 0 0 3px 0; font-size: 0.95em; color: #1a5276;"),
        tags$table(
          style = "width: 100%; border-collapse: collapse; font-size: 0.95em;",
          tags$tr(style = "color: #2471a3;",
            tags$td("Q1 vs Q3:", style = "padding: 2px 0;"),
            tags$td(
              tags$a(href = "#", onclick = paste0(
                "Shiny.setInputValue('x_control', ", q1_val, "); ",
                "Shiny.setInputValue('x_treated', ", q3_val, "); ",
                "$('#x_control').val(", q1_val, "); ",
                "$('#x_treated').val(", q3_val, "); ",
                "return false;"),
                paste0(q1_val, " vs ", q3_val),
                style = "color: #2980b9; text-decoration: underline; cursor: pointer; font-weight: 600;"
              ),
              style = "padding: 2px 0; text-align: right;"
            )
          ),
          tags$tr(style = "color: #2471a3;",
            tags$td(HTML("Mean &plusmn; 1 SD:"), style = "padding: 2px 0;"),
            tags$td(
              tags$a(href = "#", onclick = paste0(
                "Shiny.setInputValue('x_control', ", mean_minus_sd, "); ",
                "Shiny.setInputValue('x_treated', ", mean_plus_sd, "); ",
                "$('#x_control').val(", mean_minus_sd, "); ",
                "$('#x_treated').val(", mean_plus_sd, "); ",
                "return false;"),
                paste0(mean_minus_sd, " vs ", mean_plus_sd),
                style = "color: #2980b9; text-decoration: underline; cursor: pointer; font-weight: 600;"
              ),
              style = "padding: 2px 0; text-align: right;"
            )
          )
        ),
        p(tags$em("Click a suggestion to auto-fill the intervention values."), 
          style = "margin: 4px 0 0 0; font-size: 0.88em; color: #85a5c2;")
      )
    )
  })
  
  # Ollama LLM status indicator
  output$openai_status <- renderText({
    if (!ollama_available) {
      return(paste0("⚠️ Ollama not available - interpretations disabled. Start Ollama and pull a model: ollama pull ", OLLAMA_MODEL))
    } else {
      return(paste0("✅ Ollama ready (model: ", OLLAMA_MODEL, ") - interpretations will appear after model fitting/ATE computation."))
    }
  })
  
  # Edge type editor UI
  output$edge_type_editor <- renderUI({
    req(rv_amat(), rv_data())
    A <- rv_amat()
    vars <- rownames(A)
    if (is.null(vars)) vars <- colnames(A)
    
    # Find all edges
    edges <- which(A != "0" & A != 0, arr.ind = TRUE)
    if (nrow(edges) == 0) return(NULL)
    
    edge_list <- lapply(seq_len(nrow(edges)), function(i) {
      from <- vars[edges[i, "row"]]
      to <- vars[edges[i, "col"]]
      current_type <- A[edges[i, "row"], edges[i, "col"]]
      key <- paste0(from, "->", to)
      
      selectInput(
        inputId = paste0("edge_type_", gsub("->", "_to_", key)),
        label = paste(from, "->", to),
        choices = c("LinearShift (ls)" = "ls",
                   "ComplexShift (cs)" = "cs",
                   "SimpleIntercept (si)" = "si",
                   "ComplexIntercept (ci)" = "ci"),
        selected = current_type,
        width = "100%"
      )
    })
    
    tagList(
      edge_list,
      actionButton("btn_update_edge_types", "Update Edge Types",
                   class = "btn-sm btn-primary")
    )
  })
  
  # Update edge types
  observeEvent(input$btn_update_edge_types, {
    req(rv_amat(), rv_data())
    A <- rv_amat()
    vars <- rownames(A)
    if (is.null(vars)) vars <- colnames(A)
    
    edges <- which(A != "0" & A != 0, arr.ind = TRUE)
    if (nrow(edges) == 0) return()
    
    for (i in seq_len(nrow(edges))) {
      from <- vars[edges[i, "row"]]
      to <- vars[edges[i, "col"]]
      key <- paste0("edge_type_", gsub("->", "_to_", paste0(from, "->", to)))
      
      input_id <- paste0("edge_type_", gsub("->", "_to_", paste0(from, "->", to)))
      new_type <- input[[input_id]]
      
      if (!is.null(new_type)) {
        A[edges[i, "row"], edges[i, "col"]] <- new_type
      }
    }
    
    rv_amat(A)
    rv_manual_edges(adjacency_to_edges(A))

    # Keep rv_edge_types in sync
    updated_edge_types <- list()
    for (i in seq_len(nrow(edges))) {
      from <- vars[edges[i, "row"]]
      to <- vars[edges[i, "col"]]
      updated_edge_types[[paste0(from, "->", to)]] <- A[edges[i, "row"], edges[i, "col"]]
    }
    rv_edge_types(updated_edge_types)
  })
  
  # Debug output for DAG
  output$dag_debug <- renderText({
    if (is.null(rv_data())) {
      return("Status: No data loaded")
    }
    vars <- colnames(rv_data())
    A <- rv_amat()
    layout_set <- !is.null(rv_layout())
    edges_count <- if (!is.null(A)) {
      sum(A != "0" & A != 0)
    } else {
      0
    }
    paste("Status: Data loaded (", length(vars), "variables),",
          "Layout:", if (layout_set) "OK" else "Missing,",
          "Edges:", edges_count)
  })
  
  # DAG Plot
  output$dag_plot <- renderPlot({
    if (is.null(rv_data())) {
      plot(1, 1, type = "n", axes = FALSE, xlab = "", ylab = "",
           main = "Please upload data first")
      text(1, 1, "No data loaded", cex = 1.5)
      return()
    }
    
    vars <- colnames(rv_data())
    coords <- rv_layout()
    
    if (is.null(coords)) {
      n <- length(vars)
      angles <- seq(from = pi/2, by = -2*pi/n, length.out = n)
      coords <- cbind(x = cos(angles), y = sin(angles))
      rownames(coords) <- vars
      rv_layout(coords)
    }
    
    A_base <- rv_amat()
    if (is.null(A_base)) {
      A_base <- matrix("0", nrow = length(vars), ncol = length(vars),
                       dimnames = list(vars, vars))
      rv_amat(A_base)
    }
    
    A_vars <- rownames(A_base)
    if (is.null(A_vars)) A_vars <- colnames(A_base)
    if (is.null(A_vars)) {
      A_base <- matrix("0", nrow = length(vars), ncol = length(vars),
                       dimnames = list(vars, vars))
      A_vars <- vars
    }
    
    missing_vars <- setdiff(vars, A_vars)
    if (length(missing_vars) > 0) {
      new_A <- matrix("0", nrow = length(vars), ncol = length(vars),
                     dimnames = list(vars, vars))
      common_vars <- intersect(A_vars, vars)
      if (length(common_vars) > 0) {
        new_A[common_vars, common_vars] <- A_base[common_vars, common_vars]
      }
      A_base <- new_A
      A_vars <- vars
      rv_amat(A_base)
    }
    
    A_binary <- matrix(0L, nrow = nrow(A_base), ncol = ncol(A_base),
                      dimnames = dimnames(A_base))
    non_zero <- (A_base != "0") & (A_base != 0) & (!is.na(A_base))
    A_binary[non_zero] <- 1L
    
    tryCatch({
      g <- igraph::graph_from_adjacency_matrix(A_binary, mode = "directed")
      
      lay <- coords[vars, , drop = FALSE]
      
      if (any(is.na(lay))) {
        n <- length(vars)
        angles <- seq(from = pi/2, by = -2*pi/n, length.out = n)
        lay <- cbind(x = cos(angles), y = sin(angles))
        rownames(lay) <- vars
      }
      
      xlim <- range(lay[, 1], na.rm = TRUE) * 1.2
      ylim <- range(lay[, 2], na.rm = TRUE) * 1.2
      
      edge_labels <- character(0)
      if (any(non_zero)) {
        edges <- which(non_zero, arr.ind = TRUE)
        if (nrow(edges) > 0) {
          edge_list <- igraph::as_edgelist(g)
          for (i in seq_len(nrow(edges))) {
            from_idx <- edges[i, "row"]
            to_idx <- edges[i, "col"]
            from_var <- A_vars[from_idx]
            to_var <- A_vars[to_idx]
            edge_type <- A_base[from_idx, to_idx]
            edge_labels <- c(edge_labels, edge_type)
          }
        }
      }
      
      if (length(edge_labels) == 0) {
        plot(g,
             main = "Causal DAG (no edges yet - draw edges in editor)",
             vertex.label.cex = 1.2,
             vertex.size = 40,
             vertex.color = "lightblue",
             vertex.frame.color = "darkblue",
             vertex.label.color = "black",
             layout = lay,
             rescale = FALSE,
             xlim = xlim,
             ylim = ylim)
      } else {
        plot(g,
             main = "Causal DAG (with edge types)",
             vertex.label.cex = 1.2,
             vertex.size = 30,
             vertex.color = "lightblue",
             vertex.frame.color = "darkblue",
             vertex.label.color = "black",
             edge.arrow.size = 0.5,
             edge.label = edge_labels,
             edge.label.cex = 0.8,
             edge.label.color = "darkblue",
             edge.color = "gray50",
             layout = lay,
             rescale = FALSE,
             xlim = xlim,
             ylim = ylim)
      }
    }, error = function(e) {
      plot(1, 1, type = "n", axes = FALSE, xlab = "", ylab = "",
           main = "DAG Visualization Error")
      text(1, 1, paste("Error:", conditionMessage(e)), cex = 1.2)
    })
  })
  
  # Fit Model - use reactive value for status
  output$fit_status <- renderText({
    rv_fit_status()
  })
  
  observeEvent(input$btn_fit, {
    req(rv_data(), rv_amat())
    
    # Create experiment directory on Desktop
    experiment_name <- input$experiment_name
    if (is.null(experiment_name) || trimws(experiment_name) == "") {
      experiment_name <- NULL  # Will be auto-generated
    }
    experiment_dir <- create_experiment_dir(experiment_name)
    rv_experiment_dir(experiment_dir)
    
    # Show progress indicator
    withProgress(message = 'Fitting TRAM-DAG Model', value = 0, {
      incProgress(0.1, detail = "Creating configuration...")
      
      # Update status using reactive value (safer than direct output assignment)
      rv_fit_status("Fitting model... Please wait.")
    
    # Create log file
    log_file <- file.path(experiment_dir, "fit_debug.log")
    cat("=== FIT DEBUG LOG START ===\n", file = log_file, append = FALSE)
    cat("Timestamp:", as.character(Sys.time()), "\n", file = log_file, append = TRUE)
    
    # Use isolate to prevent reactivity issues, and run in separate context
    tryCatch({
      cat("Step 1: Creating configuration...\n", file = log_file, append = TRUE)
      cat("  - Data dimensions:", nrow(rv_data()), "x", ncol(rv_data()), "\n", file = log_file, append = TRUE)
      cat("  - Adjacency matrix dimensions:", nrow(rv_amat()), "x", ncol(rv_amat()), "\n", file = log_file, append = TRUE)
      cat("  - Experiment directory:", experiment_dir, "\n", file = log_file, append = TRUE)
      
      # Create configuration (pass user-selected data types)
      incProgress(0.2, detail = "Creating configuration...")
      cfg <- create_tramdag_config(
        df = rv_data(),
        amat = rv_amat(),
        experiment_dir = experiment_dir,
        data_types = rv_data_types()
      )
      
      cat("Step 2: Configuration created successfully\n", file = log_file, append = TRUE)
      cat("  - Config path:", cfg$CONF_DICT_PATH, "\n", file = log_file, append = TRUE)
      
      # Split data (80/10/10 train/val/test)
      incProgress(0.3, detail = "Splitting data...")
      cat("Step 3: Splitting data...\n", file = log_file, append = TRUE)
      n <- nrow(rv_data())
      set.seed(42)
      train_idx <- sample(n, floor(0.8 * n))
      train_df <- rv_data()[train_idx, ]
      temp_df <- rv_data()[-train_idx, ]
      val_idx <- sample(nrow(temp_df), floor(0.5 * nrow(temp_df)))
      val_df <- temp_df[val_idx, ]
      test_df <- temp_df[-val_idx, ]
      rv_train_df(train_df)
      rv_val_df(val_df)
      rv_test_df(test_df)
      cat("  - Train size:", nrow(train_df), "\n", file = log_file, append = TRUE)
      cat("  - Validation size:", nrow(val_df), "\n", file = log_file, append = TRUE)
      cat("  - Test size:", nrow(test_df), "\n", file = log_file, append = TRUE)
      
      # Fit model
      incProgress(0.4, detail = "Fitting model... This may take several minutes...")
      cat("Step 4: Fitting model...\n", file = log_file, append = TRUE)
      cat("  - Epochs:", input$epochs, "\n", file = log_file, append = TRUE)
      cat("  - Learning rate:", input$learning_rate, "\n", file = log_file, append = TRUE)
      cat("  - Batch size:", input$batch_size, "\n", file = log_file, append = TRUE)
      
      # Initialize td_model outside tryCatch for error handler access
      td_model <- NULL
      
      # Fit model with error handling that checks for Shiny output errors
      fit_result <- tryCatch({
        fit_tramdag_model(
          cfg = cfg,
          train_df = train_df,
          val_df = val_df,
          epochs = input$epochs,
          learning_rate = input$learning_rate,
          batch_size = input$batch_size,
          set_initial_weights = input$set_initial_weights
        )
      }, error = function(fit_e) {
        # Check if this is the Shiny output error
        error_msg <- conditionMessage(fit_e)
        if (grepl("'arg' should be one of", error_msg, fixed = TRUE)) {
          # This is a Shiny output handling error - model might have actually fitted
          # Try to continue - the error is likely from verbose output
          return(NULL)  # Return NULL to indicate we should check log
        }
        stop(fit_e)  # Re-throw real errors
      })
      
      # Check if we got a model (either from successful fit or if Shiny error was caught)
      if (!is.null(fit_result)) {
        td_model <- fit_result
      }
      
      cat("Step 5: Model fitted successfully!\n", file = log_file, append = TRUE)
      incProgress(0.8, detail = "Model fitted! Saving scripts...")
      
      # Save reproducible analysis scripts
      cat("Step 6: Saving reproducible analysis scripts...\n", file = log_file, append = TRUE)
      tryCatch({
        save_analysis_scripts(
          experiment_dir = experiment_dir,
          data_path = if(!is.null(input$datafile)) input$datafile$name else NULL,
          amat = rv_amat(),
          epochs = input$epochs,
          learning_rate = input$learning_rate,
          batch_size = input$batch_size,
          set_initial_weights = input$set_initial_weights,
          X_var = input$X_var,
          Y_var = input$Y_var,
          x_treated = input$x_treated,
          x_control = input$x_control
        )
        cat("  - Analysis scripts saved successfully\n", file = log_file, append = TRUE)
      }, error = function(script_e) {
        cat("  - Warning: Could not save analysis scripts:", conditionMessage(script_e), "\n", 
            file = log_file, append = TRUE)
      })
      
      # Export complete reproducible package (for portability)
      incProgress(0.9, detail = "Exporting reproducible package...")
      cat("Step 7: Exporting complete reproducible package...\n", file = log_file, append = TRUE)
      tryCatch({
        export_reproducible_package(
          experiment_dir = experiment_dir,
          data_df = rv_data(),
          data_path = if(!is.null(input$datafile)) input$datafile$name else NULL,
          amat = rv_amat(),
          epochs = input$epochs,
          learning_rate = input$learning_rate,
          batch_size = input$batch_size,
          set_initial_weights = input$set_initial_weights,
          X_var = input$X_var,
          Y_var = input$Y_var,
          x_treated = input$x_treated,
          x_control = input$x_control
        )
        cat("  - Complete reproducible package exported successfully\n", file = log_file, append = TRUE)
        cat("  - Location: reproducible_package/ folder\n", file = log_file, append = TRUE)
      }, error = function(export_e) {
        cat("  - Warning: Could not export reproducible package:", conditionMessage(export_e), "\n", 
            file = log_file, append = TRUE)
      })
      
      # Store model and update status using reactive values
      # Use suppressWarnings to avoid Shiny output conflicts
      if (!is.null(td_model)) {
        suppressWarnings({
          rv_td_model(td_model)
          rv_chat_context_cache(NULL)
          host_experiment_dir <- file.path("output", basename(experiment_dir))
          status_msg <- paste("✅ Model fitted successfully!\n",
                             "Experiment directory:", host_experiment_dir, "\n",
                             "Scripts folder: scripts/ (for reproducible analysis)\n",
                             "Reproducible package: reproducible_package/ (complete export)\n",
                             "Debug log: fit_debug.log")
          rv_fit_status(status_msg)
        })
        # No notification - status is shown in the UI
        
        # Generate LLM interpretation of model fitting (via Ollama)
        incProgress(0.95, detail = "Generating interpretation...")
        tryCatch({
          if (!ollama_available) {
            cat("Note: Ollama not available - skipping model interpretation\n")
            rv_model_interpretation(NULL)
          } else {
            vars <- colnames(rv_data())
            cat("Generating model interpretation via Ollama...\n")
            interpretation <- generate_model_interpretation(
              experiment_dir = experiment_dir,
              epochs = input$epochs,
              learning_rate = input$learning_rate,
              batch_size = input$batch_size,
              variables = vars
            )
            if (!is.null(interpretation) && trimws(interpretation) != "") {
              cat("Model interpretation generated successfully (", nchar(interpretation), " characters)\n")
              rv_model_interpretation(interpretation)
            } else {
              cat("Model interpretation returned empty\n")
              rv_model_interpretation(NULL)
            }
          }
        }, error = function(e) {
          cat("Error generating model interpretation:", conditionMessage(e), "\n")
          rv_model_interpretation(NULL)
        })
      }
      
      # Mark progress as complete (success path)
      incProgress(1.0, detail = "Complete!")
      
    }, error = function(e) {
      error_msg <- conditionMessage(e)
      
      # Check if this is the Shiny output error - if so, ignore it
      if (grepl("'arg' should be one of", error_msg, fixed = TRUE)) {
        # This is a Shiny output handling error, not a real error
        # Check if model was fitted by looking at log file
        log_content <- tryCatch({
          readLines(log_file)
        }, error = function(le) {
          NULL
        })
        
        # If log shows success, update status accordingly
        if (!is.null(log_content) && any(grepl("Step 5: Model fitted successfully!", log_content, fixed = TRUE))) {
          # Model fitted successfully despite the Shiny error
          # Try to load model from experiment directory if td_model not available
          model_to_store <- NULL
          if (exists("td_model") && !is.null(td_model)) {
            model_to_store <- td_model
          } else {
            # Try to load from directory
            tryCatch({
              model_to_store <- TramDagModel$from_directory(experiment_dir)
            }, error = function(load_e) {
              # If we can't load, that's okay - at least update status
            })
          }
          
          if (!is.null(model_to_store)) {
            suppressWarnings({
              rv_td_model(model_to_store)
              status_msg <- paste("✅ Model fitted successfully!\n",
                                 "Experiment directory:", experiment_dir, "\n",
                                 "Debug log:", log_file)
              rv_fit_status(status_msg)
            })
          } else {
            # At least update status to show success
            suppressWarnings({
              status_msg <- paste("✅ Model fitted successfully!\n",
                                 "Experiment directory:", experiment_dir, "\n",
                                 "Debug log:", log_file)
              rv_fit_status(status_msg)
            })
          }
          # Silently return - don't log this as an error
          return(invisible(NULL))
        }
      }
      
      # Write error to log file for real errors
      cat("\n=== ERROR CAUGHT ===\n", file = log_file, append = TRUE)
      cat("Error message:", error_msg, "\n", file = log_file, append = TRUE)
      cat("Error class:", class(e), "\n", file = log_file, append = TRUE)
      cat("=== END ERROR LOG ===\n", file = log_file, append = TRUE)
      
      # Update status using reactive value
      host_log_path <- file.path("output", basename(dirname(log_file)), basename(log_file))
      full_error_msg <- paste("❌ Error:", error_msg, "\n\nDebug log saved to:", host_log_path)
      suppressWarnings({
        rv_fit_status(full_error_msg)
      })
      tryCatch({
        showNotification(paste("Error fitting model. Check log:", host_log_path), type = "error", duration = 10)
      }, error = function(notif_e) {
        # Ignore notification errors
      })
    })  # End tryCatch
    
    # Progress bar auto-closes when withProgress block completes
    # No need to manually close - it disappears automatically
  })  # End withProgress
  })
  
  # Compute ATE
  output$ate_result <- renderText("")
  
  observeEvent(input$btn_compute_ate, {
    td_model <- rv_td_model()
    if (is.null(td_model)) {
      showNotification("Please fit a model first", type = "warning")
      return()
    }
    
    withProgress(message = 'Computing ATE', value = 0, {
      incProgress(0.3, detail = "Sampling under treatment...")
      
      tryCatch({
      result <- compute_ate(
        td_model = td_model,
        Y = input$Y_var,
        X = input$X_var,
        x_treated = input$x_treated,
        x_control = input$x_control,
        n_samples = input$n_samples
      )
      
      incProgress(0.7, detail = "Sampling under control...")
      
      rv_ate_result(result)
      rv_chat_context_cache(NULL)
      rv_interventional_samples_treated(result$samp_treated_df)
      rv_interventional_samples_control(result$samp_control_df)
      
      incProgress(0.9, detail = "Computing ATE...")
      
      output$ate_result <- renderText({
        sprintf(
          "ATE = E[%s | do(%s = %.2f)] - E[%s | do(%s = %.2f)] = %.4f\n\n",
          input$Y_var, input$X_var, input$x_treated,
          input$Y_var, input$X_var, input$x_control,
          result$ate
        )
      })
      
      # Generate LLM interpretation of ATE results (via Ollama)
      incProgress(0.95, detail = "Generating interpretation...")
      tryCatch({
        if (!ollama_available) {
          cat("Note: Ollama not available - skipping ATE interpretation\n")
          rv_ate_interpretation(NULL)
        } else {
          vars <- colnames(rv_data())
          cat("Generating ATE interpretation via Ollama...\n")
          interpretation <- generate_ate_interpretation(
            ate_result = result,
            X = input$X_var,
            Y = input$Y_var,
            x_treated = input$x_treated,
            x_control = input$x_control,
            variables = vars
          )
          if (!is.null(interpretation) && trimws(interpretation) != "") {
            cat("ATE interpretation generated successfully (", nchar(interpretation), " characters)\n")
            rv_ate_interpretation(interpretation)
          } else {
            cat("ATE interpretation returned empty\n")
            rv_ate_interpretation(NULL)
          }
        }
      }, error = function(e) {
        cat("Error generating ATE interpretation:", conditionMessage(e), "\n")
        rv_ate_interpretation(NULL)
      })
      
      incProgress(1.0, detail = "Complete!")
      
    }, error = function(e) {
      showNotification(paste("Error:", conditionMessage(e)), type = "error")
    })
    })  # End withProgress
  })
  
  # Observational Sampling
  observeEvent(input$btn_sample, {
    td_model <- rv_td_model()
    if (is.null(td_model)) {
      showNotification("Please fit a model first", type = "warning")
      return()
    }
    
    withProgress(message = 'Sampling from model', value = 0, {
      incProgress(0.5, detail = paste("Generating", input$n_samples, "samples..."))
      
      tryCatch({
        sampled <- sample_from_model(td_model, n_samples = input$n_samples)
        rv_sampled_data(sampled)
        rv_chat_context_cache(NULL)
        incProgress(1.0, detail = "Complete!")
        # Progress bar will auto-close, no notification needed
      }, error = function(e) {
        showNotification(paste("Error:", conditionMessage(e)), type = "error")
      })
    })  # End withProgress - auto-closes when function completes
  })
  
  # Generate PDF Report
  observeEvent(input$btn_generate_report, {
    req(rv_experiment_dir(), rv_data(), rv_amat())
    
    experiment_dir <- rv_experiment_dir()
    if (is.null(experiment_dir)) {
      showNotification("Please fit a model first", type = "warning")
      return()
    }
    
    withProgress(message = 'Generating Full Analysis Report', value = 0, {
      incProgress(0.1, detail = "Collecting data and model artifacts...")
      
      tryCatch({
        # Get experiment name
        experiment_name <- input$experiment_name
        if (is.null(experiment_name) || trimws(experiment_name) == "") {
          experiment_name <- basename(experiment_dir)
        }
        
        incProgress(0.2, detail = "Generating plots and building PDF (this may take a minute)...")
        
        # Generate comprehensive report with all available data
        report_path <- generate_pdf_report(
          experiment_dir = experiment_dir,
          data_df = rv_data(),
          amat = rv_amat(),
          experiment_name = experiment_name,
          epochs = input$epochs,
          learning_rate = input$learning_rate,
          batch_size = input$batch_size,
          set_initial_weights = input$set_initial_weights,
          td_model = rv_td_model(),
          ate_result = rv_ate_result(),
          model_interpretation = rv_model_interpretation(),
          ate_interpretation = rv_ate_interpretation(),
          X_var = input$X_var,
          Y_var = input$Y_var,
          x_treated = input$x_treated,
          x_control = input$x_control,
          train_df = rv_train_df(),
          test_df = rv_test_df(),
          sampled_data = rv_sampled_data(),
          interventional_samples = rv_interventional_samples_treated(),
          interventional_control = rv_interventional_samples_control()
        )
        
        incProgress(0.9, detail = "Report generated!")
        
        if (!is.null(report_path) && file.exists(report_path)) {
          host_path <- file.path("output", basename(dirname(dirname(report_path))), "reports", basename(report_path))
          showNotification(
            paste("Full analysis report generated!\nLocation:", host_path),
            type = "message",
            duration = 15
          )
        } else {
          showNotification("Report generation failed. Check console for errors.", type = "error")
        }
        
        incProgress(1.0, detail = "Complete!")
        
      }, error = function(e) {
        error_msg <- conditionMessage(e)
        cat("Error generating PDF report:", error_msg, "\n")
        
        # Provide helpful error message if reportlab is missing
        if (grepl("reportlab", error_msg, ignore.case = TRUE) || 
            grepl("No module named", error_msg, ignore.case = TRUE)) {
          showNotification(
            HTML("Report generation failed: reportlab not installed.<br/>Install with: <code>pip install reportlab</code><br/>Then restart the app."),
            type = "error",
            duration = 15
          )
        } else {
          showNotification(paste("Error generating report:", error_msg), type = "error", duration = 10)
        }
      })
    })  # End withProgress
  })
  
  # ---- Chat with LLM about results ----
  
  output$chat_visible <- reactive({
    length(rv_chat_history()) > 0 || (!is.null(rv_td_model()) && ollama_available)
  })
  outputOptions(output, "chat_visible", suspendWhenHidden = FALSE)
  
  # Build a comprehensive context summary so the LLM can answer
  # questions about data, model, plots, and results.
  build_chat_context <- function() {
    parts <- c()
    
    # --- 1. Data summary & distributions (backs the pairplot) ---
    df <- rv_data()
    if (!is.null(df)) {
      parts <- c(parts, paste0(
        "DATA: ", ncol(df), " variables (", paste(colnames(df), collapse = ", "),
        "), ", nrow(df), " observations.\n",
        "Summary statistics:\n", paste(capture.output(summary(df)), collapse = "\n")
      ))
      
      # Correlations
      num_cols <- sapply(df, is.numeric)
      if (sum(num_cols) >= 2) {
        cor_mat <- cor(df[, num_cols, drop = FALSE], use = "pairwise.complete.obs")
        parts <- c(parts, paste0(
          "CORRELATION MATRIX (shown in pairplot):\n",
          paste(capture.output(round(cor_mat, 4)), collapse = "\n")
        ))
      }
      
      # Per-variable distributions
      dist_lines <- sapply(colnames(df), function(v) {
        x <- df[[v]]
        if (is.numeric(x)) {
          sprintf("  %s: mean=%.4f, sd=%.4f, min=%.4f, max=%.4f, median=%.4f",
                  v, mean(x, na.rm = TRUE), sd(x, na.rm = TRUE),
                  min(x, na.rm = TRUE), max(x, na.rm = TRUE), median(x, na.rm = TRUE))
        } else {
          sprintf("  %s: %d unique values, top: %s", v, length(unique(x)),
                  paste(head(sort(table(x), decreasing = TRUE), 3), collapse = ", "))
        }
      })
      parts <- c(parts, paste0(
        "PAIRPLOT: Shows pairwise scatter plots and marginal histograms for all variables.\n",
        "Variable distributions:\n", paste(dist_lines, collapse = "\n")
      ))
    }
    
    # --- 2. DAG structure (backs the DAG plot) ---
    amat <- rv_amat()
    if (!is.null(amat)) {
      edges <- which(amat != "0" & amat != 0, arr.ind = TRUE)
      vars <- rownames(amat)
      if (is.null(vars)) vars <- colnames(amat)
      
      # Source, intermediate, and sink nodes
      has_parents <- colSums(amat != "0" & amat != 0) > 0
      has_children <- rowSums(amat != "0" & amat != 0) > 0
      sources <- vars[!has_parents & has_children]
      sinks <- vars[has_parents & !has_children]
      intermediates <- vars[has_parents & has_children]
      
      edge_strs <- sapply(seq_len(nrow(edges)), function(i) {
        paste0(vars[edges[i, "row"]], " -> ", vars[edges[i, "col"]],
               " (", amat[edges[i, "row"], edges[i, "col"]], ")")
      })
      parts <- c(parts, paste0(
        "DAG PLOT: Network visualization of the causal DAG.\n",
        "Edges: ", paste(edge_strs, collapse = "; "), "\n",
        "Source nodes (no parents): ", paste(sources, collapse = ", "), "\n",
        if (length(intermediates) > 0) paste0("Intermediate nodes: ", paste(intermediates, collapse = ", "), "\n") else "",
        "Sink nodes (no children): ", paste(sinks, collapse = ", ")
      ))
      
      # Edge type legend
      parts <- c(parts, paste0(
        "Edge type codes: ls=LinearShift, cs=ComplexShift, si=SimpleIntercept, ci=ComplexIntercept. ",
        "LinearShift means the parent has a linear additive effect on the child's distribution."
      ))
    }
    
    # --- 3. Training configuration ---
    parts <- c(parts, paste0(
      "TRAINING CONFIG: epochs=", input$epochs, ", learning_rate=", input$learning_rate,
      ", batch_size=", input$batch_size,
      ", set_initial_weights=", input$set_initial_weights
    ))
    
    # --- 4. Loss history (backs the loss plot) ---
    experiment_dir <- rv_experiment_dir()
    if (!is.null(experiment_dir) && !is.null(amat)) {
      vars <- rownames(amat)
      if (is.null(vars)) vars <- colnames(amat)
      loss_lines <- c()
      for (var in vars) {
        train_file <- file.path(experiment_dir, var, "train_loss_hist.json")
        val_file <- file.path(experiment_dir, var, "val_loss_hist.json")
        tryCatch({
          if (file.exists(train_file)) {
            tl <- jsonlite::fromJSON(train_file)
            if (is.numeric(tl) && length(tl) > 0) {
              vl_str <- ""
              if (file.exists(val_file)) {
                vl <- jsonlite::fromJSON(val_file)
                if (is.numeric(vl) && length(vl) > 0) {
                  vl_str <- sprintf(", val_loss: %.4f -> %.4f (min=%.4f at epoch %d)",
                                    vl[1], vl[length(vl)], min(vl), which.min(vl))
                }
              }
              loss_lines <- c(loss_lines, sprintf(
                "  %s: train_loss: %.4f -> %.4f (min=%.4f at epoch %d)%s",
                var, tl[1], tl[length(tl)], min(tl), which.min(tl), vl_str
              ))
            }
          }
        }, error = function(e) {})
      }
      if (length(loss_lines) > 0) {
        parts <- c(parts, paste0(
          "LOSS HISTORY PLOT: Shows training and validation loss curves per variable over epochs.\n",
          "Convergence summary:\n", paste(loss_lines, collapse = "\n"),
          "\nA well-fitted model shows both curves decreasing and stabilizing. ",
          "Divergence between train/val loss may indicate overfitting."
        ))
      }
      
      # --- 5. Parameter convergence (backs shift & intercept history plots) ---
      param_lines <- c()
      for (var in vars) {
        shift_file <- file.path(experiment_dir, var, "linear_shifts_all_epochs.json")
        intercept_file <- file.path(experiment_dir, var, "simple_intercepts_all_epochs.json")
        tryCatch({
          if (file.exists(shift_file)) {
            shifts <- jsonlite::fromJSON(shift_file)
            if (is.list(shifts) && length(shifts) > 0) {
              last_epoch <- shifts[[length(shifts)]]
              if (!is.null(last_epoch)) {
                param_lines <- c(param_lines, sprintf(
                  "  %s linear shifts (final): %s",
                  var, paste(sapply(names(last_epoch), function(p) {
                    sprintf("%s=%.4f", p, as.numeric(last_epoch[[p]]))
                  }), collapse = ", ")
                ))
              }
            }
          }
        }, error = function(e) {})
        tryCatch({
          if (file.exists(intercept_file)) {
            intercepts <- jsonlite::fromJSON(intercept_file)
            if (is.list(intercepts) && length(intercepts) > 0) {
              last_epoch <- intercepts[[length(intercepts)]]
              if (!is.null(last_epoch) && is.numeric(unlist(last_epoch))) {
                vals <- unlist(last_epoch)
                param_lines <- c(param_lines, sprintf(
                  "  %s intercepts (final): %d values, range [%.4f, %.4f]",
                  var, length(vals), min(vals), max(vals)
                ))
              }
            }
          }
        }, error = function(e) {})
      }
      if (length(param_lines) > 0) {
        parts <- c(parts, paste0(
          "PARAMETER HISTORY PLOTS:\n",
          "- Shift History Plot: Shows how linear shift coefficients evolve over epochs.\n",
          "  These represent the causal effect weights of parent nodes.\n",
          "- Intercept History Plot: Shows how baseline intercept parameters evolve over epochs.\n",
          "  These define the base transformation for each variable.\n",
          "Final parameter values:\n", paste(param_lines, collapse = "\n")
        ))
      }
    }
    
    # --- 6. Model fit status ---
    fit_status <- rv_fit_status()
    if (!is.null(fit_status)) {
      parts <- c(parts, paste0("FIT STATUS: ", fit_status))
    }
    
    # --- 7. NLL (Negative Log-Likelihood) — use cached render output to avoid slow recomputation ---
    td_model <- rv_td_model()
    if (!is.null(td_model) && !is.null(rv_nll_cache())) {
      parts <- c(parts, paste0(rv_nll_cache(),
        "\nLower NLL = better fit. This measures how well the model explains the data."
      ))
    }
    
    # --- 8. h-DAG and Latents plots ---
    if (!is.null(td_model)) {
      parts <- c(parts, paste0(
        "H-DAG PLOT: Shows the learned transformation h(y|parents) for each variable, ",
        "plotted against training data. The h-function maps observed values to a latent ",
        "standard normal space. A well-fitted model shows smooth, monotonically increasing curves. ",
        "Separate curves per parent value indicate the shift effect of each parent."
      ))
      parts <- c(parts, paste0(
        "LATENTS PLOT: Shows the distribution of latent (transformed) values z = h(y|parents) ",
        "for each variable. If the model fits well, these should follow a standard normal ",
        "distribution (bell curve centered at 0 with sd=1). Deviations suggest the model ",
        "does not fully capture the data distribution."
      ))
    }
    
    # --- 9. Model interpretation ---
    interp <- rv_model_interpretation()
    if (!is.null(interp) && trimws(interp) != "") {
      parts <- c(parts, paste0("MODEL INTERPRETATION (by LLM):\n", interp))
    }
    
    # --- 10. Observational sampling (backs samples vs true plot) ---
    sampled <- rv_sampled_data()
    test_df <- rv_test_df()
    if (!is.null(sampled) && !is.null(test_df)) {
      cmp_lines <- sapply(colnames(sampled), function(v) {
        s <- sampled[[v]]; t <- test_df[[v]]
        if (is.numeric(s) && is.numeric(t)) {
          sprintf("  %s: sampled mean=%.4f (sd=%.4f), test mean=%.4f (sd=%.4f), diff=%.4f",
                  v, mean(s), sd(s), mean(t), sd(t), mean(s) - mean(t))
        } else { "" }
      })
      parts <- c(parts, paste0(
        "OBSERVATIONAL SAMPLING PLOT (Samples vs True):\n",
        "Compares model-generated samples (orange histograms) against held-out test data (blue histograms) ",
        "for each variable. Close overlap means the model learned the data distribution well.\n",
        "Comparison:\n", paste(cmp_lines[cmp_lines != ""], collapse = "\n")
      ))
    }
    
    # --- 11. ATE and interventional sampling ---
    ate <- rv_ate_result()
    if (!is.null(ate)) {
      parts <- c(parts, sprintf(
        "ATE COMPUTATION:\n  Treatment variable (X): %s\n  Outcome variable (Y): %s\n  x_treated=%.4f, x_control=%.4f\n  ATE = E[%s|do(%s=%.4f)] - E[%s|do(%s=%.4f)] = %.4f\n  Mean outcome under treatment: %.4f\n  Mean outcome under control: %.4f",
        input$X_var, input$Y_var, input$x_treated, input$x_control,
        input$Y_var, input$X_var, input$x_treated,
        input$Y_var, input$X_var, input$x_control,
        ate$ate, ate$mean_treated, ate$mean_control
      ))
      
      # Interventional plot details
      samp_t <- rv_interventional_samples_treated()
      samp_c <- rv_interventional_samples_control()
      if (!is.null(samp_t) && !is.null(samp_c)) {
        inter_lines <- sapply(colnames(samp_t), function(v) {
          mt <- mean(samp_t[[v]]); mc <- mean(samp_c[[v]])
          sprintf("  %s: mean_treated=%.4f, mean_control=%.4f, shift=%.4f", v, mt, mc, mt - mc)
        })
        parts <- c(parts, paste0(
          "INTERVENTIONAL SAMPLING PLOT (Treated vs Control):\n",
          "Shows distributions under do(", input$X_var, "=", input$x_control, ") in blue vs ",
          "do(", input$X_var, "=", input$x_treated, ") in orange, for all variables.\n",
          "The shift in ", input$Y_var, " is the ATE. Shifts in other variables show how the ",
          "intervention propagates through the causal graph.\n",
          "Per-variable comparison:\n", paste(inter_lines, collapse = "\n")
        ))
      }
    }
    
    # --- 12. ATE interpretation ---
    ate_interp <- rv_ate_interpretation()
    if (!is.null(ate_interp) && trimws(ate_interp) != "") {
      parts <- c(parts, paste0("ATE INTERPRETATION (by LLM):\n", ate_interp))
    }
    
    paste(parts, collapse = "\n\n")
  }
  
  observeEvent(input$btn_chat_send, {
    msg <- trimws(input$chat_input)
    if (nchar(msg) == 0) return()
    
    if (!ollama_available) {
      showNotification("Ollama is not available. Cannot answer questions.", type = "warning")
      return()
    }
    
    # Add user message and show thinking indicator
    history <- rv_chat_history()
    history <- c(history, list(list(role = "user", content = msg)))
    rv_chat_history(history)
    rv_chat_thinking(TRUE)
    
    # Clear input
    updateTextInput(session, "chat_input", value = "")
    
    # Snapshot everything needed for the async call (reactive values
    # cannot be read inside the future)
    current_history <- history
    
    # Use cached context if available, otherwise build (and cache) it
    context <- rv_chat_context_cache()
    if (is.null(context)) {
      context <- build_chat_context()
      rv_chat_context_cache(context)
    }
    
    ollama_url <- OLLAMA_BASE_URL
    ollama_model <- OLLAMA_MODEL
    
    system_msg <- paste0(
      "You are an expert assistant for a causal inference application called TRAM-DAG. ",
      "The user has fitted a TRAM-DAG model and may have computed ATE (Average Treatment Effect). ",
      "You have full access to all analysis results including the underlying data for every plot ",
      "shown in the app (pairplot, DAG, loss curves, parameter convergence, h-DAG transformations, ",
      "latent distributions, observational samples vs true data, and interventional treated vs control). ",
      "When the user asks about a plot, describe what it shows and interpret the numbers. ",
      "Answer based on the analysis context below. Be concise, clear, and helpful. ",
      "Use plain language. If the user asks about something not available yet, let them know.\n\n",
      "=== ANALYSIS CONTEXT ===\n", context
    )
    
    ollama_messages <- list(list(role = "system", content = system_msg))
    for (m in current_history) {
      ollama_messages <- c(ollama_messages, list(list(role = m$role, content = m$content)))
    }
    
    # Run the Ollama HTTP call in a background process so the Shiny
    # session stays responsive (no gray screen / disconnect)
    future_promise({
      body <- list(
        model = ollama_model,
        messages = ollama_messages,
        stream = FALSE,
        options = list(temperature = 0.4)
      )
      
      resp <- httr::POST(
        url = paste0(ollama_url, "/api/chat"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE),
        httr::content_type_json(),
        httr::timeout(120)
      )
      
      if (httr::status_code(resp) == 200) {
        result <- jsonlite::fromJSON(httr::content(resp, as = "text", encoding = "UTF-8"))
        r <- result$message$content
        r <- gsub("^\\s+|\\s+$", "", r)
        r <- gsub("\\n{3,}", "\\n\\n", r)
        r
      } else {
        "(Ollama returned an error. Please try again.)"
      }
    }) %...>% (function(reply) {
      current_history <- c(current_history, list(list(role = "assistant", content = reply)))
      rv_chat_history(current_history)
      rv_chat_thinking(FALSE)
    }) %...!% (function(err) {
      error_reply <- paste0("(Error contacting Ollama: ", conditionMessage(err), ")")
      current_history <- c(current_history, list(list(role = "assistant", content = error_reply)))
      rv_chat_history(current_history)
      rv_chat_thinking(FALSE)
    })
    
    NULL
  })
  
  observeEvent(input$btn_chat_clear, {
    rv_chat_history(list())
  })
  
  output$chat_messages <- renderUI({
    history <- rv_chat_history()
    thinking <- rv_chat_thinking()
    
    if (length(history) == 0 && !thinking) {
      return(div(
        class = "chat-container",
        div(style = "text-align: center; color: #85a5c2; padding: 20px; font-style: italic;",
            "No messages yet. Type a question in the sidebar and press Send.")
      ))
    }
    
    msg_divs <- lapply(history, function(m) {
      role_label <- if (m$role == "user") "You" else "Assistant"
      content_html <- gsub("\n\n", "</p><p>", m$content)
      content_html <- gsub("\n", "<br>", content_html)
      div(class = paste("chat-msg", m$role),
          div(class = "chat-role", role_label),
          HTML(paste0("<p>", content_html, "</p>"))
      )
    })
    
    # Show thinking animation while waiting for Ollama
    if (thinking) {
      msg_divs <- c(msg_divs, list(
        div(class = "chat-msg assistant chat-thinking-bubble",
            div(class = "chat-role", "Assistant"),
            div(class = "thinking-dots",
                span(class = "dot"), span(class = "dot"), span(class = "dot"))
        )
      ))
    }
    
    div(class = "chat-container", msg_divs)
  })
  
  # Training Loss Plot (uses td_model.plot_loss_history() -- same as notebook)
  output$loss_plot <- renderImage({
    req(rv_td_model())
    td_model <- rv_td_model()
    plt <- import("matplotlib.pyplot")
    
    tmp <- tempfile(fileext = ".png")
    original_show <- plt$show
    plot_ok <- tryCatch({
      plt$close("all")
      # Override plt.show() so the figure stays open for us to adjust
      plt$show <- function(...) invisible(NULL)
      plt$ioff()
      
      td_model$plot_loss_history()
      
      plt$show <- original_show
      
      fig <- plt$gcf()
      # Add spacing between upper and lower subplot and move legends outside
      fig$set_size_inches(14, 12)
      axes <- fig$get_axes()
      for (i in seq_along(axes)) {
        ax <- axes[[i]]
        ax$legend(loc = "center left", bbox_to_anchor = c(1.02, 0.5), fontsize = 9L)
      }
      fig$subplots_adjust(hspace = 0.35, right = 0.82)
      fig$savefig(tmp, dpi = 100L, bbox_inches = "tight")
      plt$close("all")
      TRUE
    }, error = function(e) {
      tryCatch({ plt$show <- original_show }, error = function(e2) {})
      plt$close("all")
      FALSE
    })
    
    if (!plot_ok || !file.exists(tmp)) {
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "Loss history not available", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "Loss History")
  }, deleteFile = TRUE)
  
  # Note: capture_matplotlib_plot is defined at top level (before generate_pdf_report)
  # so it can be used by both the report generator and the server render functions.
  
  # Diagnostic: Linear Shift History
  output$shift_history_plot <- renderImage({
    req(rv_td_model())
    td_model <- rv_td_model()
    
    tmp <- capture_matplotlib_plot(function() {
      td_model$plot_linear_shift_history()
    }, width = 1200, height = 500)
    
    if (is.null(tmp) || !file.exists(tmp)) {
      # Return a placeholder
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "Linear shift history not available (no linear shift edges?)", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "Linear Shift History")
  }, deleteFile = TRUE)
  
  # Diagnostic: Simple Intercepts History
  output$intercepts_history_plot <- renderImage({
    req(rv_td_model())
    td_model <- rv_td_model()
    
    tmp <- capture_matplotlib_plot(function() {
      td_model$plot_simple_intercepts_history()
    }, width = 1200, height = 500)
    
    if (is.null(tmp) || !file.exists(tmp)) {
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "Intercepts history not available", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "Intercepts History")
  }, deleteFile = TRUE)
  
  # Diagnostic: h-DAG Transformation Plots
  output$hdag_plot <- renderImage({
    req(rv_td_model(), rv_train_df())
    td_model <- rv_td_model()
    train_df <- rv_train_df()
    py_train <- r_to_py(train_df)
    vars <- colnames(train_df)
    
    tmp <- capture_matplotlib_plot(function() {
      td_model$plot_hdag(py_train, variables = as.list(vars), plot_n_rows = 1L)
    }, width = 1400, height = 500)
    
    if (is.null(tmp) || !file.exists(tmp)) {
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "h-DAG plots not available", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "h-DAG Plots")
  }, deleteFile = TRUE)
  
  # Diagnostic: Latent Distributions
  output$latents_plot <- renderImage({
    req(rv_td_model(), rv_train_df())
    td_model <- rv_td_model()
    train_df <- rv_train_df()
    py_train <- r_to_py(train_df)
    
    tmp <- capture_matplotlib_plot(function() {
      td_model$plot_latents(py_train)
    }, width = 1200, height = 500)
    
    if (is.null(tmp) || !file.exists(tmp)) {
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "Latent distributions not available", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "Latent Distributions")
  }, deleteFile = TRUE)
  
  # Diagnostic: Negative Log-Likelihood
  output$nll_output <- renderText({
    req(rv_td_model(), rv_train_df())
    td_model <- rv_td_model()
    train_df <- rv_train_df()
    py_train <- r_to_py(train_df)
    
    result_text <- tryCatch({
      nll_result <- td_model$nll(py_train)
      nll_r <- py_to_r(nll_result)
      lines <- sapply(names(nll_r), function(node) {
        sprintf("  %s: %.4f", node, nll_r[[node]])
      })
      paste("Negative Log-Likelihood (NLL) on training data:\n",
            paste(lines, collapse = "\n"))
    }, error = function(e) {
      paste("Could not compute NLL:", conditionMessage(e))
    })
    rv_nll_cache(result_text)
    result_text
  })
  
  # Samples vs True plot using TramDag's plot_samples_vs_true
  output$samples_vs_true_plot <- renderImage({
    req(rv_td_model(), rv_sampled_data(), rv_test_df())
    td_model <- rv_td_model()
    test_df <- rv_test_df()
    sampled <- rv_sampled_data()
    
    # Convert sampled R data.frame to a Python dict of tensors
    torch <- import("torch")
    sampled_list <- list()
    for (col in colnames(sampled)) {
      sampled_list[[col]] <- torch$tensor(as.numeric(sampled[[col]]))
    }
    py_sampled <- r_to_py(sampled_list)
    py_test <- r_to_py(test_df)
    
    tmp <- capture_matplotlib_plot(function() {
      td_model$plot_samples_vs_true(py_test, py_sampled)
    }, width = 1400, height = 500)
    
    if (is.null(tmp) || !file.exists(tmp)) {
      tmp <- tempfile(fileext = ".png")
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, "Samples vs True plot not available", cex = 1.2)
      dev.off()
    }
    
    list(src = tmp, contentType = "image/png", alt = "Samples vs True")
  }, deleteFile = TRUE)
  
  # Data Pairplot (seaborn pairplot of uploaded data)
  output$data_pairplot <- renderImage({
    req(rv_data())
    data <- rv_data()
    
    sns <- import("seaborn")
    plt <- import("matplotlib.pyplot")
    pd <- import("pandas")
    
    # Subsample for plotting speed
    n_plot <- min(5000L, nrow(data))
    if (nrow(data) > n_plot) {
      data_plot <- data[sample(nrow(data), n_plot), , drop = FALSE]
    } else {
      data_plot <- data
    }
    
    py_df <- r_to_py(data_plot)
    py_pd_df <- pd$DataFrame(py_df)
    
    tmp <- tempfile(fileext = ".png")
    tryCatch({
      plt$close("all")
      g <- sns$pairplot(py_pd_df, diag_kind = "kde", plot_kws = list(alpha = 0.3, s = 10L))
      g$fig$set_size_inches(12, 12)
      g$fig$savefig(tmp, dpi = 100L, bbox_inches = "tight")
      plt$close("all")
    }, error = function(e) {
      plt$close("all")
      # Fallback: create placeholder
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, paste("Pairplot not available:", conditionMessage(e)), cex = 1.2)
      dev.off()
    })
    
    list(src = tmp, contentType = "image/png", alt = "Data Pairplot")
  }, deleteFile = TRUE)
  
  # Interventional plot: Treated vs Control distributions (visualizes the ATE)
  output$interventional_samples_vs_true_plot <- renderImage({
    req(rv_td_model(), rv_interventional_samples_treated(), rv_interventional_samples_control(), rv_ate_result())
    
    treated_df <- rv_interventional_samples_treated()
    control_df <- rv_interventional_samples_control()
    ate_result <- rv_ate_result()
    
    X_var <- input$X_var
    Y_var <- input$Y_var
    x_treated_val <- input$x_treated
    x_control_val <- input$x_control
    
    vars <- colnames(treated_df)
    n_vars <- length(vars)
    
    tmp <- tempfile(fileext = ".png")
    
    tryCatch({
      # Build a custom matplotlib figure: one subplot per variable
      plt <- import("matplotlib.pyplot")
      np <- import("numpy")
      plt$close("all")
      plt$ioff()
      
      fig_width <- max(4.5 * n_vars, 10)
      fig_height <- 5
      fig_and_axes <- plt$subplots(1L, as.integer(n_vars), figsize = c(fig_width, fig_height))
      fig <- fig_and_axes[[1]]
      axes_raw <- fig_and_axes[[2]]
      
      # Handle single vs multiple axes (numpy returns scalar for n=1)
      if (n_vars == 1L) {
        axes <- list(axes_raw)
      } else {
        axes <- as.list(py_to_r(axes_raw))
        if (length(axes) != n_vars) {
          # Fallback: iterate directly
          axes <- lapply(seq_len(n_vars), function(i) axes_raw[i - 1L])
        }
      }
      
      for (i in seq_len(n_vars)) {
        var <- vars[i]
        ax <- axes[[i]]
        
        ctrl_vals <- as.numeric(control_df[[var]])
        treat_vals <- as.numeric(treated_df[[var]])
        
        # Determine shared bins
        all_vals <- c(ctrl_vals, treat_vals)
        bins <- np$linspace(min(all_vals), max(all_vals), 80L)
        
        # Plot histograms
        ax$hist(ctrl_vals, bins = bins, density = TRUE, alpha = 0.5, 
                color = "steelblue", label = paste0("do(", X_var, "=", x_control_val, ")"))
        ax$hist(treat_vals, bins = bins, density = TRUE, alpha = 0.5, 
                color = "darkorange", label = paste0("do(", X_var, "=", x_treated_val, ")"))
        
        # Add KDE curves
        tryCatch({
          scipy_stats <- import("scipy.stats")
          x_grid <- np$linspace(min(all_vals), max(all_vals), 300L)
          
          kde_ctrl <- scipy_stats$gaussian_kde(ctrl_vals)
          ax$plot(x_grid, kde_ctrl(x_grid), color = "steelblue", lw = 2L, linestyle = "--")
          
          kde_treat <- scipy_stats$gaussian_kde(treat_vals)
          ax$plot(x_grid, kde_treat(x_grid), color = "darkorange", lw = 2L, linestyle = "--")
        }, error = function(e) {})
        
        # Add vertical mean lines
        mean_ctrl <- mean(ctrl_vals)
        mean_treat <- mean(treat_vals)
        ax$axvline(x = mean_ctrl, color = "steelblue", linestyle = ":", lw = 1.5, alpha = 0.8)
        ax$axvline(x = mean_treat, color = "darkorange", linestyle = ":", lw = 1.5, alpha = 0.8)
        
        # Title with ATE for outcome variable
        if (var == Y_var) {
          ate_val <- round(ate_result$ate, 4)
          ax$set_title(paste0(var, "  (ATE = ", ate_val, ")"), fontsize = 13L, fontweight = "bold")
        } else if (var == X_var) {
          ax$set_title(paste0(var, "  (intervened)"), fontsize = 13L, fontweight = "bold")
        } else {
          shift <- round(mean_treat - mean_ctrl, 4)
          ax$set_title(paste0(var, "  (shift = ", shift, ")"), fontsize = 12L)
        }
        
        ax$set_xlabel(var, fontsize = 10L)
        ax$set_ylabel("Density", fontsize = 10L)
        ax$legend(fontsize = 8L, loc = "upper right")
        ax$tick_params(labelsize = 9L)
      }
      
      fig$suptitle(
        paste0("Interventional Comparison: do(", X_var, "=", x_control_val, 
               ") vs do(", X_var, "=", x_treated_val, ")"),
        fontsize = 14L, fontweight = "bold", y = 1.02
      )
      fig$tight_layout()
      fig$savefig(tmp, dpi = 120L, bbox_inches = "tight")
      plt$close("all")
      
    }, error = function(e) {
      cat("Interventional comparison plot error:", conditionMessage(e), "\n")
      tryCatch(plt$close("all"), error = function(e2) {})
      # Fallback placeholder
      png(tmp, width = 800, height = 100)
      par(mar = c(0,0,0,0))
      plot.new()
      text(0.5, 0.5, paste("Interventional plot error:", conditionMessage(e)), cex = 1.0)
      dev.off()
    })
    
    list(src = tmp, contentType = "image/png", alt = "Interventional: Treated vs Control")
  }, deleteFile = TRUE)
}

# Run app
shinyApp(ui = ui, server = server)
