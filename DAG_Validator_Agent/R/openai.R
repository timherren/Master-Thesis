# ============================================================
# R/openai.R — LLM API helpers (Ollama, OpenAI-compatible)
# ============================================================

# ---- Configuration defaults ----
.ollama_base_url <- function() {
  Sys.getenv("OLLAMA_HOST", unset = "http://localhost:11434")
}
.ollama_default_model <- function() {
  Sys.getenv("OLLAMA_MODEL", unset = "llama3.2:latest")
}
.ollama_dag_model <- function() {
  Sys.getenv("OLLAMA_DAG_MODEL", unset = "llama3.2:latest")
}

# ---- Low-level helpers ----

#' Ping the Ollama server once
.ollama_available <- function(base_url = .ollama_base_url()) {
  tryCatch({
    resp <- httr2::request(base_url) |>
      httr2::req_timeout(5) |>
      httr2::req_perform()
    httr2::resp_status(resp) == 200
  }, error = function(e) FALSE)
}

#' Wait for the Ollama server to become reachable (retry loop).
#' Essential for Docker Compose where the Ollama container may
#' still be booting when the app container starts.
#'
#' @param max_attempts Maximum number of connection attempts.
#' @param delay_s      Seconds between attempts.
#' @return TRUE if Ollama became reachable, FALSE otherwise.
.ollama_wait <- function(max_attempts = 30, delay_s = 2) {
  base_url <- .ollama_base_url()
  for (i in seq_len(max_attempts)) {
    if (.ollama_available(base_url)) {
      message("[ollama] Server reachable at ", base_url)
      return(TRUE)
    }
    message(sprintf("[ollama] Waiting for server at %s (attempt %d/%d) ...",
                    base_url, i, max_attempts))
    Sys.sleep(delay_s)
  }
  message("[ollama] Server not reachable after ", max_attempts, " attempts.")
  FALSE
}

#' List model names that Ollama has locally
.ollama_list_models <- function(base_url = .ollama_base_url()) {
  tryCatch({
    resp <- httr2::request(paste0(base_url, "/api/tags")) |>
      httr2::req_timeout(10) |>
      httr2::req_perform()
    j <- httr2::resp_body_json(resp, simplifyVector = FALSE)
    vapply(j$models, function(m) m$name, character(1))
  }, error = function(e) character(0))
}

#' Pull a model from the Ollama library (blocking).
#' @return TRUE on success, FALSE on failure.
.ollama_pull_model <- function(model = .ollama_default_model(),
                               base_url = .ollama_base_url()) {
  message("[ollama] Pulling model '", model, "' — this may take a few minutes ...")
  tryCatch({
    resp <- httr2::request(paste0(base_url, "/api/pull")) |>
      httr2::req_headers(`Content-Type` = "application/json") |>
      httr2::req_body_json(list(name = model, stream = FALSE)) |>
      httr2::req_timeout(1800) |>
      httr2::req_perform()
    j <- httr2::resp_body_json(resp, simplifyVector = FALSE)
    ok <- identical(j$status, "success")
    if (ok) message("[ollama] Model '", model, "' pulled successfully.")
    else    message("[ollama] Pull response: ", j$status)
    ok
  }, error = function(e) {
    message("[ollama] Pull failed: ", conditionMessage(e))
    FALSE
  })
}

# ---- Startup orchestrator ----

#' Ensure Ollama is running and the configured model is available.
#' Called once at app startup. Stores result in a global variable
#' so the UI can display status.
#'
#' @return A list with components: ok (logical), model (character), host (character), message (character).
ollama_ensure_ready <- function() {
  summary_model <- .ollama_default_model()
  dag_model     <- .ollama_dag_model()
  base_url      <- .ollama_base_url()

  # Step 1: Wait for server
  if (!.ollama_wait()) {
    return(list(ok = FALSE, model = summary_model, host = base_url,
                message = paste0("Could not reach Ollama at ", base_url,
                                 ". Start it with: ollama serve")))
  }

  # Step 2: Check / pull both models
  local_models <- .ollama_list_models(base_url)

  for (model in unique(c(summary_model, dag_model))) {
    has_model <- any(
      local_models == model |
      grepl(paste0("^", gsub(":", ":", model)), local_models)
    )
    if (!has_model) {
      message("[ollama] Model '", model, "' not found locally. ",
              "Available models: ",
              if (length(local_models)) paste(local_models, collapse = ", ") else "<none>")
      .ollama_pull_model(model, base_url)
    }
  }

  msg <- paste0("Ollama ready — summary: ", summary_model, ", DAG: ", dag_model)
  message("[ollama] ", msg)
  list(ok = TRUE, model = summary_model, host = base_url, message = msg)
}


# ---- Chat completion ----

#' Chat completion via Ollama (OpenAI-compatible endpoint)
#'
#' @param system_msg  Character string for the system role.
#' @param user_msg    Character string for the user role.
#' @param model       Model name (default from env or "llama3.2:latest").
#' @param temperature Sampling temperature.
#' @return The assistant's reply as a single character string, or NULL on failure.
openai_chat <- function(system_msg, user_msg,
                        model = .ollama_default_model(),
                        temperature = 0) {
  base_url <- .ollama_base_url()

  if (!.ollama_available(base_url)) {
    message("Ollama is not reachable at ", base_url,
            ". Make sure Ollama is running (ollama serve).")
    return(NULL)
  }

  body <- list(
    model       = model,
    temperature = temperature,
    messages    = list(
      list(role = "system", content = system_msg),
      list(role = "user",   content = user_msg)
    )
  )

  endpoint <- paste0(base_url, "/v1/chat/completions")

  req <- httr2::request(endpoint) |>
    httr2::req_headers(`Content-Type` = "application/json") |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(600)

  resp <- httr2::req_perform(req)
  j    <- httr2::resp_body_json(resp, simplifyVector = FALSE)

  if (!is.null(j$error)) {
    stop("Ollama API error: ", j$error$message)
  }

  content <- j$choices[[1]]$message$content
  if (is.list(content)) {
    content <- paste(vapply(content, function(part) {
      if (!is.null(part$text)) part$text
      else if (!is.null(part$content)) part$content
      else ""
    }, character(1)), collapse = "")
  }
  as.character(content)
}
