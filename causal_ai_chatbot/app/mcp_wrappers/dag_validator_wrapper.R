#!/usr/bin/env Rscript

required_pkgs <- c("jsonlite", "dplyr", "tibble", "dagitty", "comets", "httr2", "igraph")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  message("Installing missing R packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tibble)
  library(dagitty)
  library(comets)
  library(httr2)
  library(igraph)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

parse_args <- function(args) {
  out <- list()
  for (i in seq_along(args)) {
    if (args[i] == "--input" && i < length(args)) out$input <- args[i + 1]
    if (args[i] == "--output" && i < length(args)) out$output <- args[i + 1]
  }
  out
}

safe_out <- function(path, payload) {
  writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", pretty = TRUE), path)
}

adj2dag <- function(adj) {
  nodes <- rownames(adj)
  if (is.null(nodes)) nodes <- colnames(adj)
  if (is.null(nodes)) stop("Adjacency matrix must have dimnames.")
  s <- "dag {"
  for (i in seq_len(nrow(adj))) {
    for (j in seq_len(ncol(adj))) {
      if (adj[i, j] == 1) s <- paste(s, nodes[i], "->", nodes[j], ";")
    }
  }
  dagitty::dagitty(paste0(s, "}"))
}

cis_with_Z <- function(amat) {
  g <- adj2dag(amat)
  cis <- dagitty::impliedConditionalIndependencies(g)
  cis[vapply(cis, function(x) length(x$Z) > 0, logical(1))]
}

run_ci_tests <- function(amat, dat, tests = c("gcm", "pcm"), alpha = 0.05) {
  tcis <- cis_with_Z(amat)
  if (length(tcis) == 0) {
    return(tibble::tibble(
      test = character(0), CI = character(0), p.value = numeric(0),
      adj.p.value = numeric(0), rejected = logical(0)
    ))
  }

  lapply(tests, function(tst) {
    pv <- vapply(seq_along(tcis), function(k) {
      ci <- tcis[[k]]
      fm <- reformulate(
        paste0(paste0(ci$Y, collapse = "+"), "|", paste0(ci$Z, collapse = "+")),
        response = ci$X
      )
      comets::comets(fm, dat, test = tst, coin = TRUE)$p.value
    }, numeric(1))

    ci_labels <- vapply(tcis, function(ci) {
      x <- if (length(ci$X) > 0) paste(ci$X, collapse = "+") else ""
      y <- if (length(ci$Y) > 0) paste(ci$Y, collapse = "+") else ""
      z <- if (length(ci$Z) > 0) paste(ci$Z, collapse = ", ") else "∅"
      paste(x, "_||_", y, "|", z)
    }, character(1))

    tibble::tibble(
      test = tst,
      CI = ci_labels,
      p.value = pv,
      adj.p.value = stats::p.adjust(pv, "holm"),
      rejected = stats::p.adjust(pv, "holm") < alpha
    )
  }) |> dplyr::bind_rows()
}

parse_ci_pair <- function(ci_str) {
  if (is.null(ci_str) || is.na(ci_str) || !nzchar(ci_str)) return(c(NA_character_, NA_character_))
  main <- trimws(ci_str)
  main <- gsub("_\\|\\|_", "⟂", main)
  main <- gsub("\\|\\|", "⟂", main)
  main <- gsub("⫫", "⟂", main)
  main <- gsub("⟂⟂", "⟂", main)
  main <- sub("\\s*\\|.*$", "", main)
  parts <- trimws(strsplit(main, "⟂", fixed = TRUE)[[1]])
  if (length(parts) < 2) return(c(NA_character_, NA_character_))
  c(parts[1], parts[2])
}

build_testable_matrix <- function(ci_results, vars) {
  A <- matrix(0L, nrow = length(vars), ncol = length(vars), dimnames = list(vars, vars))
  if (is.null(ci_results) || !nrow(ci_results)) return(A)
  for (ci in ci_results$CI) {
    pair <- parse_ci_pair(ci)
    X <- pair[1]; Y <- pair[2]
    if (!is.na(X) && !is.na(Y) && X %in% vars && Y %in% vars) {
      A[X, Y] <- 1L; A[Y, X] <- 1L
    }
  }
  A
}

infer_missing_edges_from_ci <- function(ci_results, vars) {
  A <- matrix(0L, nrow = length(vars), ncol = length(vars), dimnames = list(vars, vars))
  if (is.null(ci_results) || !nrow(ci_results)) return(A)
  rejs <- ci_results[ci_results$rejected, , drop = FALSE]
  if (!nrow(rejs)) return(A)
  for (ci in rejs$CI) {
    pair <- parse_ci_pair(ci)
    X <- pair[1]; Y <- pair[2]
    if (!is.na(X) && !is.na(Y) && X %in% vars && Y %in% vars) {
      A[X, Y] <- 1L; A[Y, X] <- 1L
    }
  }
  A
}

plot_dag_with_annotations <- function(amat_base, amat_testable = NULL, amat_new = NULL, main = "DAG") {
  g <- igraph::graph_from_adjacency_matrix(amat_base, mode = "directed")
  igraph::E(g)$color <- "black"
  igraph::E(g)$lwd <- 1
  vars <- rownames(amat_base)

  if (!is.null(amat_new)) {
    for (i in seq_len(nrow(amat_new))) {
      for (j in seq_len(ncol(amat_new))) {
        if (amat_new[i, j] == 1L && amat_base[i, j] == 0L && amat_base[j, i] == 0L) {
          v1 <- rownames(amat_new)[i]; v2 <- colnames(amat_new)[j]
          g <- igraph::add_edges(g, c(v1, v2), color = "red", lwd = 4)
        }
      }
    }
  }

  lay <- igraph::layout_in_circle(g)
  rownames(lay) <- igraph::V(g)$name
  xlim <- range(lay[, 1]) * 1.2
  ylim <- range(lay[, 2]) * 1.2
  plot(
    g,
    main = main,
    vertex.label.cex = 1.1,
    vertex.size = 25,
    edge.arrow.size = 0.4,
    layout = lay,
    rescale = FALSE,
    xlim = xlim,
    ylim = ylim
  )

  if (!is.null(amat_testable)) {
    test_rows <- rownames(amat_testable)
    test_cols <- colnames(amat_testable)
    if (!is.null(test_rows) && !is.null(test_cols) &&
        all(vars %in% test_rows) && all(vars %in% test_cols)) {
      A_test <- amat_testable[vars, vars, drop = FALSE]
    } else {
      A_test <- amat_testable
    }

    for (i in seq_len(nrow(A_test))) {
      for (j in seq_len(ncol(A_test))) {
        if (i < j && isTRUE(A_test[i, j] == 1L)) {
          v1 <- rownames(A_test)[i]
          v2 <- colnames(A_test)[j]
          idx1 <- match(v1, rownames(lay))
          idx2 <- match(v2, rownames(lay))
          if (is.na(idx1) || is.na(idx2)) next
          segments(
            lay[idx1, 1], lay[idx1, 2],
            lay[idx2, 1], lay[idx2, 2],
            col = "red", lty = 3, lwd = 2
          )
        }
      }
    }
  }
}

adjacency_to_edges <- function(A) {
  idx <- which(A != 0, arr.ind = TRUE)
  if (nrow(idx) == 0) return(data.frame(from = character(0), to = character(0), stringsAsFactors = FALSE))
  data.frame(from = rownames(A)[idx[, 1]], to = colnames(A)[idx[, 2]], stringsAsFactors = FALSE)
}

ollama_available <- function() {
  host <- Sys.getenv("OLLAMA_HOST", "http://localhost:11434")
  ok <- tryCatch({
    httr2::request(host) |> httr2::req_timeout(2) |> httr2::req_perform()
    TRUE
  }, error = function(e) FALSE)
  ok
}

ollama_chat <- function(system_msg, user_msg, model = Sys.getenv("OLLAMA_DAG_MODEL", "llama3.2:latest"), temperature = 0.2) {
  host <- Sys.getenv("OLLAMA_HOST", "http://localhost:11434")
  body <- list(
    model = model,
    messages = list(
      list(role = "system", content = system_msg),
      list(role = "user", content = user_msg)
    ),
    stream = FALSE,
    options = list(temperature = temperature)
  )
  resp <- httr2::request(paste0(host, "/api/chat")) |>
    httr2::req_method("POST") |>
    httr2::req_body_json(body, auto_unbox = TRUE) |>
    httr2::req_timeout(120) |>
    httr2::req_perform()
  j <- httr2::resp_body_json(resp, simplifyVector = TRUE)
  j$message$content
}

propose_dag_from_llm <- function(vars, expert_text = NULL) {
  empty_A <- matrix(0L, nrow = length(vars), ncol = length(vars), dimnames = list(vars, vars))
  if (!ollama_available()) return(list(A = empty_A, error = "Ollama is not running. Start it with: ollama serve"))

  sys_msg <- paste(
    "You are an expert in causal discovery and cell signaling.",
    "Output ONLY directed edges in format Parent,Child using provided variable names."
  )
  expert_block <- if (!is.null(expert_text) && nzchar(expert_text)) paste0("\n\nExpert background knowledge:\n", expert_text) else ""
  usr_msg <- paste0("Variable names:\n", paste(vars, collapse = ", "), expert_block)
  txt <- tryCatch(ollama_chat(sys_msg, usr_msg), error = function(e) NULL)
  if (is.null(txt) || !nzchar(txt)) return(list(A = empty_A, error = "LLM did not respond."))

  lines <- strsplit(txt, "\n", fixed = TRUE)[[1]]
  lines <- trimws(lines); lines <- lines[nzchar(lines)]
  A <- empty_A
  for (ln in lines) {
    parts <- strsplit(ln, ",", fixed = TRUE)[[1]]
    if (length(parts) != 2) next
    parent <- trimws(parts[1]); child <- trimws(parts[2])
    if (parent %in% vars && child %in% vars && parent != child) A[parent, child] <- 1L
  }
  list(A = A, error = NULL)
}

to_matrix <- function(x, vars) {
  mat <- matrix(unlist(x), nrow = length(vars), byrow = TRUE)
  dimnames(mat) <- list(vars, vars)
  mat
}

argv <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(argv$input) || is.null(argv$output)) stop("Usage: --input <input.json> --output <output.json>")
input <- fromJSON(argv$input, simplifyVector = TRUE)
tool <- input$tool
payload <- input$payload
artifact_dir <- input$artifact_dir %||% tempdir()
dir.create(artifact_dir, recursive = TRUE, showWarnings = FALSE)

result <- list(success = FALSE, data = list(), artifacts = list(), tables = list(), messages = list(), error = NULL)

tryCatch({
  if (tool == "propose_dag") {
    vars <- unlist(payload$vars %||% character(0))
    proposal <- propose_dag_from_llm(vars = vars, expert_text = payload$expert_text %||% NULL)
    result$success <- TRUE
    result$data <- list(
      adjacency_matrix = unname(proposal$A),
      variables = vars,
      edges = adjacency_to_edges(proposal$A),
      llm_explanation = proposal$error %||% ""
    )
    result$messages <- c("DAG proposal completed.")
  } else if (tool == "test_dag") {
    dag <- payload$dag
    vars <- unlist(dag$variables)
    amat <- to_matrix(dag$adjacency_matrix, vars)
    alpha <- payload$alpha %||% 0.05
    tests <- unlist(payload$tests %||% c("gcm", "pcm"))
    data_path <- payload$data_path
    if (is.null(data_path) || !file.exists(data_path)) stop("payload$data_path missing or not found.")

    dat <- if (grepl("\\.xlsx?$", data_path, ignore.case = TRUE)) {
      suppressPackageStartupMessages(library(readxl))
      as.data.frame(read_excel(data_path))
    } else read.csv(data_path, check.names = FALSE)

    res <- run_ci_tests(amat, dat, tests = tests, alpha = alpha)
    rej_count <- sum(res$rejected)
    tested_mat <- build_testable_matrix(res, vars)
    missing_mat <- infer_missing_edges_from_ci(res, vars)

    tested_plot <- file.path(artifact_dir, "ci_tested_plot.png")
    rejected_plot <- file.path(artifact_dir, "ci_rejected_plot.png")
    png(filename = tested_plot, width = 1000, height = 800)
    plot_dag_with_annotations(amat_base = amat, amat_testable = tested_mat, main = "CI Tests Performed")
    dev.off()
    png(filename = rejected_plot, width = 1000, height = 800)
    plot_dag_with_annotations(amat_base = amat, amat_new = missing_mat, main = "DAG with Rejected CIs")
    dev.off()

    tests_out <- lapply(seq_len(nrow(res)), function(i) {
      list(
        ci = as.character(res$CI[i]),
        p_value = as.numeric(res$p.value[i]),
        adj_p_value = as.numeric(res$adj.p.value[i]),
        rejected = as.logical(res$rejected[i]),
        test = as.character(res$test[i])
      )
    })

    result$success <- TRUE
    result$data <- list(
      consistent = (rej_count == 0),
      rejected_count = as.integer(rej_count),
      tests = tests_out,
      tested_matrix = unname(tested_mat),
      rejected_matrix = unname(missing_mat)
    )
    result$tables <- list(
      summary = as.data.frame(res %>% group_by(test) %>% summarise(
        tests = n(), rejected = sum(rejected), min_adj_p = min(adj.p.value, na.rm = TRUE), .groups = "drop"
      )),
      rejected = as.data.frame(res %>% filter(rejected))
    )
    result$artifacts <- list(tested_plot, rejected_plot)
    result$messages <- c("DAG CI testing completed.")
  } else {
    stop(sprintf("Unsupported DAG tool: %s", tool))
  }
}, error = function(e) {
  result$success <<- FALSE
  result$error <<- as.character(e$message)
})

safe_out(argv$output, result)
