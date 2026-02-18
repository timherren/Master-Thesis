# ============================================================
# R/ci_testing.R — Conditional independence testing via COMETs
# ============================================================

#' Extract implied CIs that have a non-empty conditioning set Z
#'
#' @param amat Adjacency matrix (with dimnames).
#' @return A list of dagitty CI objects where length(Z) > 0.
cis_with_Z <- function(amat) {
  g   <- adj2dag(amat)
  cis <- dagitty::impliedConditionalIndependencies(g)
  cis[vapply(cis, function(x) length(x$Z) > 0, logical(1))]
}


#' Run COMETs CI tests for every implied CI with non-empty Z
#'
#' @param amat  Adjacency matrix (named).
#' @param dat   Data frame with columns matching the DAG node names.
#' @param tests Character vector of test types, e.g. c("gcm","pcm").
#' @param alpha Significance level for Holm-adjusted p-values.
#' @return A tibble with columns: test, CI, p.value, adj.p.value, rejected.
run_ci_tests <- function(amat, dat, tests = c("gcm", "pcm"), alpha = 0.05) {
  tcis <- cis_with_Z(amat)

  if (length(tcis) == 0) {
    return(tibble::tibble(
      test        = character(0),
      CI          = character(0),
      p.value     = numeric(0),
      adj.p.value = numeric(0),
      rejected    = logical(0)
    ))
  }

  res <- lapply(tests, function(tst) {
    pv <- vapply(seq_along(tcis), function(k) {
      ci <- tcis[[k]]
      fm <- reformulate(
        paste0(paste0(ci$Y, collapse = "+"), "|", paste0(ci$Z, collapse = "+")),
        response = ci$X
      )
      comets::comets(fm, dat, test = tst, coin = TRUE)$p.value
    }, numeric(1))

    tibble::tibble(
      test        = tst,
      CI          = vapply(tcis, paste, character(1)),
      p.value     = pv,
      adj.p.value = stats::p.adjust(pv, "holm"),
      rejected    = stats::p.adjust(pv, "holm") < alpha
    )
  }) |> dplyr::bind_rows()

  res
}


#' List all implied CIs (with Z) as a tidy data frame
#'
#' @param amat Adjacency matrix (named).
#' @return data.frame with columns X, Y, Z, CI_Statement.
get_all_ci_tests <- function(amat) {
  if (is.null(amat)) {
    return(data.frame(
      X = character(0), Y = character(0),
      Z = character(0), CI_Statement = character(0),
      stringsAsFactors = FALSE
    ))
  }

  g       <- adj2dag(amat)
  cis_all <- dagitty::impliedConditionalIndependencies(g)
  cis     <- cis_all[vapply(cis_all, function(ci) length(ci$Z) > 0, logical(1))]

  if (length(cis) == 0) {
    return(data.frame(
      X = character(0), Y = character(0),
      Z = character(0), CI_Statement = character(0),
      stringsAsFactors = FALSE
    ))
  }

  ci_df <- lapply(cis, function(ci) {
    x_vars <- if (length(ci$X) > 0) ci$X else character(0)
    y_vars <- if (length(ci$Y) > 0) ci$Y else character(0)
    z_vars <- if (length(ci$Z) > 0) ci$Z else character(0)

    data.frame(
      X = paste(x_vars, collapse = ", "),
      Y = paste(y_vars, collapse = ", "),
      Z = if (length(z_vars) > 0) paste(z_vars, collapse = ", ") else "\u2205",
      CI_Statement = paste(
        paste(x_vars, collapse = "+"),
        "_||_",
        paste(y_vars, collapse = "+"),
        "|",
        if (length(z_vars) > 0) paste(z_vars, collapse = ", ") else "\u2205"
      ),
      stringsAsFactors = FALSE
    )
  }) |> dplyr::bind_rows()

  ci_df
}


#' Parse a CI string like "X _||_ Y | Z1, Z2" into c(X, Y)
#'
#' @param ci_str A single CI statement string.
#' @return Character vector of length 2: c(X, Y), or c(NA, NA).
parse_ci_pair <- function(ci_str) {
  if (is.null(ci_str) || is.na(ci_str) || !nzchar(ci_str)) {
    return(c(NA_character_, NA_character_))
  }

  main <- trimws(ci_str)
  # Normalise independence symbols
  main <- gsub("_\\|\\|_", "\u27C2", main)
  main <- gsub("\\|\\|",   "\u27C2", main)
  main <- gsub("\u2AEB",   "\u27C2", main)
  main <- gsub("\u27C2\u27C2", "\u27C2", main)

  # Drop conditioning set (everything after " |")
  main <- sub("\\s*\\|.*$", "", main)
  main <- trimws(main)

  parts <- strsplit(main, "\u27C2", fixed = TRUE)[[1]]
  parts <- trimws(parts)

  if (length(parts) < 2) return(c(NA_character_, NA_character_))
  c(parts[1], parts[2])
}


#' Build a symmetric (undirected) matrix marking which pairs were tested
build_testable_matrix <- function(ci_results, vars) {
  A <- matrix(0L, nrow = length(vars), ncol = length(vars),
              dimnames = list(vars, vars))
  if (is.null(ci_results) || !nrow(ci_results)) return(A)

  for (ci in ci_results$CI) {
    pair <- parse_ci_pair(ci)
    X <- pair[1]; Y <- pair[2]
    if (!is.na(X) && !is.na(Y) && X %in% vars && Y %in% vars) {
      A[X, Y] <- 1L
      A[Y, X] <- 1L
    }
  }
  A
}


#' Build a symmetric matrix marking pairs whose CI was rejected
infer_missing_edges_from_ci <- function(ci_results, vars) {
  A <- matrix(0L, nrow = length(vars), ncol = length(vars),
              dimnames = list(vars, vars))
  if (is.null(ci_results) || !nrow(ci_results)) return(A)

  rejs <- ci_results[ci_results$rejected, , drop = FALSE]
  if (!nrow(rejs)) return(A)

  for (ci in rejs$CI) {
    pair <- parse_ci_pair(ci)
    X <- pair[1]; Y <- pair[2]
    if (!is.na(X) && !is.na(Y) && X %in% vars && Y %in% vars) {
      A[X, Y] <- 1L
      A[Y, X] <- 1L
    }
  }
  A
}
