# ============================================================
# R/dag_utils.R — DAG construction, conversion, and layout
# ============================================================

#' Convert an adjacency matrix to a dagitty DAG string
#'
#' @param adj Square adjacency matrix with named rows/columns.
#'            adj[i,j] == 1 means i -> j.
#' @return A dagitty object.
adj2dag <- function(adj) {
  nodes <- rownames(adj)
  if (is.null(nodes)) nodes <- colnames(adj)
  if (is.null(nodes)) stop("Adjacency matrix must have dimnames.")

  s <- "dag {"
  for (i in seq_len(nrow(adj))) {
    for (j in seq_len(ncol(adj))) {
      if (adj[i, j] == 1) {
        s <- paste(s, nodes[i], "->", nodes[j], ";")
      }
    }
  }
  dagitty::dagitty(paste0(s, "}"))
}


#' Sanitise column names to valid R identifiers
make_safe_names <- function(x) {
  x2 <- gsub("[^0-9A-Za-z_\\.]", "_", x)
  x2 <- ifelse(grepl("^[A-Za-z]", x2), x2, paste0("V", x2))
  make.unique(x2)
}


#' Convert an adjacency matrix to an edge-list data.frame
#'
#' @param A Adjacency matrix.
#' @return data.frame with columns `from`, `to`.
adjacency_to_edges <- function(A) {
  vars_row <- rownames(A)
  vars_col <- colnames(A)
  if (is.null(vars_row)) vars_row <- vars_col
  if (is.null(vars_col)) vars_col <- vars_row

  idx <- which(A != 0, arr.ind = TRUE)
  if (nrow(idx) == 0) {
    return(data.frame(from = character(0), to = character(0),
                      stringsAsFactors = FALSE))
  }
  data.frame(
    from = vars_row[idx[, "row"]],
    to   = vars_col[idx[, "col"]],
    stringsAsFactors = FALSE
  )
}


#' Create a circular node layout (for visNetwork / base plots)
#'
#' @param vars       Character vector of variable names.
#' @param paper_order Optional preferred ordering around the circle.
#' @param radius     Circle radius in pixel-like units.
#' @return data.frame with columns `id`, `x`, `y`.
make_circle_nodes <- function(vars, paper_order = NULL, radius = 300) {
  if (!is.null(paper_order)) {
    vars_ordered <- c(
      paper_order[paper_order %in% vars],
      vars[!vars %in% paper_order]
    )
  } else {
    vars_ordered <- vars
  }

  n      <- length(vars_ordered)
  angles <- seq(0, 2 * pi - 2 * pi / n, length.out = n)

  coords <- data.frame(
    id = vars_ordered,
    x  = radius * sin(angles),
    y  = radius * cos(angles),
    stringsAsFactors = FALSE
  )

  nodes <- merge(
    data.frame(id = vars, stringsAsFactors = FALSE),
    coords, by = "id", all.x = TRUE, sort = FALSE
  )

  # Fallback for any vars not in paper_order

  na_idx <- is.na(nodes$x)
  if (any(na_idx)) {
    m       <- sum(na_idx)
    angles2 <- seq(0, 2 * pi - 2 * pi / m, length.out = m)
    nodes$x[na_idx] <- (radius * 0.6) * sin(angles2)
    nodes$y[na_idx] <- (radius * 0.6) * cos(angles2)
  }
  nodes
}
