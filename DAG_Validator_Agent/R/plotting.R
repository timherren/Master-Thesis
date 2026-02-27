# ============================================================
# R/plotting.R — DAG visualisation with CI annotations
# ============================================================

#' Plot a DAG with optional annotation overlays
#'
#' @param amat_base     Directed adjacency matrix (the hypothesised DAG).
#' @param amat_testable Optional symmetric matrix: 1 = pair was CI-tested
#'                      (drawn as red dotted lines).
#' @param amat_new      Optional symmetric matrix: 1 = rejected CI pair
#'                      (drawn as red solid edges).
#' @param main          Plot title.
#' @param layout        Two-column matrix (x, y) with rownames = node names.
plot_dag_with_annotations <- function(amat_base,
                                      amat_testable = NULL,
                                      amat_new      = NULL,
                                      main   = "DAG",
                                      layout = NULL) {
  vars <- rownames(amat_base)
  if (is.null(vars)) stop("amat_base must have dimnames (variable names).")

  g <- igraph::graph_from_adjacency_matrix(amat_base, mode = "directed")
  igraph::E(g)$color <- "black"
  igraph::E(g)$lty   <- 1
  igraph::E(g)$lwd   <- 1


  # 1) Add "missing" edges from rejected CIs (solid red)
  if (!is.null(amat_new)) {
    for (i in seq_len(nrow(amat_new))) {
      for (j in seq_len(ncol(amat_new))) {
        if (amat_new[i, j] == 1L &&
            amat_base[i, j] == 0L && amat_base[j, i] == 0L) {
          v1 <- rownames(amat_new)[i]
          v2 <- colnames(amat_new)[j]
          if (v1 %in% vars && v2 %in% vars) {
            g <- igraph::add_edges(g, c(v1, v2),
                                   color = "red", lty = 1, lwd = 3)
          }
        }
      }
    }
  }

  # 2) Layout
  if (!is.null(layout)) {
    lay <- layout[igraph::V(g)$name, , drop = FALSE]
  } else {
    lay <- igraph::layout_in_circle(g)
  }

  xlim <- range(lay[, 1]) * 1.2
  ylim <- range(lay[, 2]) * 1.2

  # Reserve space at the bottom for the legend
  op <- par(mar = c(4, 1, 2, 1))
  on.exit(par(op), add = TRUE)

  plot(
    g,
    main             = main,
    vertex.label.cex = 1.1,
    vertex.size      = 25,
    edge.arrow.size  = 0.4,
    layout           = lay,
    rescale          = FALSE,
    xlim             = xlim,
    ylim             = ylim
  )

  # 3) Overlay CI-tested pairs as red dotted lines (undirected)
  if (!is.null(amat_testable)) {
    test_rows <- rownames(amat_testable)
    test_cols <- colnames(amat_testable)
    # Align tested-CI matrix to the plotted DAG node order to avoid
    # index/order drift between matrices and layout coordinates.
    if (!is.null(test_rows) && !is.null(test_cols) &&
        all(vars %in% test_rows) && all(vars %in% test_cols)) {
      A_test <- amat_testable[vars, vars, drop = FALSE]
    } else {
      A_test <- amat_testable
    }

    lay_names <- rownames(lay)
    if (!is.null(lay_names) && all(vars %in% lay_names)) {
      lay_aligned <- lay[vars, , drop = FALSE]
    } else {
      lay_aligned <- lay
    }

    n_test <- min(nrow(A_test), ncol(A_test), nrow(lay_aligned))
    if (n_test >= 2) {
      for (i in seq_len(n_test)) {
        for (j in seq_len(n_test)) {
          if (i < j && isTRUE(A_test[i, j] == 1L)) {
            segments(
              x0 = lay_aligned[i, 1], y0 = lay_aligned[i, 2],
              x1 = lay_aligned[j, 1], y1 = lay_aligned[j, 2],
              col = "red", lty = 3, lwd = 2
            )
          }
        }
      }
    }
  }

  # 4) Legend — drawn in the bottom margin so it never gets clipped
  leg_labels <- "DAG edges"
  leg_col    <- "black"
  leg_lty    <- 1
  leg_lwd    <- 2

  if (!is.null(amat_testable)) {
    leg_labels <- c(leg_labels, "CI tests (tested)")
    leg_col    <- c(leg_col, "red")
    leg_lty    <- c(leg_lty, 3)
    leg_lwd    <- c(leg_lwd, 2)
  }
  if (!is.null(amat_new)) {
    leg_labels <- c(leg_labels, "Rejected CIs")
    leg_col    <- c(leg_col, "red")
    leg_lty    <- c(leg_lty, 1)
    leg_lwd    <- c(leg_lwd, 3)
  }

  # xpd = TRUE allows drawing in the margin area
  par(xpd = TRUE)
  legend("bottom",
         legend  = leg_labels,
         col     = leg_col,
         lty     = leg_lty,
         lwd     = leg_lwd,
         bg      = "white",
         box.col = "grey80",
         cex     = 0.95,
         horiz   = TRUE,
         inset   = c(0, -0.15))
}
