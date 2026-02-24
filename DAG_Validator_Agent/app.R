# ============================================================
# app.R — DAG Validator Agent Shiny application
#
# All helper functions live in R/ and are auto-sourced by Shiny.
# This file only contains: library loading, .env setup, UI, server.
# ============================================================

## ---- Libraries ----
library(shiny)
library(dplyr)
library(tibble)
library(igraph)
library(httr2)
library(visNetwork)
library(comets)
library(dagitty)
library(shinybusy)
library(DT)
library(dotenv)
library(readxl)
library(rmarkdown)
library(zip)

## ---- .env ----
if (file.exists(".env")) dotenv::load_dot_env(".env")

## ---- Ollama auto-connect (runs once at startup) ----
.ollama_status <- ollama_ensure_ready()


## ======================================================
## UI
## ======================================================

ui <- fluidPage(
  titlePanel("DAG Validator Agent"),

  tags$head(
    tags$meta(`http-equiv` = "Cache-Control", content = "no-cache, no-store, must-revalidate"),
    tags$meta(`http-equiv` = "Pragma", content = "no-cache"),
    tags$meta(`http-equiv` = "Expires", content = "0"),
    tags$style(HTML("
      /* ===== Global & Typography ===== */
      body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        background-color: #f0f4f8;
        color: #2c3e50;
      }

      /* ===== Title ===== */
      h2 { color: #1a5276; }

      /* ===== Sidebar Panel — Blue Medical Theme ===== */
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

      /* ===== Sidebar Buttons ===== */
      .well .btn-default,
      .well .btn-primary {
        background: linear-gradient(135deg, #2980b9 0%, #1a6da0 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(26, 82, 118, 0.15);
        width: 100%;
        padding: 8px 16px;
      }
      .well .btn-default:hover, .well .btn-default:focus,
      .well .btn-primary:hover, .well .btn-primary:focus {
        background: linear-gradient(135deg, #1a6da0 0%, #15577f 100%);
        box-shadow: 0 4px 12px rgba(26, 109, 160, 0.35);
        color: #fff;
      }

      /* Download buttons */
      .btn-default.shiny-download-link {
        background: linear-gradient(135deg, #1a5276 0%, #154360 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 8px 16px;
        box-shadow: 0 2px 4px rgba(26, 82, 118, 0.15);
        transition: all 0.2s ease;
      }
      .btn-default.shiny-download-link:hover,
      .btn-default.shiny-download-link:focus {
        background: linear-gradient(135deg, #154360 0%, #0e2f44 100%);
        box-shadow: 0 4px 12px rgba(26, 82, 118, 0.35);
        color: #fff;
      }

      /* ===== Ollama Status Badge ===== */
      .ollama-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 4px;
      }
      .ollama-ok {
        background-color: #d6eaf8;
        color: #1a5276;
        border: 1px solid #85c1e9;
      }
      .ollama-fail {
        background-color: #fadbd8;
        color: #922b21;
        border: 1px solid #f1948a;
      }

      /* ===== LLM Interpretation Box ===== */
      #llm_interpretation {
        white-space: pre-wrap;
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
        font-size: 15px;
        margin-top: 12px;
        margin-bottom: 40px;
        line-height: 1.5;
        border: none;
        border-left: 4px solid #5dade2;
        padding: 14px 16px;
        background-color: #eaf2fb;
        border-radius: 4px;
        color: #2c3e50;
      }

      /* ===== Main Panel Section Headings ===== */
      .main-panel h4 {
        color: #1a5276;
        font-weight: 600;
      }

      /* ===== Plot Containers ===== */
      .plot-container img {
        cursor: zoom-in;
        transition: opacity 0.2s, box-shadow 0.2s;
        border: 1px solid #c8ddf0;
        border-radius: 6px;
      }
      .plot-container img:hover {
        opacity: 0.88;
        box-shadow: 0 3px 12px rgba(41, 128, 185, 0.2);
      }

      /* ===== Full-size Modal Overlay ===== */
      #plot-modal-overlay {
        display: none;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(10, 30, 50, 0.85);
        z-index: 10000;
        cursor: zoom-out;
        overflow: auto;
        text-align: center;
        padding: 20px;
      }
      #plot-modal-overlay.active {
        display: flex;
        justify-content: center;
        align-items: center;
      }
      #plot-modal-overlay img {
        max-width: 95%;
        max-height: 95vh;
        border-radius: 6px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
      }
      #plot-modal-close {
        position: fixed;
        top: 15px; right: 25px;
        color: white;
        font-size: 32px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10001;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
      }
      #plot-modal-close:hover { color: #aed6f1; }

      /* ===== Tables ===== */
      .table { border-collapse: collapse; }
      .table th {
        background-color: #eaf2fb;
        color: #1a5276;
        border-bottom: 2px solid #aed6f1;
      }
      .table td {
        border-bottom: 1px solid #e0ecf5;
      }

      /* ===== Notifications ===== */
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

      /* ===== Busy Spinner ===== */
      .shiny-busy .shinybusy {
        border-top-color: #2980b9 !important;
      }
    ")),
    tags$script(HTML("
      $(document).on('click', '.plot-container img', function() {
        var src = $(this).attr('src');
        $('#plot-modal-img').attr('src', src);
        $('#plot-modal-overlay').addClass('active');
      });
      $(document).on('click', '#plot-modal-overlay, #plot-modal-close', function() {
        $('#plot-modal-overlay').removeClass('active');
      });
      $(document).on('keydown', function(e) {
        if (e.key === 'Escape') $('#plot-modal-overlay').removeClass('active');
      });
    "))
  ),

  # Fullscreen modal (hidden by default)
  tags$div(id = "plot-modal-overlay",
    tags$span(id = "plot-modal-close", HTML("&times;")),
    tags$img(id = "plot-modal-img", src = "")
  ),

  add_busy_spinner(
    spin     = "fading-circle",
    position = "top-right",
    margins  = c(10, 10),
    timeout  = 100
  ),

  sidebarLayout(
    # ---- Sidebar ----
    sidebarPanel(
      uiOutput("ollama_status_badge"),
      hr(),
      h4("1. Upload data"),
      fileInput("datafile", "Upload dataset",
                accept = c(".csv", ".tsv", ".txt", ".xlsx", ".xls",
                           "text/csv",
                           "application/vnd.ms-excel",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
      hr(),
      uiOutput("var_info"),

      hr(),
      h4("1b. Preprocessing"),
      radioButtons("preprocess", "Transform numeric columns",
                   choices = c("None (raw data)"       = "none",
                               "Log transform (ln)"    = "log",
                               "Standardize (z-score)" = "zscore"),
                   selected = "none"),
      uiOutput("preprocess_warning"),

      hr(),
      h4("2. Choose DAG hypothesis"),
      radioButtons("dag_source", "DAG source",
                   choices = c("LLM proposal"           = "llm",
                               "Upload adjacency matrix" = "upload",
                               "Manual editor"           = "manual"),
                   inline = FALSE),

      conditionalPanel(
        condition = "input.dag_source == 'llm'",
        textAreaInput("expert_text", "Expert knowledge (optional)",
                      placeholder = "e.g. 'Akt influences Erk and PIP3 acts on Akt'",
                      rows = 4),
        actionButton("btn_llm_dag", "Propose LLM DAG")
      ),

      conditionalPanel(
        condition = "input.dag_source == 'upload'",
        fileInput("amat_file", "Adjacency-matrix CSV (rows/cols = variables)"),
        actionButton("btn_upload_dag", "Load DAG")
      ),

      hr(),
      h4("3. CI tests"),
      numericInput("alpha", "Significance level \u03B1",
                   value = 0.05, min = 0, max = 0.5, step = 0.01),
      checkboxGroupInput("tests", "CI test types",
                         choices  = c("GCM" = "gcm", "PCM" = "pcm"),
                         selected = c("gcm", "pcm")),
      actionButton("btn_run_ci", "Run CI tests"),

      hr(),
      div(style = "text-align: center; color: #85a5c2; font-size: 0.82em; padding: 8px 0 4px 0;",
          "DAG Validator Agent")
    ),

    # ---- Main panel ----
    mainPanel(
      conditionalPanel(
        condition = "input.dag_source == 'manual'",
        h4("Manual DAG editor"),
        p("Drag nodes, add edges via the '+' icon, delete via the bin icon."),
        visNetworkOutput("dag_editor", height = "400px"),
        br(),
        actionButton("btn_manual_apply", "Apply edges from editor as DAG")
      ),
      hr(),

      h4("Hypothesised DAG"),
      fluidRow(
        column(6, div(class = "plot-container", plotOutput("dag_plot",   height = "500px"))),
        column(6, div(class = "plot-container", plotOutput("dag_plot_2", height = "500px")))
      ),

      hr(),
      h4("Conditional Independence Tests (COMETs)"),
      p("CI tests implied by the DAG (only those with a non-empty conditioning set):"),
      DT::dataTableOutput("ci_tests_table"),

      hr(),
      h4("Consistency check"),
      verbatimTextOutput("consistency_msg"),

      hr(),
      h4("CI test summary"),
      tableOutput("tbl_summary"),

      hr(),
      h4("Rejected CIs"),
      tableOutput("tbl_rejected"),

      hr(),
      h4("CI Test Results"),
      fluidRow(
        column(6, div(class = "plot-container", plotOutput("dag_ci_tested_plot", height = "500px"))),
        column(6, div(class = "plot-container", plotOutput("dag_ci_plot", height = "500px")))
      ),

      hr(),
      div(style = "margin-top: 20px; margin-bottom: 40px;",
        h4("LLM interpretation"),
        verbatimTextOutput("llm_interpretation")
      ),

      hr(),
      div(style = "margin-top: 20px; margin-bottom: 60px;",
        h4("Export"),
        p("Download the DAG adjacency matrix, or generate a full report ",
          "with all results, plots, and CSV files bundled into a ZIP folder."),
        fluidRow(
          column(4, downloadButton("btn_export_dag", "Download DAG as CSV")),
          column(4, downloadButton("btn_export_report", "Download Full Report (ZIP)"))
        )
      )
    )
  )
)


## ======================================================
## Server
## ======================================================

server <- function(input, output, session) {

  # -- Ollama status badge --
  output$ollama_status_badge <- renderUI({
    if (.ollama_status$ok) {
      div(class = "ollama-badge ollama-ok",
          paste0("Ollama: ", .ollama_status$model))
    } else {
      div(class = "ollama-badge ollama-fail",
          .ollama_status$message)
    }
  })

  # -- Reactive values --
  rv_manual_edges        <- reactiveVal(NULL)
  rv_layout              <- reactiveVal(NULL)
  rv_amat                <- reactiveVal(NULL)
  rv_ci                  <- reactiveVal(NULL)
  rv_testable            <- reactiveVal(NULL)
  rv_missing             <- reactiveVal(NULL)
  rv_ci_tests_to_perform <- reactiveVal(NULL)

  # Clear stale CI results whenever the DAG changes so the user
  # sees that re-running tests is needed after editing the DAG.
  observeEvent(rv_amat(), {
    rv_ci(NULL)
    rv_testable(NULL)
    rv_missing(NULL)
  }, ignoreInit = TRUE)


  ## ---- DATA (auto header/sep detection) ----
  dat <- reactive({
    req(input$datafile)
    ext  <- tools::file_ext(input$datafile$name)
    path <- input$datafile$datapath

    if (ext %in% c("xlsx", "xls")) {
      df <- as.data.frame(readxl::read_excel(path))
      colnames(df) <- make.names(colnames(df), unique = TRUE)
    } else {
      # Auto-detect separator
      line1 <- readLines(path, n = 1)
      sep_candidates <- c(",", ";", "\t")
      counts <- vapply(sep_candidates, function(s) {
        m <- gregexpr(s, line1, fixed = TRUE)[[1]]
        if (identical(m, -1L)) 0L else length(m)
      }, integer(1))
      sep <- if (all(counts == 0)) "," else sep_candidates[which.max(counts)]

      # Guess header
      peek <- read.table(path, sep = sep, nrows = 5, header = FALSE,
                         stringsAsFactors = FALSE, check.names = FALSE)
      row1 <- peek[1, , drop = TRUE]
      row2 <- if (nrow(peek) >= 2) peek[2, , drop = TRUE] else NULL
      has_letters_row1 <- any(sapply(row1, function(x) grepl("[A-Za-z]", x)))
      all_unique_row1  <- length(unique(as.character(row1))) == ncol(peek)
      has_numbers_row2 <- !is.null(row2) &&
        any(sapply(row2, function(x) suppressWarnings(!is.na(as.numeric(x)))))
      header_guess <- has_letters_row1 && all_unique_row1 && has_numbers_row2

      df <- read.csv(path, header = header_guess, sep = sep, check.names = TRUE)
    }

    # Compute shared circular layout
    vars   <- colnames(df)
    n      <- length(vars)
    angles <- seq(from = pi / 2, by = -2 * pi / n, length.out = n)
    coords <- cbind(x = cos(angles), y = sin(angles))
    rownames(coords) <- vars
    rv_layout(coords)

    df
  })


  ## ---- Preprocessed data ----
  dat_processed <- reactive({
    req(dat())
    df     <- dat()
    method <- input$preprocess
    if (is.null(method) || method == "none") return(df)

    num_cols <- vapply(df, is.numeric, logical(1))
    if (method == "log") {
      for (j in which(num_cols)) df[[j]] <- log(df[[j]])
    } else if (method == "zscore") {
      for (j in which(num_cols)) {
        s <- sd(df[[j]], na.rm = TRUE)
        df[[j]] <- if (s > 0) scale(df[[j]])[, 1] else df[[j]]
      }
    }
    df
  })


  ## ---- Variable info ----
  output$var_info <- renderUI({
    req(dat())
    tagList(
      h5("Variables in dataset:"),
      verbatimTextOutput("var_names")
    )
  })
  output$var_names <- renderText({
    req(dat())
    paste(colnames(dat()), collapse = ", ")
  })


  ## ---- Preprocessing warning ----
  output$preprocess_warning <- renderUI({
    req(dat())
    if (is.null(input$preprocess) || input$preprocess != "log") return(NULL)
    nums <- dat()[vapply(dat(), is.numeric, logical(1))]
    has_nonpos <- any(nums <= 0, na.rm = TRUE)
    if (has_nonpos) {
      tags$div(style = "color: #c0392b; font-size: 0.88em; margin-top: 4px;",
        "Warning: Data contains zeros or negative values.",
        "Log transform will produce -Inf/NaN for those cells.")
    }
  })


  ## ---- Manual DAG editor (visNetwork) ----
  output$dag_editor <- renderVisNetwork({
    req(dat(), rv_layout())
    vars   <- colnames(dat())
    coords <- rv_layout()
    coords_scaled <- coords * 300

    nodes <- data.frame(
      id = vars, label = vars,
      x  = coords_scaled[vars, "x"],
      y  = coords_scaled[vars, "y"],
      fixed = TRUE, physics = FALSE,
      stringsAsFactors = FALSE
    )

    edges <- rv_manual_edges()
    if (is.null(edges) || !is.data.frame(edges) || nrow(edges) == 0 ||
        !all(c("from", "to") %in% colnames(edges))) {
      edges <- data.frame(from = character(0), to = character(0),
                          stringsAsFactors = FALSE)
    } else {
      edges <- edges[, c("from", "to"), drop = FALSE]
    }

    visNetwork(nodes, edges, height = "500px") |>
      visEdges(arrows = "to") |>
      visOptions(manipulation = TRUE, nodesIdSelection = TRUE)
  })

  # Flag: TRUE while waiting for visGetEdges() to return after Apply click
  rv_apply_pending <- reactiveVal(FALSE)

  # When Apply is clicked, request the current edges from the browser widget
  observeEvent(input$btn_manual_apply, {
    req(dat())
    rv_apply_pending(TRUE)
    visNetworkProxy("dag_editor") |> visGetEdges()
  })

  # Fires when visNetwork delivers its edge list (after visGetEdges call).
  # IMPORTANT: do NOT update rv_manual_edges() here — that would trigger
  # renderVisNetwork and wipe the editor. Only update rv_amat().
  observeEvent(input$dag_editor_edges, {
    if (!rv_apply_pending()) return()
    rv_apply_pending(FALSE)

    req(dat())
    vars      <- colnames(dat())
    edges_raw <- input$dag_editor_edges

    # visGetEdges() returns a list-of-lists (one list per edge).
    # Convert robustly to a data.frame with "from" and "to" columns.
    edge_df <- tryCatch({
      if (is.null(edges_raw) || length(edges_raw) == 0) {
        data.frame(from = character(0), to = character(0),
                   stringsAsFactors = FALSE)
      } else if (is.data.frame(edges_raw)) {
        edges_raw
      } else {
        do.call(rbind, lapply(edges_raw, function(e) {
          data.frame(from = as.character(e$from),
                     to   = as.character(e$to),
                     stringsAsFactors = FALSE)
        }))
      }
    }, error = function(e) {
      data.frame(from = character(0), to = character(0),
                 stringsAsFactors = FALSE)
    })

    A <- matrix(0L, nrow = length(vars), ncol = length(vars),
                dimnames = list(vars, vars))
    if (nrow(edge_df) > 0 &&
        all(c("from", "to") %in% colnames(edge_df))) {
      for (i in seq_len(nrow(edge_df))) {
        from <- as.character(edge_df$from[i])
        to   <- as.character(edge_df$to[i])
        if (from %in% vars && to %in% vars && from != to) A[from, to] <- 1L
      }
    }
    rv_amat(A)
    showNotification("Manual DAG edges applied.", type = "message")
  }, ignoreInit = TRUE)


  ## ---- LLM DAG proposal ----
  observeEvent(input$btn_llm_dag, {
    req(dat())
    vars   <- colnames(dat())
    result <- propose_dag_from_llm(vars, expert_text = input$expert_text)

    if (!is.null(result$error)) {
      showNotification(result$error, type = "error", duration = 12)
      return()
    }

    A <- result$A
    rv_amat(A)
    rv_manual_edges(adjacency_to_edges(A))
    updateRadioButtons(session, "dag_source", selected = "manual")
    showNotification("LLM DAG created and loaded into editor.", type = "message")
  })


  ## ---- Upload adjacency matrix ----
  observeEvent(input$btn_upload_dag, {
    req(input$amat_file, dat())
    amat_raw <- as.matrix(read.csv(input$amat_file$datapath,
                                   row.names = 1, check.names = FALSE))
    if (nrow(amat_raw) != ncol(amat_raw)) {
      showNotification("Adjacency matrix is not square.", type = "error")
      return()
    }
    data_vars <- colnames(dat())
    missing   <- setdiff(rownames(amat_raw), data_vars)
    if (length(missing) > 0) {
      showNotification(
        paste("These nodes are not in the dataset:", paste(missing, collapse = ", ")),
        type = "error")
      return()
    }
    rv_amat(amat_raw)
    rv_manual_edges(adjacency_to_edges(amat_raw))
    updateRadioButtons(session, "dag_source", selected = "manual")
    showNotification("Adjacency matrix loaded into editor.", type = "message")
  })


  ## ---- Run CI tests ----
  observeEvent(input$btn_run_ci, {
    req(dat(), rv_amat())
    A         <- rv_amat()
    data_vars <- colnames(dat())
    A_vars    <- rownames(A)
    if (is.null(A_vars)) A_vars <- colnames(A)
    missing <- setdiff(A_vars, data_vars)
    if (length(missing) > 0) {
      showNotification(
        paste("Error: these DAG nodes are not in the data:",
              paste(missing, collapse = ", ")),
        type = "error")
      return()
    }

    out <- dag_ci_agent(dat_processed(), amat_llm = A,
                        alpha = input$alpha,
                        tests = input$tests,
                        graph_name = "Hypothesized DAG")
    rv_ci(out)

    vars <- rownames(A)
    if (is.null(vars)) vars <- colnames(A)
    rv_testable(build_testable_matrix(out$raw_results, vars))
    rv_missing(infer_missing_edges_from_ci(out$raw_results, vars))
  })


  ## ---- Consistency message ----
  output$consistency_msg <- renderPrint({
    ci <- rv_ci(); req(ci)
    n_rej <- nrow(ci$rejected)
    if (n_rej == 0) {
      cat("Result: DAG is NOT falsified at \u03B1 =", input$alpha, "\n",
          "(no CI statements were rejected).\n")
    } else {
      cat("Result: DAG is NOT consistent with the data.\n",
          n_rej, "CI statements were rejected at \u03B1 =", input$alpha, ".\n")
    }
  })


  ## ---- Tables ----
  output$tbl_summary <- renderTable({
    ci <- rv_ci(); req(ci); ci$summary
  })
  output$tbl_rejected <- renderTable({
    ci <- rv_ci(); req(ci); ci$rejected
  })
  output$llm_interpretation <- renderText({
    ci <- rv_ci(); req(ci)
    if (is.null(ci$interpretation)) "" else ci$interpretation
  })


  ## ---- CI tests to perform (computed from DAG, before running) ----
  ci_tests_to_perform <- reactive({
    A_base <- rv_amat()
    if (is.null(A_base)) return(NULL)

    g       <- adj2dag(A_base)
    cis_all <- dagitty::impliedConditionalIndependencies(g)
    cis     <- cis_all[vapply(cis_all, function(ci) length(ci$Z) > 0, logical(1))]
    if (length(cis) == 0) return(NULL)

    vars <- rownames(A_base)
    if (is.null(vars)) vars <- colnames(A_base)
    A_testable <- matrix(0L, nrow = length(vars), ncol = length(vars),
                         dimnames = list(vars, vars))

    for (ci in cis) {
      for (x in ci$X) for (y in ci$Y) {
        if (x %in% vars && y %in% vars && x != y) {
          A_testable[x, y] <- 1L
          A_testable[y, x] <- 1L
        }
      }
    }

    list(ci_tests_df = get_all_ci_tests(A_base), amat_testable = A_testable)
  })

  ## ---- CI tests table ----
  output$ci_tests_table <- DT::renderDataTable({
    ci_info <- ci_tests_to_perform()
    if (is.null(ci_info) || is.null(rv_amat())) {
      return(data.frame(Message = "No DAG defined yet.", stringsAsFactors = FALSE))
    }
    ci_tests <- ci_info$ci_tests_df
    if (nrow(ci_tests) == 0) {
      return(data.frame(
        Message = "No conditional independence tests found for this DAG.",
        stringsAsFactors = FALSE))
    }
    DT::datatable(ci_tests,
                  options = list(pageLength = 20, scrollX = TRUE,
                                 order = list(list(0, "asc"))),
                  rownames = FALSE,
                  caption  = "Implied CIs (red dotted lines in the DAG plot)")
  })


  ## ---- DAG plot 1: hypothesis + red dotted CI lines ----
  output$dag_plot <- renderPlot({
    req(dat(), rv_layout())
    coords <- rv_layout()
    A_base <- rv_amat()
    if (is.null(A_base)) {
      vars   <- colnames(dat())
      A_base <- matrix(0L, nrow = length(vars), ncol = length(vars),
                       dimnames = list(vars, vars))
    }
    vars       <- rownames(A_base)
    if (is.null(vars)) vars <- colnames(A_base)
    layout_mat <- coords[vars, , drop = FALSE]
    ci_info    <- ci_tests_to_perform()
    A_testable <- if (!is.null(ci_info)) ci_info$amat_testable else NULL

    plot_dag_with_annotations(
      amat_base     = A_base,
      amat_testable = A_testable,
      main          = "Hypothesised DAG (red dotted = CI tests to run)",
      layout        = layout_mat
    )
  })

  ## ---- DAG plot 2: clean view ----
  output$dag_plot_2 <- renderPlot({
    req(dat(), rv_layout())
    coords <- rv_layout()
    A_base <- rv_amat()
    if (is.null(A_base)) {
      vars   <- colnames(dat())
      A_base <- matrix(0L, nrow = length(vars), ncol = length(vars),
                       dimnames = list(vars, vars))
    }
    vars       <- rownames(A_base)
    if (is.null(vars)) vars <- colnames(A_base)
    layout_mat <- coords[vars, , drop = FALSE]

    plot_dag_with_annotations(
      amat_base = A_base,
      main      = "Hypothesised DAG",
      layout    = layout_mat
    )
  })


  ## ---- DAG plot with tested CIs (dotted lines, post-test view) ----
  output$dag_ci_tested_plot <- renderPlot({
    req(dat(), rv_layout(), rv_ci())
    A_base <- rv_amat()
    if (is.null(A_base)) return(invisible(NULL))

    coords     <- rv_layout()
    vars       <- rownames(A_base)
    if (is.null(vars)) vars <- colnames(A_base)
    layout_mat <- coords[vars, , drop = FALSE]

    ci_info    <- ci_tests_to_perform()
    A_testable <- if (!is.null(ci_info)) ci_info$amat_testable else NULL

    plot_dag_with_annotations(
      amat_base     = A_base,
      amat_testable = A_testable,
      main          = "CI Tests Performed",
      layout        = layout_mat
    )
  })

  ## ---- DAG plot with rejected CIs (bold solid overlay) ----
  output$dag_ci_plot <- renderPlot({
    req(dat(), rv_layout(), rv_ci())
    A_base <- rv_amat()
    if (is.null(A_base)) return(invisible(NULL))

    coords     <- rv_layout()
    vars       <- rownames(A_base)
    if (is.null(vars)) vars <- colnames(A_base)
    layout_mat <- coords[vars, , drop = FALSE]
    # Use the precomputed rejected-CI matrix directly to avoid
    # fragile string parsing / name canonicalisation mismatches.
    A_new <- rv_missing()
    if (!is.null(A_new)) {
      A_new <- A_new[vars, vars, drop = FALSE]
    }

    plot_dag_with_annotations(
      amat_base = A_base,
      amat_new  = A_new,
      main      = "DAG with Rejected CIs",
      layout    = layout_mat
    )
  })


  ## ---- Export DAG as CSV ----
  output$btn_export_dag <- downloadHandler(
    filename = function() {
      paste0("dag_adjacency_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    },
    content = function(file) {
      A <- rv_amat()
      if (is.null(A)) {
        showNotification("No DAG to export. Define a DAG first.", type = "error")
        return()
      }
      write.csv(A, file, row.names = TRUE)
    }
  )


  ## ---- Generate reproducible R script ----
  generate_repro_script <- function(A, alpha, tests, dat_df) {
    vars <- rownames(A)
    if (is.null(vars)) vars <- colnames(A)

    # Serialise the adjacency matrix as R code
    vals <- paste(as.vector(A), collapse = ", ")
    n    <- nrow(A)
    var_str <- paste0('"', paste(vars, collapse = '", "'), '"')
    amat_code <- paste0(
      'vars <- c(', var_str, ')\n',
      'A <- matrix(c(', vals, '),\n',
      '             nrow = ', n, ', ncol = ', n, ',\n',
      '             dimnames = list(vars, vars))\n'
    )

    tests_str <- paste0('c("', paste(tests, collapse = '", "'), '")')

    paste0(
'# ============================================================
# Reproducible DAG Validation Analysis
# Generated by DAG Validator Agent — ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '
#
# This script reproduces the full CI-testing analysis.
# Place it next to "data.csv" and "dag_adjacency.csv"
# (both included in this ZIP), then run:
#   Rscript reproduce_analysis.R
# ============================================================

# ---- 1. Required packages ----
library(dplyr)
library(tibble)
library(igraph)
library(comets)
library(dagitty)

# ---- 2. Load data ----
dat <- read.csv("data.csv", check.names = TRUE)
cat("Loaded data:", nrow(dat), "rows,", ncol(dat), "columns\\n")
cat("Variables:", paste(colnames(dat), collapse = ", "), "\\n\\n")

# ---- 3. Define adjacency matrix (DAG hypothesis) ----
', amat_code, '
cat("DAG edges:", sum(A), "\\n\\n")

# ---- 4. Helper functions ----

# Convert adjacency matrix to dagitty DAG
adj2dag <- function(adj) {
  nodes <- rownames(adj)
  if (is.null(nodes)) nodes <- colnames(adj)
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

# Extract implied CIs with non-empty conditioning set
cis_with_Z <- function(amat) {
  g   <- adj2dag(amat)
  cis <- dagitty::impliedConditionalIndependencies(g)
  cis[vapply(cis, function(x) length(x$Z) > 0, logical(1))]
}

# Run COMETs CI tests
run_ci_tests <- function(amat, dat, tests = c("gcm", "pcm"), alpha = 0.05) {
  tcis <- cis_with_Z(amat)
  if (length(tcis) == 0) {
    return(tibble::tibble(
      test = character(0), CI = character(0),
      p.value = numeric(0), adj.p.value = numeric(0), rejected = logical(0)
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
      test = tst, CI = vapply(tcis, paste, character(1)),
      p.value = pv,
      adj.p.value = stats::p.adjust(pv, "holm"),
      rejected = stats::p.adjust(pv, "holm") < alpha
    )
  }) |> dplyr::bind_rows()
  res
}

# Plot DAG with CI-test overlay
plot_dag_with_ci <- function(amat, layout, ci_results = NULL, main = "DAG") {
  g <- igraph::graph_from_adjacency_matrix(amat, mode = "directed")
  lay <- layout[igraph::V(g)$name, , drop = FALSE]
  xlim <- range(lay[, 1]) * 1.2
  ylim <- range(lay[, 2]) * 1.2
  op <- par(mar = c(4, 1, 2, 1))
  on.exit(par(op), add = TRUE)
  plot(g, main = main, vertex.label.cex = 1.1, vertex.size = 25,
       edge.arrow.size = 0.4, layout = lay,
       rescale = FALSE, xlim = xlim, ylim = ylim)
  if (!is.null(ci_results)) {
    rejs <- ci_results[ci_results$rejected, , drop = FALSE]
    if (nrow(rejs) > 0) {
      vs <- rownames(amat)
      for (k in seq_len(nrow(rejs))) {
        ci_str <- rejs$CI[k]
        main_part <- sub("\\\\s*\\\\|.*$", "", ci_str)
        main_part <- gsub("_\\\\|\\\\|_", "\\u27C2", main_part)
        parts <- strsplit(main_part, "\\u27C2", fixed = TRUE)[[1]]
        parts <- trimws(parts)
        if (length(parts) < 2) next
        X <- parts[1]; Y <- parts[2]
        i <- match(X, vs); j <- match(Y, vs)
        if (!is.na(i) && !is.na(j)) {
          segments(lay[i,1], lay[i,2], lay[j,1], lay[j,2],
                   col = "red", lty = 1, lwd = 4)
        }
      }
    }
  }
  par(xpd = TRUE)
  legend("bottom", legend = c("DAG edges", "Rejected CIs"),
         col = c("black", "red"), lty = c(1, 1), lwd = c(2, 4),
         bg = "white", box.col = "grey80", cex = 0.95,
         horiz = TRUE, inset = c(0, -0.15))
}

# ---- 5. Parameters ----
alpha <- ', alpha, '
tests <- ', tests_str, '

# ---- 6. Circular layout ----
n      <- length(vars)
angles <- seq(from = pi / 2, by = -2 * pi / n, length.out = n)
coords <- cbind(x = cos(angles), y = sin(angles))
rownames(coords) <- vars

# ---- 7. List implied CI statements ----
g_dagitty <- adj2dag(A)
cis_all   <- dagitty::impliedConditionalIndependencies(g_dagitty)
cis       <- cis_all[vapply(cis_all, function(ci) length(ci$Z) > 0, logical(1))]
cat("Implied CI statements (with non-empty Z):", length(cis), "\\n")
for (ci in cis) cat(" ", paste(ci), "\\n")
cat("\\n")

# ---- 8. Run CI tests ----
cat("Running CI tests (", paste(tests, collapse = ", "), ") at alpha =", alpha, "...\\n")
ci_results <- run_ci_tests(A, dat, tests = tests, alpha = alpha)

# Summary
summary_tbl <- ci_results |>
  dplyr::group_by(test) |>
  dplyr::summarise(
    tests       = dplyr::n(),
    rejected    = sum(rejected),
    `min adj.p` = sprintf("%.4f", min(adj.p.value, na.rm = TRUE)),
    .groups     = "drop"
  )
cat("\\n--- Summary ---\\n")
print(summary_tbl, n = Inf)

# Rejected CIs
rejected_tbl <- ci_results |>
  dplyr::filter(rejected) |>
  dplyr::arrange(adj.p.value) |>
  dplyr::mutate(`adj.p` = sprintf("%.4f", adj.p.value)) |>
  dplyr::select(test, CI, `adj.p`)
cat("\\n--- Rejected CIs ---\\n")
if (nrow(rejected_tbl) > 0) {
  print(rejected_tbl, n = Inf)
} else {
  cat("None — DAG is not falsified at alpha =", alpha, "\\n")
}

# ---- 9. Verdict ----
n_rej <- sum(ci_results$rejected)
cat("\\n--- Verdict ---\\n")
if (n_rej == 0) {
  cat("DAG is NOT falsified at alpha =", alpha,
      "(no CI statements rejected).\\n")
} else {
  cat("DAG is NOT consistent with the data.",
      n_rej, "CI statement(s) rejected at alpha =", alpha, ".\\n")
}

# ---- 10. Save results ----
write.csv(ci_results, "ci_results_all.csv", row.names = FALSE)
write.csv(as.data.frame(summary_tbl), "ci_summary.csv", row.names = FALSE)
if (nrow(rejected_tbl) > 0) {
  write.csv(as.data.frame(rejected_tbl), "ci_rejected.csv", row.names = FALSE)
}
cat("\\nResults saved to CSV files.\\n")

# ---- 11. Plot ----
pdf("dag_validation_plot.pdf", width = 10, height = 5)
par(mfrow = c(1, 2))
plot_dag_with_ci(A, coords, main = "Hypothesised DAG")
plot_dag_with_ci(A, coords, ci_results = ci_results,
                 main = "DAG with Rejected CIs")
dev.off()
cat("Plot saved to dag_validation_plot.pdf\\n")

cat("\\n=== Done ===\\n")
')
  }


  ## ---- Export full report as ZIP ----
  output$btn_export_report <- downloadHandler(
    filename = function() {
      paste0("DAG_Validation_Report_",
             format(Sys.time(), "%Y%m%d_%H%M%S"), ".zip")
    },
    content = function(file) {
      A <- rv_amat()
      if (is.null(A)) {
        showNotification("No DAG to export. Define a DAG first.", type = "error")
        return()
      }

      # Create temp directory for all files
      tmp_dir <- file.path(tempdir(), paste0("report_", format(Sys.time(), "%Y%m%d_%H%M%S")))
      dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

      # 1) Processed data (after any preprocessing applied)
      write.csv(dat_processed(), file.path(tmp_dir, "data.csv"), row.names = FALSE)

      # 2) DAG adjacency matrix
      write.csv(A, file.path(tmp_dir, "dag_adjacency.csv"), row.names = TRUE)

      # 3) CI results (if available)
      ci <- rv_ci()
      ci_all_raw   <- NULL
      ci_summary   <- NULL
      ci_rejected  <- NULL
      interp_text  <- ""

      if (!is.null(ci)) {
        if (!is.null(ci$raw_results) && nrow(ci$raw_results) > 0) {
          ci_all_raw <- ci$raw_results
          write.csv(ci$raw_results,
                    file.path(tmp_dir, "ci_results_all.csv"),
                    row.names = FALSE)
        }
        if (!is.null(ci$summary) && nrow(ci$summary) > 0) {
          ci_summary <- ci$summary
          write.csv(ci$summary,
                    file.path(tmp_dir, "ci_summary.csv"),
                    row.names = FALSE)
        }
        if (!is.null(ci$rejected) && nrow(ci$rejected) > 0) {
          ci_rejected <- ci$rejected
          write.csv(ci$rejected,
                    file.path(tmp_dir, "ci_rejected.csv"),
                    row.names = FALSE)
        }
        if (!is.null(ci$interpretation) && nzchar(ci$interpretation)) {
          interp_text <- ci$interpretation
          writeLines(interp_text,
                     file.path(tmp_dir, "llm_interpretation.txt"))
        }
      }

      # 4) Reproducible R script
      writeLines(
        generate_repro_script(A, input$alpha, input$tests, dat_processed()),
        file.path(tmp_dir, "reproduce_analysis.R")
      )

      # 5) Render PDF report
      rmd_src <- file.path(getwd(), "report_template.Rmd")
      if (file.exists(rmd_src)) {
        rmd_copy <- file.path(tmp_dir, "report_template.Rmd")
        file.copy(rmd_src, rmd_copy, overwrite = TRUE)

        tryCatch({
          rmarkdown::render(
            input       = rmd_copy,
            output_file = "report.pdf",
            output_dir  = tmp_dir,
            params = list(
              amat           = A,
              amat_testable  = {
                ci_info <- ci_tests_to_perform()
                if (!is.null(ci_info)) ci_info$amat_testable else NULL
              },
              ci_summary     = ci_summary,
              ci_rejected    = ci_rejected,
              ci_all         = ci_all_raw,
              alpha          = input$alpha,
              tests          = paste(input$tests, collapse = ", "),
              interpretation = interp_text,
              layout_mat     = rv_layout()
            ),
            envir = new.env(parent = globalenv()),
            quiet = TRUE
          )
        }, error = function(e) {
          showNotification(
            paste("PDF generation failed:", e$message,
                  "\nCSV files are still included in the ZIP."),
            type = "warning", duration = 10)
        })

        # Clean up the copied Rmd
        unlink(rmd_copy)
      }

      # 6) Zip everything
      files_to_zip <- list.files(tmp_dir, full.names = TRUE)
      zip::zip(file, files = basename(files_to_zip), root = tmp_dir)
    }
  )
}


## ======================================================
## Launch
## ======================================================
shinyApp(ui, server)
