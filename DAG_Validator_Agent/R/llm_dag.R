# ============================================================
# R/llm_dag.R — LLM-powered DAG proposal & CI interpretation
# ============================================================

#' Ask the LLM to propose a DAG as an adjacency matrix
#'
#' @param vars        Character vector of variable names.
#' @param expert_text Optional free-text expert knowledge.
#' @param model       Ollama model name.
#' @param temperature Sampling temperature.
#' @return Adjacency matrix A where A[i,j]==1 means i->j.
propose_dag_from_llm <- function(vars,
                                 expert_text = NULL,
                                 model = .ollama_dag_model(),
                                 temperature = 0.2) {
  empty_A <- matrix(0L, nrow = length(vars), ncol = length(vars),
                    dimnames = list(vars, vars))

  if (!.ollama_available()) {
    message("Ollama not reachable - falling back to empty DAG.")
    return(list(A = empty_A, error = "Ollama is not running. Start it with: ollama serve"))
  }

  sys_msg <- paste(
    "You are an expert in causal discovery and cell signaling.",
    "You will be given a list of variable names from a biological or medical dataset,",
    "optionally with expert background knowledge.",
    "Your task is to propose a plausible directed acyclic graph (DAG)",
    "representing causal relations between these variables.",
    "",
    "IMPORTANT OUTPUT FORMAT:",
    "- If you can recognise the variables and infer plausible causal relations,",
    "  output ONLY directed edges, one per line: Parent,Child",
    "- Use ONLY variable names from the provided list.",
    "- Do NOT include any headers, explanations, or extra text.",
    "- The edges must define a DAG (no directed cycles).",
    "",
    "- If the variable names are cryptic, generic (e.g. V1, V2, X1, X2),",
    "  or you cannot determine any meaningful causal structure from the names alone,",
    "  output EXACTLY the single line: NO_DAG_POSSIBLE",
    "  Do NOT guess random edges when you have no domain knowledge about the variables."
  )

  expert_block <- if (!is.null(expert_text) && nzchar(expert_text)) {
    paste0("\n\nExpert background knowledge (free text):\n", expert_text)
  } else {
    ""
  }

  usr_msg <- paste0(
    "Variable names:\n",
    paste(vars, collapse = ", "),
    expert_block,
    "\n\nPropose a plausible DAG for these variables in the exact format:\n",
    "Parent,Child\nParent,Child\n...\n"
  )

  txt <- tryCatch(
    openai_chat(sys_msg, usr_msg, model = model, temperature = temperature),
    error = function(e) {
      message("[LLM] DAG proposal failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(txt) || !nzchar(txt)) {
    msg <- paste0(
      "LLM did not respond (it may be too slow on CPU-only Docker).\n",
      "Try again, or use 'Upload adjacency matrix' / 'Manual editor' instead."
    )
    message(msg)
    return(list(A = empty_A, error = msg))
  }

  # Check if LLM explicitly signalled it cannot propose a DAG
  txt_clean <- trimws(txt)
  if (grepl("NO_DAG_POSSIBLE", txt_clean, fixed = TRUE)) {
    msg <- paste0(
      "The LLM could not infer a causal DAG from the variable names alone.\n",
      "The names may be too generic or cryptic (e.g. V1, V2, X1, X2).\n\n",
      "Suggestions:\n",
      "- Provide expert knowledge in the text box to guide the LLM.\n",
      "- Rename your columns to meaningful names before uploading.\n",
      "- Use 'Upload adjacency matrix' or 'Manual editor' instead.\n",
      "- Try a larger Ollama model (e.g. llama3.1:70b) for better domain knowledge."
    )
    message(msg)
    return(list(A = empty_A, error = msg))
  }

  # Parse "Parent,Child" lines
  lines <- strsplit(txt, "\n", fixed = TRUE)[[1]]
  lines <- trimws(lines)
  lines <- lines[nzchar(lines)]

  A <- empty_A
  for (ln in lines) {
    parts  <- strsplit(ln, ",", fixed = TRUE)[[1]]
    if (length(parts) != 2) next
    parent <- trimws(parts[1])
    child  <- trimws(parts[2])
    if (!(parent %in% vars) || !(child %in% vars)) next
    if (parent == child) next
    A[parent, child] <- 1L
  }

  # If the LLM produced text but we couldn't parse any valid edges
  if (sum(A) == 0) {
    msg <- paste0(
      "The LLM response did not contain any valid edges.\n",
      "This usually means the variable names are not recognisable ",
      "as domain-specific concepts.\n\n",
      "Suggestions:\n",
      "- Add expert knowledge describing the expected causal relationships.\n",
      "- Rename columns to meaningful names before uploading.\n",
      "- Use 'Upload adjacency matrix' or 'Manual editor' instead."
    )
    message(msg)
    return(list(A = empty_A, error = msg))
  }

  # Acyclicity check
  g <- igraph::graph_from_adjacency_matrix(A, mode = "directed")
  if (!igraph::is_dag(g)) {
    warning("LLM-proposed graph contains cycles. ",
            "You may need to adjust the prompt or post-process edges.")
  }

  message("LLM DAG proposal: ", sum(A), " directed edges created.")
  list(A = A, error = NULL)
}


#' Run CI tests and have the LLM explain the results
#'
#' @param dat       Data frame.
#' @param amat_llm  Adjacency matrix.
#' @param alpha     Significance level.
#' @param tests     Character vector of test types.
#' @param graph_name Label for the graph.
#' @return List with raw_results, summary, rejected, interpretation.
dag_ci_agent <- function(dat, amat_llm,
                         alpha = 0.05,
                         tests = c("gcm", "pcm"),
                         graph_name = "Hypothesized DAG") {

  # ---- statistical testing ----
  res <- run_ci_tests(amat_llm, dat, tests = tests, alpha = alpha)

  summary_tbl <- res |>
    dplyr::group_by(test) |>
    dplyr::summarise(
      `tests`    = dplyr::n(),
      `rejected` = sum(rejected),
      `min adj. p` = sprintf("%.4f", min(adj.p.value, na.rm = TRUE)),
      .groups    = "drop"
    )

  rejected_tbl <- res |>
    dplyr::filter(rejected) |>
    dplyr::arrange(adj.p.value) |>
    dplyr::mutate(`adj. p-value` = sprintf("%.4f", adj.p.value)) |>
    dplyr::select(test, CI, `adj. p-value`)

  # ---- LLM interpretation ----
  smry_txt <- paste(capture.output(print(summary_tbl, n = Inf)), collapse = "\n")
  rej_txt  <- if (nrow(rejected_tbl)) {
    paste(capture.output(print(rejected_tbl, n = min(10, nrow(rejected_tbl)))),
          collapse = "\n")
  } else {
    "<none>"
  }

  sys_msg <- paste(
    "You are a statistician and medical data scientist.",
    "You explain results of conditional independence (CI) tests used to",
    "evaluate causal assumptions in biological or clinical datasets.",
    "Your audience is medical professionals and researchers with basic",
    "statistical knowledge but not experts in causal inference.",
    "",
    "STRICT FORMATTING RULES (you must follow these exactly):",
    "- Write 3-4 short bullet points, maximum 5.",
    "- Each bullet is one sentence, two at most.",
    "- Do NOT use markdown formatting: no bold (**), no italic (*), no headers (#).",
    "- Use plain text only. Start each bullet with a dash (-).",
    "- Keep the total response under 120 words.",
    "- Do NOT repeat the numbers from the summary table; the user can see them.",
    "- Prefer plain language, explain any technical term briefly inline.",
    "",
    "CONTENT FOCUS:",
    "1) One-sentence verdict: is the DAG broadly compatible or clearly contradicted?",
    "2) Name the 1-2 most important rejected assumptions and what they mean.",
    "3) One sentence reminding that passing these tests does not prove the DAG is correct.",
    "4) Optionally one concrete next step (e.g. add a specific edge)."
  )

  usr_msg <- paste0(
    "We tested whether the DAG '", graph_name, "' is consistent with this ",
    "dataset using CI tests.\n",
    sprintf("Significance level alpha = %.3f.\n\n", alpha),
    "Summary of CI tests per test type:\n", smry_txt, "\n\n",
    "Falsified independence assumptions (Holm-adjusted p-values):\n",
    rej_txt, "\n\n",
    "Give a short plain-text interpretation (no markdown formatting, no bold, ",
    "no asterisks). 3-4 bullet points, under 120 words total."
  )

  interp <- tryCatch(
    openai_chat(sys_msg, usr_msg),
    error = function(e) {
      message("[LLM] Interpretation failed (", conditionMessage(e),
              "). CI results are still available.")
      NULL
    }
  )

  list(
    raw_results    = res,
    summary        = summary_tbl,
    rejected       = rejected_tbl,
    interpretation = interp
  )
}
