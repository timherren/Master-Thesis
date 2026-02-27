#!/usr/bin/env python3
"""Local MCP wrapper for TRAM-DAG tools (kept inside causal_ai_chatbot)."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tramdag import TramDagConfig, TramDagModel
from tramdag.utils.configuration import (
    create_node_dict,
    create_nn_model_names,
    load_configuration_dict,
    read_adj_matrix_from_configuration,
    read_nn_names_matrix_from_configuration,
    write_adj_matrix_to_configuration,
    write_configuration_dict,
    write_nn_names_matrix_to_configuration,
    write_nodes_information_to_configuration,
)

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - optional dependency
    gaussian_kde = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _jsonify(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonify(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_jsonify(x) for x in v]
    if isinstance(v, tuple):
        return [_jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _read_data(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _deterministic_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_n = int(0.8 * n)
    remain = n - train_n
    val_n = remain // 2
    train_idx = perm[:train_n]
    val_idx = perm[train_n:train_n + val_n]
    test_idx = perm[train_n + val_n:]
    return (
        df.iloc[train_idx].copy(),
        df.iloc[val_idx].copy(),
        df.iloc[test_idx].copy(),
    )


def _save_current_figure(path: Path, width_in: float | None = None, height_in: float | None = None, dpi: int = 120) -> None:
    fig = plt.gcf()
    if width_in is not None and height_in is not None:
        fig.set_size_inches(width_in, height_in)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_dag_structure(dag: Dict[str, Any], out_path: Path) -> None:
    variables = dag.get("variables", [])
    edges = dag.get("edges", [])
    g = nx.DiGraph()
    g.add_nodes_from(variables)
    g.add_edges_from(edges)

    n = max(len(variables), 1)
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, n, endpoint=False)
    pos = {v: (float(np.cos(angles[i])), float(np.sin(angles[i]))) for i, v in enumerate(variables)}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(
        g, pos, node_size=1800, node_color="lightblue", edgecolors="darkblue", linewidths=1.5
    )
    nx.draw_networkx_labels(g, pos, font_size=11, font_color="black")
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=20, edge_color="gray", width=2.0)
    plt.title("Causal DAG Structure")
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _artifact_entry(kind: str, path: Path) -> Dict[str, Any]:
    return {"kind": kind, "path": str(path)}


def _to_numpy_array(obj: Any) -> np.ndarray:
    """Convert tensor/array/list-like values to a 1D numpy array."""
    if obj is None:
        return np.array([], dtype=float)
    # torch tensors or tensor-like objects
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            arr = obj.detach().cpu().numpy()
            return np.asarray(arr).reshape(-1)
        except Exception:
            pass
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            arr = obj.cpu().numpy()
            return np.asarray(arr).reshape(-1)
        except Exception:
            pass
    if hasattr(obj, "numpy"):
        try:
            arr = obj.numpy()
            return np.asarray(arr).reshape(-1)
        except Exception:
            pass
    try:
        return np.asarray(obj).reshape(-1)
    except Exception:
        return np.array([], dtype=float)


def _normalize_sample_outputs(sampled: Any, latents: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize model.sample outputs to dictionaries keyed by variable.
    Handles occasional wrapper/library return shape differences.
    """
    sampled_dict = sampled if isinstance(sampled, dict) else {}
    latents_dict = latents if isinstance(latents, dict) else {}

    if not sampled_dict and isinstance(sampled, (list, tuple)) and len(sampled) >= 1 and isinstance(sampled[0], dict):
        sampled_dict = sampled[0]
    if not latents_dict and isinstance(sampled, (list, tuple)) and len(sampled) >= 2 and isinstance(sampled[1], dict):
        latents_dict = sampled[1]
    if not latents_dict and isinstance(latents, (list, tuple)) and len(latents) >= 1 and isinstance(latents[0], dict):
        latents_dict = latents[0]

    return sampled_dict, latents_dict


def _ordered_columns(experiment_dir: Path, fallback: List[str]) -> List[str]:
    test_path = experiment_dir / "chatbot_splits" / "test.csv"
    try:
        if test_path.exists():
            cols = list(pd.read_csv(test_path, nrows=1).columns)
            if cols:
                return cols
    except Exception:
        pass
    return fallback


def _collect_dag_edges(dag: Dict[str, Any], variables: List[str]) -> List[Tuple[str, str]]:
    vars_set = set(variables)
    edges: set[Tuple[str, str]] = set()

    raw_edges = dag.get("edges", []) or []
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            src, dst = str(e[0]), str(e[1])
            if src in vars_set and dst in vars_set and src != dst:
                edges.add((src, dst))

    # Fallback: derive from adjacency_matrix if needed.
    adj = dag.get("adjacency_matrix")
    if isinstance(adj, list) and len(adj) == len(variables):
        for i, row in enumerate(adj):
            if not isinstance(row, list) or len(row) != len(variables):
                continue
            for j, val in enumerate(row):
                try:
                    edge_on = int(val) != 0
                except Exception:
                    edge_on = bool(val)
                if edge_on and i != j:
                    src, dst = variables[i], variables[j]
                    if src in vars_set and dst in vars_set:
                        edges.add((src, dst))

    return sorted(edges)


def _build_tram_ready_adjacency(dag: Dict[str, Any], df_columns: List[str]) -> Tuple[List[str], np.ndarray]:
    vars_from_dag = [str(v) for v in (dag.get("variables") or []) if str(v) in df_columns]
    if not vars_from_dag:
        vars_from_dag = [str(c) for c in df_columns]

    edges = _collect_dag_edges(dag, vars_from_dag)
    g = nx.DiGraph()
    g.add_nodes_from(vars_from_dag)
    g.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g, orientation="original")
            cycle_edges = [f"{u}->{v}" for u, v, _ in cycle]
            raise ValueError(
                "DAG contains a directed cycle and cannot be fitted: "
                + ", ".join(cycle_edges)
            )
        except ValueError:
            raise
        except Exception:
            raise ValueError("DAG contains a directed cycle and cannot be fitted.")

    try:
        ordered_vars = list(nx.topological_sort(g))
    except Exception:
        ordered_vars = list(vars_from_dag)

    idx = {v: i for i, v in enumerate(ordered_vars)}
    adj_codes = np.full((len(ordered_vars), len(ordered_vars)), "0", dtype=object)
    for src, dst in edges:
        i = idx.get(src)
        j = idx.get(dst)
        if i is None or j is None or i == j:
            continue
        if i >= j:
            raise ValueError(
                f"Failed to construct upper-triangular TRAM adjacency for edge {src}->{dst}."
            )
        adj_codes[i, j] = "ls"
    return ordered_vars, adj_codes


def _plot_samples_vs_true_overlay(
    test_df: pd.DataFrame,
    sampled_dict: Dict[str, Any],
    out_path: Path,
    width_per_var: float = 4.5,
) -> bool:
    cols = [c for c in list(test_df.columns) if c in sampled_dict]
    if not cols:
        return False

    fig_w = max(width_per_var * len(cols), 10.0)
    fig, axes = plt.subplots(1, len(cols), figsize=(fig_w, 5.0))
    if len(cols) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for i, col in enumerate(cols):
        ax = axes[i]
        true_vals = pd.to_numeric(test_df[col], errors="coerce").dropna().to_numpy()
        samp_vals = _to_numpy_array(sampled_dict[col])
        samp_vals = samp_vals[np.isfinite(samp_vals)]
        if len(true_vals) == 0 or len(samp_vals) == 0:
            ax.set_title(str(col))
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
            ax.set_axis_off()
            continue

        all_vals = np.concatenate([true_vals, samp_vals])
        lo = float(np.min(all_vals))
        hi = float(np.max(all_vals))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 80)

        ax.hist(true_vals, bins=bins, density=True, alpha=0.5, color="steelblue", label="Test")
        ax.hist(samp_vals, bins=bins, density=True, alpha=0.5, color="darkorange", label="Sampled")

        if gaussian_kde is not None and len(np.unique(true_vals)) > 1 and len(np.unique(samp_vals)) > 1:
            x_grid = np.linspace(lo, hi, 300)
            try:
                ax.plot(x_grid, gaussian_kde(true_vals)(x_grid), color="steelblue", linestyle="--", lw=2)
                ax.plot(x_grid, gaussian_kde(samp_vals)(x_grid), color="darkorange", linestyle="--", lw=2)
            except Exception:
                pass

        ax.axvline(float(np.mean(true_vals)), color="steelblue", linestyle=":", lw=1.5, alpha=0.8)
        ax.axvline(float(np.mean(samp_vals)), color="darkorange", linestyle=":", lw=1.5, alpha=0.8)
        ax.set_title(str(col), fontsize=12)
        ax.set_xlabel(str(col), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)

    fig.suptitle("Samples vs True (Test Data vs Model Samples)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_latent_distributions(
    latents_dict: Dict[str, Any],
    variables: List[str],
    out_path: Path,
) -> bool:
    cols = [v for v in variables if v in latents_dict]
    if not cols:
        return False

    fig_w = max(4.5 * len(cols), 10.0)
    fig, axes = plt.subplots(1, len(cols), figsize=(fig_w, 5.0))
    if len(cols) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for i, col in enumerate(cols):
        ax = axes[i]
        vals = _to_numpy_array(latents_dict.get(col))
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.set_title(f"{col} (no latent values)")
            ax.set_axis_off()
            continue
        ax.hist(vals, bins=80, density=True, alpha=0.7, color="steelblue")
        if gaussian_kde is not None and len(np.unique(vals)) > 1:
            try:
                x_grid = np.linspace(float(np.min(vals)), float(np.max(vals)), 300)
                ax.plot(x_grid, gaussian_kde(vals)(x_grid), color="#154360", lw=2)
            except Exception:
                pass
        ax.set_title(str(col), fontsize=12)
        ax.set_xlabel("Latent value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Latent Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _maybe_plot_model_figure(model: TramDagModel, method_name: str, out_path: Path, width: float, height: float, dpi: int) -> bool:
    if not hasattr(model, method_name):
        return False
    try:
        method = getattr(model, method_name)
        method()
        _save_current_figure(out_path, width, height, dpi)
        return True
    except Exception:
        plt.close("all")
        return False


def _fit_model(payload: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    dag = payload["dag"]
    data_path = payload["data_path"]
    experiment_dir = Path(payload["experiment_dir"])
    epochs = int(payload.get("epochs", 100))
    learning_rate = float(payload.get("learning_rate", 0.01))
    batch_size = int(payload.get("batch_size", 512))
    seed = int(payload.get("random_seed", 42))

    _set_deterministic(seed)
    df = _read_data(data_path)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ordered_vars, adj_codes = _build_tram_ready_adjacency(dag, list(df.columns))
    # Keep data columns aligned with adjacency/data_type ordering expected by tramdag.
    df = df[ordered_vars].copy()

    cfg = TramDagConfig()
    cfg.setup_configuration(experiment_name=experiment_dir.name, EXPERIMENT_DIR=str(experiment_dir))

    data_types: Dict[str, str] = {}
    for col in df.columns:
        if str(df[col].dtype).startswith("int") and df[col].nunique() < 20:
            data_types[col] = "ordinal_Xn_Yo"
        else:
            data_types[col] = "continous"
    cfg.set_data_type(data_types)

    write_adj_matrix_to_configuration(adj_codes, cfg.CONF_DICT_PATH)
    cfg.update()
    nn_names = create_nn_model_names(adj_codes, data_types)
    # Some tramdag builds can return None unexpectedly; create a safe default.
    if nn_names is None:
        nn_names = np.full_like(adj_codes, "0", dtype=object)
        for i in range(adj_codes.shape[0]):
            for j in range(adj_codes.shape[1]):
                code = str(adj_codes[i, j])
                if code == "0":
                    continue
                if code.startswith("ls"):
                    nn_names[i, j] = "LinearShift"
                elif code.startswith("si"):
                    nn_names[i, j] = "SimpleIntercept"
                elif code.startswith("cs"):
                    nn_names[i, j] = "ComplexShiftDefaultTabular"
                elif code.startswith("ci"):
                    nn_names[i, j] = "ComplexInterceptDefaultTabular"
                else:
                    nn_names[i, j] = code
    write_nn_names_matrix_to_configuration(nn_names, cfg.CONF_DICT_PATH)
    cfg.update()
    write_nodes_information_to_configuration(cfg.CONF_DICT_PATH)
    cfg.update()

    # Repair buggy configs where tramdag leaves nodes as null.
    conf = load_configuration_dict(cfg.CONF_DICT_PATH)
    if conf.get("nodes") is None:
        adj_from_conf = read_adj_matrix_from_configuration(cfg.CONF_DICT_PATH)
        nn_from_conf = read_nn_names_matrix_from_configuration(cfg.CONF_DICT_PATH)
        if nn_from_conf is None:
            raise RuntimeError("tramdag config has model_names=None; cannot build nodes dictionary.")
        conf["nodes"] = create_node_dict(
            adj_from_conf,
            nn_from_conf,
            data_types,
            min_vals=None,
            max_vals=None,
            levels_dict=None,
        )
        write_configuration_dict(conf, cfg.CONF_DICT_PATH)
        cfg.update()
    if cfg.conf_dict.get("nodes") is None:
        raise RuntimeError("tramdag configuration repair failed: nodes is still None.")

    if any("ordinal" in dt for dt in data_types.values()):
        cfg.compute_levels(df)
    cfg.save()

    model = TramDagModel.from_config(cfg, device="auto", verbose=True)
    train_df, val_df, test_df = _deterministic_split(df, seed)
    split_dir = experiment_dir / "chatbot_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    model.load_or_compute_minmax(td_train_data=train_df, use_existing=False, write=True)
    model.fit(
        train_data=train_df,
        val_data=val_df,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=True,
    )

    artifacts: List[str] = []
    manifest: List[Dict[str, Any]] = []

    loss_path = artifact_dir / "loss_history.png"
    if _maybe_plot_model_figure(model, "plot_loss_history", loss_path, 14.0, 10.0, 120):
        artifacts.append(str(loss_path))
        manifest.append(_artifact_entry("loss_history", loss_path))

    shift_path = artifact_dir / "linear_shift_history.png"
    if _maybe_plot_model_figure(model, "plot_linear_shift_history", shift_path, 12.0, 5.0, 120):
        artifacts.append(str(shift_path))
        manifest.append(_artifact_entry("linear_shift_history", shift_path))

    intercept_path = artifact_dir / "simple_intercepts_history.png"
    if _maybe_plot_model_figure(model, "plot_simple_intercepts_history", intercept_path, 12.0, 5.0, 120):
        artifacts.append(str(intercept_path))
        manifest.append(_artifact_entry("simple_intercepts_history", intercept_path))

    if hasattr(model, "plot_hdag"):
        try:
            model.plot_hdag(train_df, list(df.columns), plot_n_rows=1)
            hdag_path = artifact_dir / "hdag.png"
            _save_current_figure(hdag_path, 14.0, 5.0, 100)
            artifacts.append(str(hdag_path))
            manifest.append(_artifact_entry("hdag", hdag_path))
        except Exception:
            plt.close("all")

    # Build latent distributions from direct sampled latents to ensure all variables are shown.
    try:
        _set_deterministic(seed)
        _, fit_latents_raw = model.sample(number_of_samples=max(len(train_df), 1000))
        _, fit_latents = _normalize_sample_outputs({}, fit_latents_raw)
        latents_path = artifact_dir / "latents.png"
        if _plot_latent_distributions(fit_latents, list(df.columns), latents_path):
            artifacts.append(str(latents_path))
            manifest.append(_artifact_entry("latents", latents_path))
    except Exception:
        plt.close("all")

    # Match TRAM left-panel flow: show samples-vs-true after fitting.
    try:
        _set_deterministic(seed)
        sampled_obs_raw, _ = model.sample(number_of_samples=len(test_df))
        sampled_obs, _ = _normalize_sample_outputs(sampled_obs_raw, None)
        svt_path = artifact_dir / "samples_vs_true.png"
        if _plot_samples_vs_true_overlay(test_df, sampled_obs, svt_path):
            artifacts.append(str(svt_path))
            manifest.append(_artifact_entry("samples_vs_true", svt_path))
    except Exception:
        plt.close("all")

    dag_path = artifact_dir / "dag_structure.png"
    try:
        _plot_dag_structure(dag, dag_path)
        artifacts.append(str(dag_path))
        manifest.append(_artifact_entry("dag_structure", dag_path))
    except Exception:
        plt.close("all")

    return {
        "data": {
            "experiment_dir": str(experiment_dir),
            "config_path": cfg.CONF_DICT_PATH,
            "loss_history": _jsonify(model.loss_history()),
            "random_seed": seed,
            "split_paths": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
            "artifact_manifest": manifest,
        },
        "artifacts": artifacts,
    }


def _sample(payload: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    seed = int(payload.get("random_seed", 42))
    _set_deterministic(seed)

    model = TramDagModel.from_directory(payload["experiment_dir"], device="auto")
    sampled_raw, latents_raw = model.sample(
        do_interventions=payload.get("do_interventions") or {},
        number_of_samples=int(payload.get("n_samples", 10000)),
    )
    sampled, latents = _normalize_sample_outputs(sampled_raw, latents_raw)
    out: Dict[str, Any] = {}
    for node in sampled.keys():
        values = _to_numpy_array(sampled[node])
        if values.size == 0:
            continue
        out[node] = {
            "values": values.tolist(),
            "summary": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            },
            "latents": _to_numpy_array(latents.get(node)).tolist() if isinstance(latents, dict) and latents.get(node) is not None else None,
        }

    artifacts: List[str] = []
    manifest: List[Dict[str, Any]] = []

    # Canonical "samples vs true" plot if split file is available.
    experiment_dir = Path(payload["experiment_dir"])
    test_path = experiment_dir / "chatbot_splits" / "test.csv"
    if test_path.exists():
        try:
            test_df = pd.read_csv(test_path)
            svt_path = artifact_dir / "samples_vs_true.png"
            if _plot_samples_vs_true_overlay(test_df, sampled, svt_path):
                artifacts.append(str(svt_path))
                manifest.append(_artifact_entry("samples_vs_true", svt_path))
        except Exception:
            plt.close("all")

    # Deterministic distribution histograms from sampled values.
    sample_keys = list(sampled.keys())
    ordered = [c for c in _ordered_columns(experiment_dir, sample_keys) if c in sampled]
    nodes = [c for c in ordered if _to_numpy_array(sampled.get(c)).size > 0]
    if nodes:
        fig, axes = plt.subplots(1, len(nodes), figsize=(max(4.5 * len(nodes), 10.0), 5.0))
        if len(nodes) == 1:
            axes = [axes]
        for i, node in enumerate(nodes):
            vals = _to_numpy_array(sampled[node])
            axes[i].hist(vals, bins=80, density=True, alpha=0.7, color="steelblue")
            axes[i].set_title(str(node))
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
            axes[i].grid(alpha=0.2)
        fig.tight_layout()
        dist_path = artifact_dir / "sampling_distributions.png"
        fig.savefig(dist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(str(dist_path))
        manifest.append(_artifact_entry("sampling_distributions", dist_path))

    return {
        "data": {
            "samples": out,
            "random_seed": seed,
            "artifact_manifest": manifest,
        },
        "artifacts": artifacts,
    }


def _compute_ate(payload: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    model = TramDagModel.from_directory(payload["experiment_dir"], device="auto")
    X = payload["X"]
    Y = payload["Y"]
    x_treated = float(payload.get("x_treated", 1.0))
    x_control = float(payload.get("x_control", 0.0))
    n_samples = int(payload.get("n_samples", 10000))
    seed = int(payload.get("random_seed", 42))

    _set_deterministic(seed)
    samp_treated, _ = model.sample(do_interventions={X: x_treated}, number_of_samples=n_samples)
    _set_deterministic(seed + 1)
    samp_control, _ = model.sample(do_interventions={X: x_control}, number_of_samples=n_samples)
    y_treated = samp_treated[Y].cpu().numpy()
    y_control = samp_control[Y].cpu().numpy()

    artifacts: List[str] = []
    manifest: List[Dict[str, Any]] = []

    nodes = list(samp_treated.keys())
    fig_w = max(4.5 * len(nodes), 10.0)
    fig, axes = plt.subplots(1, len(nodes), figsize=(fig_w, 5.0))
    if len(nodes) == 1:
        axes = [axes]
    ate_value = float(np.mean(y_treated) - np.mean(y_control))
    for i, node in enumerate(nodes):
        ctrl_vals = samp_control[node].cpu().numpy()
        trt_vals = samp_treated[node].cpu().numpy()
        all_vals = np.concatenate([ctrl_vals, trt_vals])
        if np.max(all_vals) == np.min(all_vals):
            bins = np.linspace(np.min(all_vals) - 1.0, np.max(all_vals) + 1.0, 80)
        else:
            bins = np.linspace(np.min(all_vals), np.max(all_vals), 80)

        axes[i].hist(ctrl_vals, bins=bins, density=True, alpha=0.5, color="steelblue", label=f"do({X}={x_control})")
        axes[i].hist(trt_vals, bins=bins, density=True, alpha=0.5, color="darkorange", label=f"do({X}={x_treated})")

        if gaussian_kde is not None and len(np.unique(ctrl_vals)) > 1 and len(np.unique(trt_vals)) > 1:
            x_grid = np.linspace(np.min(all_vals), np.max(all_vals), 300)
            axes[i].plot(x_grid, gaussian_kde(ctrl_vals)(x_grid), color="steelblue", linestyle="--", lw=2)
            axes[i].plot(x_grid, gaussian_kde(trt_vals)(x_grid), color="darkorange", linestyle="--", lw=2)

        ctrl_mean = float(np.mean(ctrl_vals))
        trt_mean = float(np.mean(trt_vals))
        axes[i].axvline(ctrl_mean, color="steelblue", linestyle=":", lw=1.5, alpha=0.8)
        axes[i].axvline(trt_mean, color="darkorange", linestyle=":", lw=1.5, alpha=0.8)

        if node == Y:
            title = f"{node} (ATE={ate_value:.3f})"
        elif node == X:
            title = f"{node} (intervened)"
        else:
            title = f"{node} (shift={trt_mean - ctrl_mean:.3f})"
        axes[i].set_title(title)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.2)
    fig.tight_layout()
    inter_path = artifact_dir / "intervention_treated_vs_control.png"
    fig.savefig(inter_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(inter_path))
    manifest.append(_artifact_entry("intervention", inter_path))

    return {
        "data": {
            "X": X,
            "Y": Y,
            "x_treated": x_treated,
            "x_control": x_control,
            "ate": ate_value,
            "y_treated_mean": float(np.mean(y_treated)),
            "y_control_mean": float(np.mean(y_control)),
            "y_treated_std": float(np.std(y_treated)),
            "y_control_std": float(np.std(y_control)),
            "y_treated_values": y_treated.tolist(),
            "y_control_values": y_control.tolist(),
            "random_seed": seed,
            "artifact_manifest": manifest,
        },
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tool = request["tool"]
    payload = request.get("payload", {})
    artifact_dir = Path(request.get("artifact_dir", Path(args.output).parent / "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    response: Dict[str, Any] = {
        "success": False,
        "data": {},
        "artifacts": [],
        "tables": {},
        "messages": [],
        "error": None,
    }

    try:
        if tool == "fit_model":
            payload_out = _fit_model(payload, artifact_dir)
            response["data"] = payload_out["data"]
            response["artifacts"] = payload_out.get("artifacts", [])
            response["messages"] = ["TRAM-DAG model fitting completed with canonical plots."]
        elif tool == "sample":
            payload_out = _sample(payload, artifact_dir)
            response["data"] = payload_out["data"]
            response["artifacts"] = payload_out.get("artifacts", [])
            response["messages"] = ["Sampling completed with canonical plots."]
        elif tool == "compute_ate":
            payload_out = _compute_ate(payload, artifact_dir)
            response["data"] = payload_out["data"]
            response["artifacts"] = payload_out.get("artifacts", [])
            response["messages"] = ["ATE computation completed with canonical plots."]
        else:
            raise ValueError(f"Unsupported TRAM tool: {tool}")
        response["success"] = True
    except Exception as exc:
        response["success"] = False
        response["error"] = f"{type(exc).__name__}: {exc}"
        response["messages"] = [traceback.format_exc()]

    Path(args.output).write_text(json.dumps(_jsonify(response), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

