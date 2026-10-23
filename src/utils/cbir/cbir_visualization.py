"""Plot catalogue ground truth, triplets, and model-predicted CBIR rankings."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from torchvision import transforms

from utils.image_preprocessing import pad_img, transform_fn
from utils.catalogue_utils import Catalogue, catalogue_type_to_id, parse_catalogue_folder, parse_filename
from utils.cbir.cbir_evaluation import comp_pairwise_distance


def visualize_catalogue(catalogue: Catalogue) -> None:
    """Display a catalogue query and its neighbors with filename metadata."""
    neighbors = catalogue.neighbors
    num_neighbors = len(neighbors)
    if num_neighbors == 0:
        return

    neighbor_rows = 2
    neighbor_cols = max(1, (num_neighbors + 1) // neighbor_rows)
    fig = plt.figure(figsize=(2.5 + 2 * neighbor_cols, 4.5))
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.5, neighbor_cols], wspace=0.02)
    gs_left = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.1, 1, 0.15], hspace=0.02
    )

    title_ax = fig.add_subplot(gs_left[0, 0])
    title_ax.text(
        0.5,
        0.5,
        f"Catalogue\n{catalogue.catalogue_type}_{catalogue.catalogue_id}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    title_ax.axis("off")

    query_ax = fig.add_subplot(gs_left[1, 0])
    query_ax.imshow(transforms.Resize((224, 224))(pad_img(catalogue.query)))
    query_ax.set_title("Query", fontsize=10, pad=3)
    query_metadata = parse_filename(os.path.basename(catalogue.query))
    if query_metadata:
        query_legend = "\n".join(
            f'{query_metadata[key]}' for key in ("age", "height", "weight", "bra_cup", "bra_size")
        )
        query_ax.text(
            0.05,
            0.95,
            query_legend,
            transform=query_ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    query_ax.axis("off")

    subtitle_ax = fig.add_subplot(gs_left[2, 0])
    subtitle_ax.text(
        0.5,
        0.5,
        "Legend:\nage (years)\nheight (cm)\nweight (kg)\nbra cup\nbra size",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    subtitle_ax.axis("off")

    gs_right = GridSpecFromSubplotSpec(
        neighbor_rows, neighbor_cols, subplot_spec=gs_main[0, 1], wspace=0.0, hspace=0.0
    )
    for idx, neighbor_path in enumerate(neighbors):
        row = idx // neighbor_cols
        col = idx % neighbor_cols
        if row >= neighbor_rows:
            break

        ax = fig.add_subplot(gs_right[row, col])
        ax.imshow(transforms.Resize((224, 224))(pad_img(neighbor_path)))
        neighbor_metadata = parse_filename(os.path.basename(neighbor_path))
        if neighbor_metadata:
            ax.set_title(f"Rank {neighbor_metadata['rank']}", fontsize=8, pad=3)
            neighbor_legend = "\n".join(
                f'{neighbor_metadata[key]}' for key in ("age", "height", "weight", "bra_cup", "bra_size")
            )
            ax.text(
                0.05,
                0.95,
                neighbor_legend,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
        ax.axis("off")

    plt.show()


def visualize_triplet(triplet: Tuple[str, str, str]) -> None:
    """Visualize a query, positive, and negative image in a row."""
    query_path, pos_path, neg_path = triplet
    query_img = transforms.Resize((224, 224))(pad_img(query_path))
    pos_img = transforms.Resize((224, 224))(pad_img(pos_path))
    neg_img = transforms.Resize((224, 224))(pad_img(neg_path))
    catalogue_folder = os.path.basename(os.path.dirname(query_path))
    catalogue_type, catalogue_id = parse_catalogue_folder(catalogue_folder)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.0, 2.6),
        gridspec_kw={"wspace": 0.0005, "width_ratios": [1.0, 1.0, 1.0, 0.45]},
    )
    fig.suptitle(f"Catalogue {catalogue_type}_{catalogue_id}", fontsize=12, fontweight="bold", y=0.85)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.62, bottom=0.0)
    for ax, image, title, color in zip(
        axes[:3], (query_img, pos_img, neg_img), ("Query", "Positive", "Negative"), (None, "green", "red")
    ):
        ax.imshow(image)
        ax.set_title(title)
        if color is not None:
            ax.title.set_color(color)
        ax.axis("off")

    for ax, img_path in zip(axes[:3], (query_path, pos_path, neg_path)):
        metadata = parse_filename(os.path.basename(img_path))
        if metadata:
            legend_text = "\n".join(
                f'{metadata[key]}' for key in ("age", "height", "weight", "bra_cup", "bra_size")
            )
            ax.text(
                0.05,
                0.95,
                legend_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    axes[3].text(
        0.075,
        0.5,
        "Legend:\nage (years)\nheight (cm)\nweight (kg)\nbra cup\nbra size",
        transform=axes[3].transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
    )
    axes[3].axis("off")
    plt.show()


def visualize_predicted_ranking(
    model: torch.nn.Module,
    catalogue: Catalogue,
    device: str = "cpu",
    top_k: Optional[int] = None,
    distance_metric: str = "euclidean",
) -> None:
    """Visualize the model's predicted ranking against ground truth."""
    query_path = catalogue.query
    neighbor_paths = catalogue.neighbors
    if len(neighbor_paths) == 0:
        raise ValueError("No neighbor images were provided for ranking visualization.")

    display_neighbor_paths = neighbor_paths[:top_k] if top_k is not None else list(neighbor_paths)
    gt_rank_by_path = {path: idx + 1 for idx, path in enumerate(neighbor_paths)}
    model.eval()
    with torch.no_grad():
        q_tensor = transform_fn(query_path).unsqueeze(0).to(device)
        cat_type_id_value = catalogue_type_to_id(catalogue.catalogue_type)
        q_type_id = torch.full((1,), fill_value=cat_type_id_value, dtype=torch.long, device=device)
        try:
            q_emb = F.normalize(model(q_tensor, q_type_id), p=2, dim=1)
        except TypeError:
            q_emb = F.normalize(model(q_tensor), p=2, dim=1)

        n_tensors = torch.stack([transform_fn(path) for path in display_neighbor_paths]).to(device)
        n_type_id = torch.full((n_tensors.size(0),), fill_value=cat_type_id_value, dtype=torch.long, device=device)
        try:
            n_embs = F.normalize(model(n_tensors, n_type_id), p=2, dim=1)
        except TypeError:
            n_embs = F.normalize(model(n_tensors), p=2, dim=1)
        distances = comp_pairwise_distance(q_emb.expand_as(n_embs), n_embs, distance_metric=distance_metric)
        distances = distances.detach().cpu().numpy()

    ranked_neighbors = sorted(zip(display_neighbor_paths, distances.tolist()), key=lambda item: item[1])
    num_neighbors = len(ranked_neighbors)
    neighbor_rows = 2
    neighbor_cols = max(1, (num_neighbors + 1) // neighbor_rows)
    fig = plt.figure(figsize=(2.5 + 2 * neighbor_cols, 4.5))
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.5, neighbor_cols], wspace=0.02)
    gs_left = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.1, 1, 0.15], hspace=0.02
    )
    title_ax = fig.add_subplot(gs_left[0, 0])
    title_ax.text(
        0.5,
        0.5,
        f"Catalogue\n{catalogue.catalogue_type}_{catalogue.catalogue_id}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    title_ax.axis("off")

    query_ax = fig.add_subplot(gs_left[1, 0])
    query_ax.imshow(transforms.Resize((224, 224))(pad_img(query_path)))
    query_ax.set_title("Query", fontsize=10, pad=3)
    query_ax.axis("off")
    query_metadata = parse_filename(os.path.basename(query_path))
    if query_metadata:
        query_legend = "\n".join(
            f'{query_metadata[key]}' for key in ("age", "height", "weight", "bra_cup", "bra_size")
        )
        query_ax.text(
            0.05,
            0.95,
            query_legend,
            transform=query_ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    subtitle_ax = fig.add_subplot(gs_left[2, 0])
    subtitle_ax.text(
        0.5,
        0.5,
        "Legend:\nage (years)\nheight (cm)\nweight (kg)\nbra cup\nbra size",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    subtitle_ax.axis("off")

    gs_right = GridSpecFromSubplotSpec(
        neighbor_rows, neighbor_cols, subplot_spec=gs_main[0, 1], wspace=0.0, hspace=0.0
    )
    for idx, (neighbor_path, _dist) in enumerate(ranked_neighbors):
        row = idx // neighbor_cols
        col = idx % neighbor_cols
        if row >= neighbor_rows:
            break

        pred_rank = idx + 1
        gt_rank = gt_rank_by_path.get(neighbor_path)
        color = "green" if gt_rank == pred_rank else "red"
        ax = fig.add_subplot(gs_right[row, col])
        ax.imshow(transforms.Resize((224, 224))(pad_img(neighbor_path)))
        ax.set_title(f"Pred {pred_rank} | GT {gt_rank}", fontsize=8, pad=3, color=color)
        ax.axis("off")
        metadata = parse_filename(os.path.basename(neighbor_path))
        if metadata:
            legend_text = "\n".join(
                f'{metadata[key]}' for key in ("age", "height", "weight", "bra_cup", "bra_size")
            )
            ax.text(
                0.05,
                0.95,
                legend_text,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    plt.show()

