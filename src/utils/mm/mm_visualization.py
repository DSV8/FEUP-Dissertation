"""Visualize multimodal catalogue features and predicted neighbour order."""

import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from utils.image_preprocessing import pad_img, transform_fn
from utils.catalogue_utils import (
    Catalogue,
    parse_catalogue_folder,
    parse_filename,
)
from models.multimodal import MultimodalEncoder
from utils.mm.mm_data import MultimodalTriplet, normalize_tabular_if_needed, tabular_vec_from_img_path
from utils.mm.mm_evaluation import comp_pairwise_distance

def visualize_catalogue_multimodal(catalogue: Catalogue, tabular_stats: Optional[Dict[str, np.ndarray]] = None) -> None:
	"""
	Display multimodal catalogue in a grid layout:
	- Left column: Catalogue title (top), Query image (middle), Legend (bottom)
	- Right columns: Two rows of neighbor images with legends inside
	"""
	neighbors = catalogue.neighbors
	num_neighbors = len(neighbors)

	if num_neighbors == 0:
		raise ValueError("No neighbor images were provided for visualization.")

	neighbor_rows = 2
	neighbor_cols = max(1, (num_neighbors + 1) // neighbor_rows)

	fig = plt.figure(figsize=(2.5 + 2 * neighbor_cols, 4.5))
	gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.5, neighbor_cols], wspace=0.02)

	gs_left = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.1, 1, 0.15], hspace=0.02)

	title_ax = fig.add_subplot(gs_left[0, 0])
	title_ax.text(0.5, 0.5, f'Catalogue\n{catalogue.catalogue_type}_{catalogue.catalogue_id}', ha='center', va='center', fontsize=10, fontweight='bold')
	title_ax.axis('off')

	query_img = transforms.Resize((224, 224))(pad_img(catalogue.query))
	query_ax = fig.add_subplot(gs_left[1, 0])
	query_ax.imshow(query_img)
	query_ax.set_title('Query', fontsize=10, pad=3)

	query_metadata = parse_filename(os.path.basename(catalogue.query))
	if query_metadata:
		query_legend = "\n".join([
			f'{query_metadata["age"]}',
			f'{query_metadata["height"]}',
			f'{query_metadata["weight"]}',
			f'{query_metadata["bra_cup"]}',
			f'{query_metadata["bra_size"]}',
		])
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
	subtitle_ax.text(0.5, 0.5, 'Legend:\nage (years)\nheight (cm)\nweight (kg)\nbra cup\nbra size', ha='center', va='center', fontsize=8, style='italic')
	subtitle_ax.axis('off')

	gs_right = GridSpecFromSubplotSpec(neighbor_rows, neighbor_cols, subplot_spec=gs_main[0, 1], wspace=0.0, hspace=0.0)

	for idx, neighbor_path in enumerate(neighbors):
		row = idx // neighbor_cols
		col = idx % neighbor_cols
		if row >= neighbor_rows:
			break

		gt_rank = idx + 1
		neighbor_img = transforms.Resize((224, 224))(pad_img(neighbor_path))

		ax = fig.add_subplot(gs_right[row, col])
		ax.imshow(neighbor_img)
		ax.set_title(f'Rank {gt_rank}', fontsize=8, pad=3)
		ax.axis('off')

		metadata = parse_filename(os.path.basename(neighbor_path))
		if metadata:
			legend_text = "\n".join([
				f'{metadata["age"]}',
				f'{metadata["height"]}',
				f'{metadata["weight"]}',
				f'{metadata["bra_cup"]}',
				f'{metadata["bra_size"]}',
			])
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

def visualize_triplet_multimodal(
	triplet: MultimodalTriplet,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
) -> None:
	"""
	Visualize a multimodal triplet (query, pos, neg) with images and tabular features.
	"""
	query_img = transforms.Resize((224, 224))(pad_img(triplet.query.image_path))
	pos_img = transforms.Resize((224, 224))(pad_img(triplet.pos.image_path))
	neg_img = transforms.Resize((224, 224))(pad_img(triplet.neg.image_path))

	catalogue_folder = os.path.basename(os.path.dirname(triplet.query.image_path))
	catalogue_type, catalogue_id = parse_catalogue_folder(catalogue_folder)

	fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.0005, "width_ratios": [1.0, 1.0, 1.0, 0.45]})

	fig.suptitle(f"Catalogue {catalogue_type}_{catalogue_id}", fontsize=12, fontweight="bold", y=0.85)
	fig.subplots_adjust(left=0.0, right=1.0, top=0.62, bottom=0.0)

	axes[0].imshow(query_img)
	axes[0].set_title('Query')
	axes[0].axis('off')

	axes[1].imshow(pos_img)
	axes[1].set_title('Positive')
	axes[1].title.set_color('green')
	axes[1].axis('off')

	axes[2].imshow(neg_img)
	axes[2].set_title('Negative')
	axes[2].title.set_color('red')
	axes[2].axis('off')

	# Add tabular feature legends
	for ax, sample in zip(axes, [triplet.query, triplet.pos, triplet.neg]):
		legend_text = "\n".join([f"{v:.2f}" for v in sample.tabular])
		ax.text(
			0.05,
			0.95,
			legend_text,
			transform=ax.transAxes,
			fontsize=7,
			verticalalignment="top",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
		)

	# Add info legend
	axes[3].text(0.075, 0.5, 'Legend:\nage (years)\nheight (cm)\nweight (kg)\nbra cup\nbra size', transform=axes[3].transAxes, fontsize=9, verticalalignment='center', horizontalalignment='left')
	axes[3].axis('off')

	plt.show()

def visualize_predicted_ranking_multimodal(
	model: MultimodalEncoder,
	catalogue: Catalogue,
	device: str = "cpu",
	top_k: Optional[int] = None,
	distance_metric: str = "cosine",
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	image_transform: Optional[Callable[[str], torch.Tensor]] = None,
) -> None:
	"""
	Visualize multimodal predicted ranking against ground truth.

	The images are displayed in predicted order. Neighbor title is green if
	predicted rank equals ground-truth rank, red otherwise.
	"""
	query_path = catalogue.query
	neighbor_paths = catalogue.neighbors

	if len(neighbor_paths) == 0:
		raise ValueError("No neighbor images were provided for ranking visualization.")

	transform = image_transform if image_transform is not None else transform_fn

	q_tab = tabular_vec_from_img_path(query_path)
	if q_tab is None:
		raise ValueError(f"Unable to parse query tabular features for {query_path}")
	q_tab = normalize_tabular_if_needed(q_tab, tabular_stats)

	display_neighbor_paths = neighbor_paths[:top_k] if top_k is not None else list(neighbor_paths)
	gt_rank_by_path = {path: idx + 1 for idx, path in enumerate(neighbor_paths)}

	n_tabs: List[np.ndarray] = []
	valid_neighbor_paths: List[str] = []
	for path in display_neighbor_paths:
		tab = tabular_vec_from_img_path(path)
		if tab is None:
			continue
		n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
		valid_neighbor_paths.append(path)

	if len(valid_neighbor_paths) == 0:
		raise ValueError("No valid neighbors with tabular features were found.")

	model.eval()
	with torch.no_grad():
		q_img = transform(query_path).unsqueeze(0).to(device)
		q_tab = torch.tensor(q_tab, dtype=torch.float32).unsqueeze(0).to(device)
		q_emb = model(q_img, q_tab)

		n_imgs = torch.stack([transform(path) for path in valid_neighbor_paths]).to(device)
		n_tabs = torch.tensor(np.stack(n_tabs, axis=0), dtype=torch.float32).to(device)
		n_embs = model(n_imgs, n_tabs)
		q_emb_expanded = q_emb.expand_as(n_embs)
		distances = comp_pairwise_distance(q_emb_expanded, n_embs, distance_metric=distance_metric).detach().cpu().numpy()

	ranked_neighbors = sorted(
		zip(valid_neighbor_paths, distances.tolist()),
		key=lambda item: item[1],
	)

	num_neighbors = len(ranked_neighbors)
	neighbor_rows = 2
	neighbor_cols = max(1, (num_neighbors + 1) // neighbor_rows)

	fig = plt.figure(figsize=(2.5 + 2 * neighbor_cols, 4.5))
	gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.5, neighbor_cols], wspace=0.02)

	gs_left = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.1, 1, 0.15], hspace=0.02)

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
	q_img = transforms.Resize((224, 224))(pad_img(query_path))
	query_ax.imshow(q_img)
	query_ax.set_title("Query", fontsize=10, pad=3)
	query_ax.axis("off")

	query_metadata = parse_filename(os.path.basename(query_path))
	if query_metadata:
		query_legend = "\n".join([
			f'{query_metadata["age"]}',
			f'{query_metadata["height"]}',
			f'{query_metadata["weight"]}',
			f'{query_metadata["bra_cup"]}',
			f'{query_metadata["bra_size"]}',
		])
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

	gs_right = GridSpecFromSubplotSpec(neighbor_rows, neighbor_cols, subplot_spec=gs_main[0, 1], wspace=0.0, hspace=0.0)

	for idx, (neighbor_path, _dist) in enumerate(ranked_neighbors):
		row = idx // neighbor_cols
		col = idx % neighbor_cols
		if row >= neighbor_rows:
			break

		pred_rank = idx + 1
		gt_rank = gt_rank_by_path.get(neighbor_path)
		is_match = gt_rank == pred_rank
		color = "green" if is_match else "red"

		ax = fig.add_subplot(gs_right[row, col])
		neighbor_img = transforms.Resize((224, 224))(pad_img(neighbor_path))
		ax.imshow(neighbor_img)
		ax.set_title(f"Pred {pred_rank} | GT {gt_rank}", fontsize=8, pad=3, color=color)
		ax.axis("off")

		metadata = parse_filename(os.path.basename(neighbor_path))
		if metadata:
			legend_text = "\n".join([
				f'{metadata["age"]}',
				f'{metadata["height"]}',
				f'{metadata["weight"]}',
				f'{metadata["bra_cup"]}',
				f'{metadata["bra_size"]}',
			])
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
