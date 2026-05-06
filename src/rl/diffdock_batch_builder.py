from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from src.rl.diffdock_loss import _diffdock_import_path
from src.rl.types import RLExample


@dataclass(frozen=True)
class DiffDockBatchBuilderConfig:
    repo_root: str | Path = "external/DiffDock"
    work_dir: str | Path = "artifacts/tmp/diffdock_rl_batches"
    lm_embeddings: bool = True
    receptor_radius: float = 30.0
    c_alpha_max_neighbors: int | None = None
    remove_hs: bool = False
    all_atoms: bool = False
    atom_radius: float = 5.0
    atom_max_neighbors: int | None = None
    knn_only_graph: bool = False
    no_torsion: bool = False
    noise_alpha: float = 1.0
    noise_beta: float = 1.0
    minimum_t: float = 0.0
    crop_beyond_cutoff: float | None = None
    include_miscellaneous_atoms: bool = False


def config_from_score_model_args(
    score_model_args: Any,
    *,
    repo_root: str | Path = "external/DiffDock",
    work_dir: str | Path = "artifacts/tmp/diffdock_rl_batches",
    lm_embeddings: bool = True,
) -> DiffDockBatchBuilderConfig:
    return DiffDockBatchBuilderConfig(
        repo_root=repo_root,
        work_dir=work_dir,
        lm_embeddings=lm_embeddings,
        receptor_radius=getattr(score_model_args, "receptor_radius", 30.0),
        c_alpha_max_neighbors=getattr(score_model_args, "c_alpha_max_neighbors", None),
        remove_hs=getattr(score_model_args, "remove_hs", False),
        all_atoms=getattr(score_model_args, "all_atoms", False),
        atom_radius=getattr(score_model_args, "atom_radius", 5.0),
        atom_max_neighbors=getattr(score_model_args, "atom_max_neighbors", None),
        knn_only_graph=(
            False
            if not hasattr(score_model_args, "not_knn_only_graph")
            else not score_model_args.not_knn_only_graph
        ),
        no_torsion=getattr(score_model_args, "no_torsion", False),
        crop_beyond_cutoff=getattr(score_model_args, "crop_beyond", None),
    )


class DiffDockGeneratedPoseBatchBuilder:
    """
    Build DiffDock training-loss batches from final generated poses.

    This adapter reuses DiffDock's inference graph builder for receptor and ligand
    topology, overwrites ligand coordinates with each generated SDF pose, then
    applies DiffDock's native `NoiseTransform` to attach score-matching labels
    needed by `utils.training.loss_function`.

    The resulting surrogate is a local score-matching proxy around each generated
    pose. It is the first in-process bridge needed before real GRPO model updates.
    """

    def __init__(
        self,
        *,
        config: DiffDockBatchBuilderConfig,
        t_to_sigma: Callable,
        device: Any,
        inference_dataset_cls: type | None = None,
        noise_transform_cls: type | None = None,
        data_loader_cls: type | None = None,
        read_molecule_fn: Callable | None = None,
        remove_hs_fn: Callable | None = None,
        tensor_fn: Callable | None = None,
    ) -> None:
        self.config = config
        self.t_to_sigma = t_to_sigma
        self.device = device
        self.inference_dataset_cls = inference_dataset_cls
        self.noise_transform_cls = noise_transform_cls
        self.data_loader_cls = data_loader_cls
        self.read_molecule_fn = read_molecule_fn
        self.remove_hs_fn = remove_hs_fn
        self.tensor_fn = tensor_fn

    @classmethod
    def from_score_model_args(
        cls,
        *,
        repo_root: str | Path,
        score_model_args: Any,
        device: Any,
        work_dir: str | Path = "artifacts/tmp/diffdock_rl_batches",
        lm_embeddings: bool = True,
    ) -> "DiffDockGeneratedPoseBatchBuilder":
        with _diffdock_import_path(repo_root):
            from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

        return cls(
            config=config_from_score_model_args(
                score_model_args,
                repo_root=repo_root,
                work_dir=work_dir,
                lm_embeddings=lm_embeddings,
            ),
            t_to_sigma=partial(t_to_sigma_compl, args=score_model_args),
            device=device,
        )

    def __call__(self, examples: Sequence[RLExample]) -> Any:
        if not examples:
            raise ValueError("Cannot build a DiffDock batch with no examples")

        symbols = self._load_diffdock_symbols()
        dataset = self._build_inference_dataset(examples, symbols["InferenceDataset"])
        noise_transform = self._build_noise_transform(symbols["NoiseTransform"])

        data_list = []
        for index, example in enumerate(examples):
            graph = dataset[index]
            success = bool(graph["success"]) if "success" in graph else True
            if not success:
                raise ValueError(
                    f"DiffDock graph construction failed for {example.complex_id} "
                    f"sample {example.sample_id}"
                )

            self._replace_ligand_pose(
                graph,
                example,
                read_molecule=symbols["read_molecule"],
                remove_all_hs=symbols["RemoveAllHs"],
                torch_module=symbols["torch"],
            )
            data_list.append(noise_transform(graph))

        if getattr(self.device, "type", str(self.device)) == "cuda":
            return [graph.to(self.device) for graph in data_list]

        loader = symbols["DataLoader"](dataset=data_list, batch_size=len(data_list))
        return next(iter(loader))

    def _load_diffdock_symbols(self) -> dict[str, Any]:
        if (
            self.inference_dataset_cls is not None
            and self.noise_transform_cls is not None
            and self.data_loader_cls is not None
            and self.read_molecule_fn is not None
            and self.remove_hs_fn is not None
            and self.tensor_fn is not None
        ):
            return {
                "InferenceDataset": self.inference_dataset_cls,
                "NoiseTransform": self.noise_transform_cls,
                "DataLoader": self.data_loader_cls,
                "read_molecule": self.read_molecule_fn,
                "RemoveAllHs": self.remove_hs_fn,
                "torch": self.tensor_fn,
            }

        repo_root = self.config.repo_root

        with _diffdock_import_path(repo_root):
            from datasets.pdbbind import NoiseTransform
            from datasets.process_mols import read_molecule
            from rdkit.Chem import RemoveAllHs
            import torch
            from torch_geometric.loader import DataLoader
            from utils.inference_utils import InferenceDataset

        return {
            "InferenceDataset": self.inference_dataset_cls or InferenceDataset,
            "NoiseTransform": self.noise_transform_cls or NoiseTransform,
            "DataLoader": self.data_loader_cls or DataLoader,
            "read_molecule": self.read_molecule_fn or read_molecule,
            "RemoveAllHs": self.remove_hs_fn or RemoveAllHs,
            "torch": self.tensor_fn or torch,
        }

    def _build_inference_dataset(
        self,
        examples: Sequence[RLExample],
        inference_dataset_cls: type,
    ) -> Any:
        config = self.config
        work_dir = Path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        return inference_dataset_cls(
            out_dir=str(work_dir),
            complex_names=[
                f"{example.complex_id}_sample_{example.sample_id}"
                for example in examples
            ],
            protein_files=[example.protein_path for example in examples],
            ligand_descriptions=[
                example.predicted_pose_path
                for example in examples
            ],
            protein_sequences=[None for _ in examples],
            lm_embeddings=config.lm_embeddings,
            receptor_radius=config.receptor_radius,
            remove_hs=config.remove_hs,
            c_alpha_max_neighbors=config.c_alpha_max_neighbors,
            all_atoms=config.all_atoms,
            atom_radius=config.atom_radius,
            atom_max_neighbors=config.atom_max_neighbors,
            knn_only_graph=config.knn_only_graph,
        )

    def _build_noise_transform(self, noise_transform_cls: type) -> Any:
        config = self.config
        return noise_transform_cls(
            self.t_to_sigma,
            config.no_torsion,
            config.all_atoms,
            alpha=config.noise_alpha,
            beta=config.noise_beta,
            include_miscellaneous_atoms=config.include_miscellaneous_atoms,
            crop_beyond_cutoff=config.crop_beyond_cutoff,
            minimum_t=config.minimum_t,
        )

    def _replace_ligand_pose(
        self,
        graph: Any,
        example: RLExample,
        *,
        read_molecule: Callable,
        remove_all_hs: Callable,
        torch_module: Any,
    ) -> None:
        mol = read_molecule(
            example.predicted_pose_path,
            remove_hs=False,
            sanitize=True,
        )
        if mol is None:
            raise ValueError(f"Could not read generated pose: {example.predicted_pose_path}")

        if self.config.remove_hs:
            mol = remove_all_hs(mol)

        conformer = mol.GetConformer()
        pose = torch_module.tensor(
            conformer.GetPositions(),
            dtype=torch_module.float32,
        )
        current_pos = graph["ligand"].pos
        if pose.shape[0] != current_pos.shape[0]:
            raise ValueError(
                "Generated pose atom count does not match DiffDock ligand graph "
                f"for {example.complex_id} sample {example.sample_id}: "
                f"{pose.shape[0]} != {current_pos.shape[0]}"
            )

        graph["ligand"].pos = pose - graph.original_center
