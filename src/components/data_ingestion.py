import sys
from dataclasses import dataclass
from pathlib import Path

from src.expection import CustomException
from src.logger import logger
from src.utils import env_flag, get_env, resolve_path

DEFAULT_BDD100K_DATA_ROOT = resolve_path(get_env("BDD100K_DATA_ROOT", "./data/bdd100k"))
DEFAULT_DOWNLOAD_DIR = resolve_path(get_env("ARTIFACTS_DIR", "artifacts")) / "data" / "kaggle_bdd100k"

KAGGLE_DATASET = get_env("KAGGLE_DATASET", "solesensei/solesensei_bdd100k")
LABELS_REL_PATH = Path("bdd100k_labels_release") / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
IMAGE_REL_CANDIDATES = [
    Path("bdd100k") / "bdd100k" / "images" / "100k" / "val",
    Path("bdd100k") / "images" / "100k" / "val",
]


@dataclass
class DataIngestionConfig:
    kaggle_dataset: str = KAGGLE_DATASET
    labels_rel_path: Path = LABELS_REL_PATH
    image_rel_candidates: tuple[Path, ...] = tuple(IMAGE_REL_CANDIDATES)
    artifact_dir: Path = DEFAULT_DOWNLOAD_DIR
    use_dummy: bool = False
    dummy_root: Path = DEFAULT_DUMMY_ROOT


@dataclass
class DataIngestionArtifact:
    source: str
    dataset_root: Path
    labels_path: Path
    image_root: Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        use_dummy = self.config.use_dummy or env_flag("USE_DUMMY_DATA")

        try:
            if use_dummy:
                logger.info("Loading dummy BDD100K dataset from local path.")
                artifact = self._load_dummy_dataset()
            else:
                logger.info("Downloading BDD100K dataset from Kaggle.")
                artifact = self._load_kaggle_dataset()

            self._validate_paths(artifact)
            logger.info("Data ingestion completed successfully.")
            logger.info("Dataset root: %s", artifact.dataset_root)
            logger.info("Labels path: %s", artifact.labels_path)
            logger.info("Image root: %s", artifact.image_root)
            return artifact

        except CustomException:
            raise
        except Exception as error:
            raise CustomException(error, sys) from error

    def _load_kaggle_dataset(self) -> DataIngestionArtifact:
        dataset_root = self._download_with_kagglehub()
        if dataset_root is None:
            dataset_root = self._download_with_kaggle_api()

        labels_path = dataset_root / self.config.labels_rel_path
        image_root = self._resolve_image_root(dataset_root)

        return DataIngestionArtifact(
            source="kaggle",
            dataset_root=dataset_root,
            labels_path=labels_path,
            image_root=image_root,
        )

    def _download_with_kagglehub(self) -> Path | None:
        try:
            import kagglehub
        except ImportError as error:
            logger.warning("kagglehub import failed: %s", error)
            return None

        try:
            download_path = kagglehub.dataset_download(self.config.kaggle_dataset)
            return Path(download_path)
        except Exception as error:
            logger.warning("kagglehub download failed, trying Kaggle API fallback: %s", error)
            return None

    def _download_with_kaggle_api(self) -> Path:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as error:
            raise CustomException(
                ImportError(
                    "Kaggle download failed. Reinstall compatible packages with:\n"
                    "  pip install kagglehub==1.0.2 kagglesdk==0.1.22 kaggle\n"
                    "Or set USE_DUMMY_DATA=true in .env"
                ),
                sys,
            ) from error

        download_dir = self.config.artifact_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading dataset via Kaggle API to %s", download_dir)
        api.dataset_download_files(
            self.config.kaggle_dataset,
            path=str(download_dir),
            unzip=True,
            quiet=False,
        )

        for item in download_dir.rglob("bdd100k_labels_images_val.json"):
            return item.parents[2]

        return download_dir

    def _load_dummy_dataset(self) -> DataIngestionArtifact:
        dummy_root = self.config.dummy_root
        labels_path = dummy_root / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
        image_root = dummy_root / "bdd100k" / "images" / "100k" / "val"

        return DataIngestionArtifact(
            source="dummy",
            dataset_root=dummy_root,
            labels_path=labels_path,
            image_root=image_root,
        )

    def _resolve_image_root(self, dataset_root: Path) -> Path:
        for rel_path in self.config.image_rel_candidates:
            candidate = dataset_root / rel_path
            if candidate.exists():
                return candidate

        raise CustomException(
            ValueError(
                "Could not locate BDD100K validation images directory under "
                f"{dataset_root}. Tried: {list(self.config.image_rel_candidates)}"
            ),
            sys,
        )

    def _validate_paths(self, artifact: DataIngestionArtifact) -> None:
        if not artifact.labels_path.is_file():
            raise CustomException(
                FileNotFoundError(f"Labels file not found: {artifact.labels_path}"),
                sys,
            )

        if not artifact.image_root.is_dir():
            raise CustomException(
                FileNotFoundError(f"Image directory not found: {artifact.image_root}"),
                sys,
            )


if __name__ == "__main__":
    ingestion = DataIngestion()
    artifact = ingestion.initiate_data_ingestion()
    print(artifact)
