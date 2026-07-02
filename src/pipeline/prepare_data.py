"""
Prepare BDD100K VQA dataset: ingest from Kaggle/dummy, then transform.
Run: python -m src.pipeline.prepare_data
"""

import sys
from pathlib import Path

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.expection import CustomException
from src.logger import logger
from src.utils import env_flag, get_env, resolve_path


class PrepareDataPipeline:
    def __init__(
        self,
        ingestion_config: DataIngestionConfig | None = None,
        transformation_config: DataTransformationConfig | None = None,
        output_path: Path | None = None,
    ):
        self.ingestion_config = ingestion_config or DataIngestionConfig(
            use_dummy=env_flag("USE_DUMMY_DATA")
        )
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.output_path = Path(
            output_path
            or resolve_path(get_env("PROCESSED_VQA_PATH", "data/processed/bdd100k_risk_vqa.json"))
        )

    def run(self) -> Path:
        try:
            logger.info("Step 1/2: Data ingestion")
            ingestion = DataIngestion(self.ingestion_config)
            ingest_artifact = ingestion.initiate_data_ingestion()

            logger.info("Step 2/2: Data transformation")
            transformation = DataTransformation(self.transformation_config)
            transform_artifact = transformation.initiate_data_transformation(
                bdd_json_path=ingest_artifact.labels_path,
                image_root=ingest_artifact.image_root,
                output_file=self.output_path,
            )

            logger.info("Pipeline complete: %s samples -> %s", transform_artifact.num_samples, transform_artifact.output_path)
            return transform_artifact.output_path

        except CustomException:
            raise
        except Exception as error:
            raise CustomException(error, sys) from error


def prepare_data(output_path: Path | None = None) -> Path:
    return PrepareDataPipeline(output_path=output_path).run()


if __name__ == "__main__":
    path = prepare_data()
    print(f"Prepared dataset: {path}")
