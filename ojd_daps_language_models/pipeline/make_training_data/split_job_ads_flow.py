"""
Clean and split job ads into to sentences in order to label to fine-tune a BERT-like model for job advertisements

python ojd_daps_language_models/pipeline/make_training_data/split_job_ads.py run --chunk_size 1000

"""
from metaflow import FlowSpec, step, Parameter
from ojd_daps_language_models import logger, get_yaml_config, PROJECT_DIR
import os

CONFIG = get_yaml_config(PROJECT_DIR / "ojd_daps_language_models/config/training.yaml")


class CleanJobsFlow(FlowSpec):
    chunk_size = Parameter(
        "chunk_size",
        help="# of job adverts to include per chunk",
        default=CONFIG["chunk_size"],
    )
    production = Parameter(
        "production", help="to run in production mode", default=False
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_job_ads)

    @step
    def get_job_ads(self):
        """Loads job ads from S3"""
        from nesta_ds_utils.loading_saving.S3 import download_obj

        logger.info("loading raw job ads")

        self.raw_job_ads = (
            download_obj(
                "open-jobs-lake",
                os.path.join(CONFIG["job_desc_path"], "raw_job_adverts_sample.csv"),
                download_as="dataframe",
            )
            .set_axis(
                ["id", "date", "job_title", "job_description"], axis=1, inplace=False
            )
            .drop_duplicates("job_description")
            .job_description.to_list()
        )

        self.raw_job_ads if self.production else self.raw_job_ads[:10]

        self.next(self.chunk_job_ads)

    @step
    def chunk_job_ads(self):
        """Chunk job ad texts"""
        from toolz import partition_all

        self.job_ad_chunks = list(partition_all(self.chunk_size, self.raw_job_ads))

        self.next(self.clean_job_ads, foreach="job_ad_chunks")

    @step
    def clean_job_ads(self):
        """Cleans job ads by:
        - removing numbers and stop words
        - Creating new sentences by Camel Case
        - splitting sentences

        and saves sentence chunks as .csv files
        """
        import uuid

        from nesta_ds_utils.loading_saving.S3 import upload_obj
        from ojd_daps_language_models.utils.text_cleaning import clean_text

        logger.info("cleaning job ads")

        cleaned_job_ads = [clean_text(job_ad) for job_ad in self.input]
        sent_chunk_filename = (
            f"cleaned_job_ad_sentences_{str(uuid.uuid4()).replace('-', '_')}.json"
        )

        if self.production:
            upload_obj(
                bucket="dap-ojobert",
                path_to=os.path.join(CONFIG["job_sent_path"], sent_chunk_filename),
                obj=cleaned_job_ads,
            )
        else:
            pass

        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        """Dummy join step"""
        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    CleanJobsFlow()
