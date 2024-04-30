import dataclasses
import random
from pathlib import Path
from typing import List

from spacy import Language
from spacy.training.example import Example
from spacy.util import compounding, minibatch
from tqdm import tqdm
from wasabi import msg


### DEFINE VARIABLES USED IN SKILLNER_FLOW ###
@dataclasses.dataclass
class TrainConfig:
    random_seed: int = 42
    test_size: float = 0.25
    print_losses: bool = True
    drop_out: float = 0.1
    num_its: int = 30
    learn_rate: float = 0.001


@dataclasses.dataclass
class DataConfig:
    bucket_name: str = "open-jobs-lake"
    data_path: Path = Path(
        "escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json"
    )
    all_labels: List[str] = dataclasses.field(
        default_factory=lambda: ["SKILL", "MULTISKILL", "EXPERIENCE", "BENEFIT"]
    )


@dataclasses.dataclass
class Config:
    model_name: str = "nestauk/skillner"
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()


config: Config = Config()


def train_ner(nlp: Language, train_data: List[tuple]) -> None:
    """Train the Named Entity Recognition component
        of the spaCy pipeline.

    Args:
        nlp (Language): spaCy language model.
        train_data (List[tuple]): List of training data.
    """
    # List of pipes you want to train
    pipe_exceptions = ["ner"]
    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    optimizer = nlp.begin_training()

    optimizer.learn_rate = config.train.learn_rate

    # Begin training by disabling other pipeline components
    all_losses = []
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for num_its iterations
        for itn in tqdm(range(config.train.num_its)):
            # shuffle examples before training
            random.seed(itn)
            random.shuffle(train_data)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=sizes)
            # Dictionary to store losses
            losses = {}
            for batch in batches:
                # Calling update() over the iteration
                for text, annotation, _ in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    # Update the model
                    nlp.update(
                        [example],
                        sgd=optimizer,
                        losses=losses,
                        drop=config.train.drop_out,
                    )
            all_losses.append(losses["ner"])
            if config.train.print_losses:
                msg.info(losses)
