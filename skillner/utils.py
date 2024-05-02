"""
Utils associated to the multiskill and skill classification flows.

It also contains utils associated to fixing entity spans and cleaning texts
based on labelling data with label-studio then subsequently with Prodigy.
"""

import dataclasses
import difflib
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

from nervaluate import Evaluator
from spacy import Language
from spacy.training import Example
from spacy.util import compounding, fix_random_seed, minibatch
from toolz import pipe
from tqdm import tqdm
from wasabi import msg


### DEFINE VARIABLES USED ACROSS THE FLOWS ###
@dataclasses.dataclass
class TrainConfig:
    random_seed: int = 42
    test_size: float = 0.25
    kernel: str = "linear"
    class_weight: str = "balanced"
    train_prop: float = 0.8
    drop_out: float = 0.3
    num_its: int = 50
    learn_rate: float = 0.001
    print_losses: bool = True
    spacy_model: str = "en_core_web_lg"


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
class HfConfig:
    namespace: str = "nestauk"
    ms_model_name: str = "multiskill-classifier"
    sn_model_name: str = "skillner"


@dataclasses.dataclass
class Config:
    train: TrainConfig = TrainConfig()
    hf: HfConfig = HfConfig()
    data: DataConfig = DataConfig()


config: Config = Config()

### DEFINE VARIABLES USED IN TEXT CLEANING ###

compiled_missing_space_pattern = re.compile("([a-z])([A-Z])")
compiled_nonalphabet_nonnumeric_pattern = re.compile(r"([^a-zA-Z0-9] )")
exception_camelcases = [
    "JavaScript",
    "WordPress",
    "PowerPoint",
    "CloudFormation",
    "CommVault",
    "InDesign",
    "GitHub",
    "GitLab",
    "DevOps",
    "QuickBooks",
    "TypeScript",
    "XenDesktop",
    "DevSecOps",
    "CircleCi",
    "LeDeR",
    "CeMap",
    "MavenAutomation",
    "SaaS",
    "iOS",
    "MySQL",
    "MongoDB",
    "NoSQL",
    "GraphQL",
    "VoIP",
    "PhD",
    "HyperV",
    "PaaS",
    "ArgoCD",
    "WinCC",
    "AutoCAD",
]
trim_chars = [" ", ".", ",", ";", ":", "\xa0"]

### FUNCTIONS USED IN TEXT CLEANING ###


def edit_ents(
    text: str, orig_ents: List[Tuple[int, int, str]]
) -> Tuple[List[Tuple[int, int, str]], bool]:
    """Fix text and entity spans by
        removing trailing whitespace and punctuation
        from the text and spans.

    Args:
        text (str): The text to be cleaned.
        orig_ents (List[Tuple[int, int, str]]): The entity spans.

    Returns:
        Tuple[List[Tuple[int, int, str]], bool]: The cleaned entity spans and a
            boolean indicating if the text was edited.
    """

    editted = False
    # Don't include trailing whitespace from entity spans
    trimmed_ents = []
    for b, e, l in orig_ents:
        if text[b] in trim_chars:
            new_b = b + 1
            editted = True
        else:
            new_b = b

        if text[e - 1] in trim_chars:
            new_e = e - 1
            editted = True
        else:
            new_e = e
        trimmed_ents.append((new_b, new_e, l))
    return trimmed_ents, editted


def fix_entity_annotations(
    text: str, ents: List[Tuple[int, int, str]]
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Clean text and entity spans for cases
        where the entity ends but the next character
        is not a spance.

        e.g. "this is OK you need to fixMe please and hereToo please"
        ents = [(8, 10, "LABEL"), (15, 26, "LABEL"), (36,44,"LABEL")]

        If the start or the end of the entity is a space,
            it is trimmed.

    Args:
        text (str): Text to be cleaned.
        ents (List[Tuple[int, int, str]]): Entity spans.

    Returns:
        Tuple[str, List[Tuple[int, int, str]]]: The cleaned text and entity spans.
    """

    ent_additions = [0] * len(ents)
    insert_index_space = []
    for i, (b, e, l) in enumerate(ents):
        # If the char before the start of this span is not a space,
        # Then update from this ent onwards
        if b != 0:
            if text[b - 1] != " ":
                ent_additions[i:] = [ea + 1 for ea in ent_additions[i:]]
                insert_index_space.append(b)

        # If the next char after this span is not a space,
        # then update the start and endings of all entities after this
        if (e) < len(text):
            if text[e] != " ":
                ent_additions[(i + 1) :] = [ea + 1 for ea in ent_additions[(i + 1) :]]
                insert_index_space.append(e)

    # Fix entity spans
    new_ents = []
    for (b, e, l), add_n in zip(ents, ent_additions):
        new_ents.append((b + add_n, e + add_n, l))

    # Add spaces in the correct places
    b = 0
    new_texts = []
    for e in insert_index_space:
        new_texts.append(text[b:e])
        b = e
    new_texts.append(text[b:])
    new_text = " ".join(new_texts)

    editted = True
    trimmed_ents = new_ents
    while editted:
        trimmed_ents, editted = edit_ents(new_text, trimmed_ents)

    return new_text, trimmed_ents


def _pad_punctuation(text: str) -> str:
    """Pad punctuation marks with spaces,
        to facilitate lemmatisation.

    Args:
        text (str): Text to be cleaned.

    Returns:
        clean_text (str): Text with padded punctuation.
    """

    clean_text = compiled_nonalphabet_nonnumeric_pattern.sub(r" \1 ", text)

    return clean_text


def _detect_camelcase(text: str) -> str:
    """Split camelcase words into separate sentences.

        i.e. "skillsBe" --> "skills. Be"

        Some camelcases are allowed though - these are found and replaced. e.g. JavaScript

        Reference: https://stackoverflow.com/questions/1097901/regular-expression-split-string-by-capital-letter-but-ignore-tla

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Split text with spaces based on camelcase.
    """

    text = compiled_missing_space_pattern.sub(r"\1. \2", str(text))
    for exception in exception_camelcases:
        exception_cleaned = compiled_missing_space_pattern.sub(r"\1. \2", exception)
        if exception_cleaned in text:
            text = text.replace(exception_cleaned, exception)

    return text


def clean_text(text: str) -> str:
    """Pipeline for preprocessing online job vacancy
        and skills-related text. It utilises the following functions:
        - _detect_camelcase: Split camelcase words into separate sentences.
        - _pad_punctuation: Pad punctuation marks with spaces.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned text.
    """

    return pipe(
        text,
        _detect_camelcase,
        _pad_punctuation,  # messes up entity spans
    )


def get_old2new_chars_dict(orig_text: str, new_text: str) -> Dict[int, int]:
    """Map the original text character indices to the new text indices.

        i.e.

        orig_text = "abcd"
        new_text = "ab cd"

        old2new_chars_dict = {0:0, 1:1, 2:3, 3:4}

    Args:
        orig_text (str): Original text.
        new_text (str): New text.

    Returns:
        Dict[int, int]: Dictionary mapping original text
            character indices to new text indices.
    """

    seq_matcher = difflib.SequenceMatcher(None, orig_text, new_text)
    old2new_chars_dict = {}
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "equal":
            step_up = j1 - i1
            for i in range(i1, i2):
                old2new_chars_dict[i] = i + step_up
        elif tag == "insert":
            old2new_chars_dict[i1] = j1
        elif tag == "replace":
            # This shouldnt really be happening since our cleaning is only
            # inserting, but sometimes it does categorise as "replace" in cases where it
            # thinks adding whitespace to either side is a replacement
            # e.g. "abcd" -> " abcd "
            step_up = j1 - i1
            for i in range(i1, i2):
                old2new_chars_dict[i] = i + 1 + step_up

    return old2new_chars_dict


def fix_all_formatting(
    text: str, ents: List[Tuple[int, int, str]]
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Fix all formatting issues in the text and entity spans. This
        includes:
        - Cleaning the text (padding punctuation, splitting camelcase)
        - Fixing entity annotations

    Args:
        text (str): Text to be cleaned.
        ents (List[Tuple[int, int, str]]): Original entity spans.

    Returns:
        Tuple[str, List[Tuple[int, int, str]]]: Cleaned text and entity spans.
    """
    new_text = clean_text(text)
    old2new_chars_dict = get_old2new_chars_dict(text, new_text)

    new_ents = []
    num_index_problems = 0
    for b, e, t in ents:
        new_b = old2new_chars_dict.get(b)
        new_e = old2new_chars_dict.get(e)
        if new_b and new_e:
            new_ents.append((new_b, new_e, t))
        else:
            num_index_problems += 1

    if num_index_problems != 0:
        print(
            f"Problems with {num_index_problems} entity spans - these will be left out of any training or testing"
        )

    return new_text, new_ents


def clean_entities_text(
    text: str, ents: List[Tuple[int, int, str]]
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Clean both text and entities for labelled data using
        label-studio. This includes:
        - Fixing all formatting issues
        - Fixing entity annotations

    Args:
        text (str): Text to be cleaned.
        ents (List[Tuple[int, int, str]]): Original entity spans.

    Returns:
        Tuple[str, List[Tuple[int, int, str]]]: Cleaned text and entity spans.
    """
    text, ents = fix_all_formatting(text, ents)
    text, ents = fix_entity_annotations(
        text, ents
    )  # apply after to deal with the padding
    return text, ents


### FUNCTIONS USED IN MULTISKILL_FLOW ###


def _process_data(
    job_advert_labels: Dict[str, str], all_labels: List[str]
) -> Tuple[str, List[Tuple[int, int, str]], List[str]]:
    """
    Process the raw labelled data but ensure that the span indices are
        correct given they have been labelled using Prodigy and
        label studio.

    Args:
        job_advert_labels : dict
            The raw label-studio labelled data for one job advert
        all_labels : list
            The list of all labels given to entities

    Returns:
        text : str
            The cleaned job advert text
        ent_list : list
            The entity span list (modified after cleaning the text)
            this is in the form [(start_char, end_char, label),...]
    """
    text = job_advert_labels["text"]
    ent_tags = job_advert_labels["labels"]

    ent_list = []
    for ent_tag in ent_tags:
        ent_tag_value = ent_tag["value"]
        label = ent_tag_value["labels"][0]
        ent_list.append((ent_tag_value["start"], ent_tag_value["end"], label))
        if label not in all_labels:
            all_labels.add(label)

    # The entity list is in the order labelled not in
    # character order
    ent_list.sort(key=lambda y: y[0])

    if job_advert_labels.get("type") == "label-studio":
        # Label-studio- specific cleaning, won't work (and not needed) for Prodigy
        text, ent_list = clean_entities_text(text, ent_list)

    return text, ent_list


def _transform_data(entity_list: Union[List[str], str]) -> List[int]:
    """Transform text data into a list of numerical features.

    Args:
        entity_list (Union[List[str], str]): Entity list.

    Returns:
        List[int]: List of numerical features.
    """

    entity_list = [entity_list] if isinstance(entity_list, str) else entity_list

    entity_vec = []
    for entity in entity_list:
        entity_vec.append([len(entity), int(" and " in entity), int("," in entity)])

    return entity_vec


### FUNCTIONS FOR TRAINING NER MODEL ###


def train_ner(
    nlp: Language,
    train_data: list,
    print_losses: bool = config.train.print_losses,
    drop_out: float = config.train.drop_out,
    num_its: int = config.train.num_its,
    learn_rate: float = config.train.learn_rate,
):
    fix_random_seed(config.train.random_seed)
    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    optimizer = nlp.create_optimizer()
    optimizer.learn_rate = learn_rate

    all_losses = []
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in tqdm(range(num_its)):
            random.seed(itn)
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                for text, annotation in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    nlp.update([example], sgd=optimizer, losses=losses, drop=drop_out)
            all_losses.append(losses["ner"])
            if print_losses:
                msg.info(f"Iteration: {itn}; Loss: {losses['ner']}")


def evaluate_ner(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    all_labels: List[str] = config.data.all_labels,
) -> dict:
    """Evaluate the NER model using nervaluate.

    Args:
        y_true (List[List[str]]): True start and end indices of entities.
        y_pred (List[List[str]]): Predicted start and end indices of entities.
        all_labels (List[str], optional): List of predicted labels. Defaults to config.data.all_labels.

    Returns:
        dict: Evaluation results.
    """
    evaluator = Evaluator(y_true, y_pred, tags=all_labels, loader="list")
    results_all, results_per_tag = evaluator.evaluate()

    results_summary = {}

    all_dict = {}
    for ev_type in ["f1", "precision", "recall"]:
        all_dict[ev_type] = results_all["partial"][ev_type]
    results_summary["All"] = all_dict

    for label, lab_res in results_per_tag.items():
        lab_dict = {}
        for ev_type in ["f1", "precision", "recall"]:
            lab_dict[ev_type] = lab_res["partial"][ev_type]
        results_summary[label] = lab_dict

    return {
        "results_summary": results_summary,
        "results_all": results_all,
        "results_per_tag": results_per_tag,
    }
