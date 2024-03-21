import pytest

from salary_extraction_utils import annualise_salary


def test_annualise_salary():
    assert annualise_salary({
        "raw_salary_unit": "hour",
        "raw_salary_currency": "GBP",
        "raw_salary": 10,
        "raw_min_salary": 10,
        "raw_max_salary": 20
    }) == {
        "min_salary": 10,
        "max_salary": 20,
        "min_annualised_salary": 20800,
        "max_annualised_salary": 41600,
        "rate": "hour"
    }

    assert annualise_salary({
        "raw_salary_unit": "hour",
        "raw_salary_currency": "GBP",
        "raw_salary": 10,
        "raw_min_salary": 10,
        "raw_max_salary": 20
    }) == {
        "min_salary": 10,
        "max_salary": 20,
        "min_annualised_salary": 20800,
        "max_annualised_salary": 41600,
        "rate": "hour"
    }
    assert annualise_salary({
        "raw_salary_unit": "hour",
        "raw_salary_currency": "GBP",
        "raw_salary": 10,
        "raw_min_salary": 10,
        "raw_max_salary": 20
    }) == {
        "min_salary": 10,
        "max_salary": 20,
        "min_annualised_salary": 20800,
        "max_annualised_salary": 41600,
        "rate": "hour"
    }
    assert annualise_salary({
        "raw_salary_unit": "hour",
        "raw_salary_currency": "GBP",
        "raw_salary": 10,
        "raw_min_salary": 10,
        "raw_max_salary": 20
    }) == {
        "min_salary": 10,
        "max_salary": 20,
        "min_annualised_salary": 20800,
        "max_annualised_salary": 41600,
        "rate": "hour"
    }
    assert annualise_salary({
        "raw_salary_unit": "hour",
        "raw_salary_currency": "GBP",
        "raw_salary": 10,
        "raw_min_salary": 10,
        "raw_max_salary": 20
    }) == {
        "min_salary": 10,
        "max_salary": 20,
        "min_annualised_salary": 20800,
        "max_annualised_salary": 41600,
        "rate": "hour"
    }
