import pytest

from .salary_extraction_utils import annualise_salary


job_ad_1 = {
    "raw_salary": 30000,
    "raw_salary_unit": "YEAR",
    "raw_salary_currency": "GBP"
}
job_ad_2 = {
    "raw_salary": 30000,
    "raw_salary_unit": "YEAR",
    "raw_salary_currency": "USD"
}
job_ad_3 = {
    "raw_salary_unit": "YEAR",
    "raw_salary_currency": "GBP",
    "raw_min_salary": 25000,
    "raw_max_salary": 35000
}
job_ad_4 = {
    "raw_salary_unit": "DAY",
    "raw_salary_currency": "GBP",
    "raw_min_salary": 250,
    "raw_max_salary": 350
}
job_ad_5 = {
    "raw_salary_unit": "YEAR",
    "raw_salary_currency": "GBP",
    "raw_min_salary": 25000,
    "raw_max_salary": 350000
}
job_ad_6 = {
    "raw_salary_unit": "YEAR",
    "raw_salary_currency": "GBP",
    "raw_min_salary": 65000,
    "raw_max_salary": 650000
}

def test_annualise_salary():
    assert annualise_salary(job_ad_1) == {
        "min_salary": 30000,
        "max_salary": 30000,
        "min_annualised_salary": 30000,
        "max_annualised_salary": 30000,
        "rate": "YEAR"
    }
    assert annualise_salary(job_ad_2) == None
    assert annualise_salary(job_ad_3) == {
        "min_salary": 25000,
        "max_salary": 35000,
        "min_annualised_salary": 25000,
        "max_annualised_salary": 35000,
        "rate": "YEAR"
    }
    assert annualise_salary(job_ad_4) == {
        "min_salary": 250,
        "max_salary": 350,
        "min_annualised_salary": 65000,
        "max_annualised_salary": 91000,
        "rate": "DAY"
    }
    assert annualise_salary(job_ad_5) == {
        "min_salary": 350000,
        "max_salary": 350000,
        "min_annualised_salary": 350000,
        "max_annualised_salary": 350000,
        "rate": "YEAR"
    }
    assert annualise_salary(job_ad_6) == None
