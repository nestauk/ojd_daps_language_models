# Salary Extraction

This is a simple project to extract an annualised salary from a job advert. The model is trained on a dataset of job adverts with salary information, and is fine-tuned on a smaller dataset of job adverts with salary information.

## 🚀 Running the salary extraction

We have refactored this into an easy to use python package, located [here](https://github.com/nestauk/ojd_daps_salaries), which can be pip installed, using:

```bash
pip install git+https://github.com/nestauk/ojd_daps_salaries.git
```

The package can be used as follows:

```python
from ojd_daps_salaries import annualise_salary

salary = {
    "raw_min_salary": 20000,
    "raw_max_salary": 30000,
    "raw_salary_currency": "GBP",
    "raw_salary_unit": "YEAR"
}

annualised_salary = annualise_salary(salary)
print(annualised_salary)
```

This will output the annualised salary, which is calculated as the average of the min and max salary, multiplied by the number of units in a year.

To use with a dataset of job adverts, where the salary information is stored in a pandas DataFrame, you can use the following code:

```python
import pandas as pd
from ojd_daps_salaries import annualise_salary

df = pd.read_csv("path/to/your/data.csv")

df["annualised_salary"] = df.apply(lambda x: annualise_salary(x), axis=1)
```

## Testing the salary extraction

To test the salary annualisation, you can run the following command, providing you have pytest installed:

```bash
python -m pytest test_salary_annualisation.py
```

We have covered the following test cases in our tests:

- The annualisation of a salary works as expected
- Ensuring that the max annual salary is greater than the min annual salary
- Ensuring the the max annual salary is not more than 10x the min annual salary
- The extraction works for different rates of pay (e.g. per hour, per day, per week, per month)




