# Example: SQL Prompt to SQL Translation

This example demonstrates a single sample from the `gretelai/synthetic_text_to_sql` dataset after it has been formatted for instruction tuning.

---

## Input Record (Raw Dataset)

- **sql_context:**
  The `employees` table contains employee_id, first_name, last_name, salary, department_id.

- **sql_prompt:**
  Retrieve the full name and salary of employees working in department 3.

- **sql:**
  SELECT CONCAT(first_name, ' ', last_name) AS full_name, salary  
  FROM employees  
  WHERE department_id = 3;

- **sql_explanation:**
  This query selects first and last names of employees, combines them as full name, filters those whose department is 3, and retrieves their salaries.

---

## Final Prompt Sent to Model

```text
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that approptiately completes the request.

### Instruction
Company database: The `employees` table contains employee_id, first_name, last_name, salary, department_id.

### Input:
SQL Prompt: Retrieve the full name and salary of employees working in department 3.

### Response:
SQL : SELECT CONCAT(first_name, ' ', last_name) AS full_name, salary
FROM employees
WHERE department_id = 3;

Explanation: This query selects first and last names of employees, combines them as full name, filters those whose department is 3, and retrieves their salaries.
```

---

This formatted example is used during fine-tuning to teach the model how to generate both SQL and explanations from a textual prompt.