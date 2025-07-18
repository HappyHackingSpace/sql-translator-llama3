name: Bug Report
description: Report a reproducible issue or unexpected behavior
title: "[Bug]: "
labels: [bug]
body:
  - type: textarea
    attributes:
      label: Description
      description: Describe the problem you’re experiencing.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Expected Behavior
      description: What should have happened?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Actual Behavior
      description: What actually happened?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: Provide a list of steps to reproduce the issue.
    validations:
      required: true

  - type: input
    attributes:
      label: Environment
      description: OS, Python version, packages, etc.
