name: Feature Request
description: Suggest a new feature or enhancement
title: "[Feature]: "
labels: [enhancement]
body:
  - type: textarea
    attributes:
      label: Problem
      description: What problem would this feature solve?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Proposed Solution
      description: Describe the feature or enhancement you’d like to see.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Alternatives
      description: Any alternative solutions or features you’ve considered?
    validations:
      required: false

  - type: input
    attributes:
      label: Additional Context
      description: Links, screenshots, references, etc.
