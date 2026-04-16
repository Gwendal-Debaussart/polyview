# Contributing to polyview

We welcome contributions to polyview! If you have an idea for a new feature, have found a bug, or want to improve the documentation, please feel free to submit a pull request directly on the [github repository](https://github.com/Gwendal-Debaussart/polyview).

## Submitting an issue


If planning on submitting an issue, please do your best to follow the following guidelines:

- **Provide a minimal, reproducible example** that demonstrates the issue. Additional information can be found here [https://stackoverflow.com/help/minimal-reproducible-example/](https://stackoverflow.com/help/minimal-reproducible-example/).
- If the previous point is not applicable, **please provide a detailed description of the issue**, including the expected behavior and the actual behavior. Include which functions or classes are involved, and any relevant error messages or stack traces.
- **Please include your environment details**, such as the version of polyview you are using, your Python version, and any other relevant libraries or dependencies.

## Contributing code

If planning on contributing code, here are some additional guidelines to follow.

### Coding guidelines

- **Follow the existing code style** as closely as possible. This includes naming conventions, formatting, and documentation style. We use `ruff` for code formatting and `numpy` style for docstrings.
- **Format docstrings** properly, as they are used for documentation generation by Sphinx. Make sure to include descriptions of parameters, return values, and any relevant reference.

### API of polyview objects

Please ensure that all public methods and attributes of the proposed change follows the API of polyview. The API of polyview objects is designed to be consistent with `sklearn`'s API, which includes methods such as ``fit``, ``predict``, and ``fit_predict``.

**Initialization.** The initialization of method through ``__init__`` may accept some values that changes the behavior of the method, but should not take as input the data itself. The data should be passed to the ``fit`` method. All parameters passed to the ``__init__`` method should have default values, so that the method can be initialized without any arguments.

**Fitting.** The ``fit`` method should be used to train the model on the provided data, and should return the instance itself. Generally, the ``fit`` method should take as input a list of views and an optional array of labels.

## Code of conduct

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting