## What to work on

We are always looking for contributors. If you are interested in contributing but are not sure to what work on, take a look at the open [Github Issues](https://github.com/google/seq2seq/issues) that are unassigned. Those with the `help wanted` label are especially good candidates. If you are working on a larger task and unsure how to approach it, just leave a comment to get feedback on design decisions. We are also always looking for the following:

- Fix issues with the documentation (typos, outdated docs, ...)
- Improve code quality through refactoring, more tests, better docstrings, etc.
- Implement standard benchmark model found in the literature
- Running benchmarks on standard datasets

## Development Setup

We recommend using Python 3. If you're on a Mac the easiest way to do this is probably using [Homebrew](http://brew.sh/). Then,

```bash
# Clone this repository.
git clone https://github.com/google/seq2seq.git
cd seq2seq

# Create a new virtual environment and activate it.
python3 -m venv ~/tf-venv
source ~/tf-venv/bin/activate

# Install package dependencies and utilities.
pip install -e .
pip install nose pylint tox yapf mkdocs

# Make sure the tests are passing.
nosetests

# Code :)

# Make sure the tests are passing
nosetests

# Before submitting a pull request,
# run the full test suite for Python 3 and Python 2.7
tox
```

## Python Style

We use [pylint](https://www.pylint.org/) to enforce coding style. Before submitting a pull request, make
sure you run:

```bash
pylint seq2seq
```

CircleCI integration tests will fail if pylint reports any critica errors, preventing use from merging your changes. If you are unsure about code formatting, you can use [yapf](https://github.com/google/yapf) for automated code formatting:

```bash
yapf -ir ./seq2seq/some/file/you/changed
```

## Recommended Tensorflow Style

### GraphModule

All classes that modify the Graph should inherit from `seq2seq.graph_module.GraphModule`, which is a wrapper around TensorFlow's [`tf.make_template`](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#make_template) function that enables easy variable sharing, allowing you to do something like this:

```python
encode_fn = SomeEncoderModule(...)

# New variables are created in this call.
output1 = encode_fn(input1)

# No new variables are created here. The variables from the above call are re-used.
# Note how this is different from normal TensorFlow where you would need to use variable scopes.
output2 = encode_fn(input2)

# Because this is a new instance a second set of variables is created.
encode_fn2 = SomeEncoderModule(...)
output3 = encode_fn2(input3)
```

### Functions vs. Classes

- Operations that **create new variables** must be implemented as classes and must inherit from `GraphModule`.
- Operations that **do not create new variables** can be implemented as standard python functions, or as classes that inherit from `GraphModule` if they have a lot of logic.