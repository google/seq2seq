## Development Setup

1\. Install Python3. If you're on a Mac the easiest way to do this is probably using [Homebrew](http://brew.sh/). Then,

```bash
# Clone this repository.
git clone https://github.com/dennybritz/seq2seq.git
cd seq2seq

# Create a new virtual environment and activate it.
python3 -m venv ~/tf-venv
source ~/tf-venv/bin/activate

# Install package dependencies and utilities.
pip install -e .
pip install nose pylint tox yapf mkdocs

# Make sure tests are passing.
nosetests

# Code :)
```

## Github Workflow

Pushing directly to the master branch is disabled and you must create feature branches and submit them via pull request. To make things easier you can also use the [Github Desktop app](https://desktop.github.com/). A typical workflow looks as follows:

```
# Make sure you are in the seq2seq root directory.
# Start from the master branch.
git checkout master

# Pull latest changes from Github.
git pull

# Create a new feature branch.
git checkout -b feature/my-new-feature

# Make changes and commits
echo "blabla" >> test
git commit -am "Test commit"

# Push the branch upstream.
git push -u origin/my-new-feature

# Submit a pull request on Github.
```

After you submit a Pull Request, one person must review the change and
CircleCI integration tests must be passing before you can merge into the
master branch.


## Python Style

We use [pylint](https://www.pylint.org/) and [yapf](https://github.com/google/yapf) for automated code formatting. Before submitting a pull request, make sure you run them:

```bash
# Run this and fix all errors.
pylint ./seq2seq

# Optional, run this to auto-format all code.
yapf -ir ./seq2seq
```

Note that CircleCI integration test will fail if pylint reports any critical
errors, preventing you from merging your changes.

## Tensorflow Style

### GraphModule

All classes that modify the Graph should inherit from `seq2seq.graph_module.GraphModule`, which is a wrapper around TensorFlow's [`tf.make_template`](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#make_template) function that enables easy variable sharing. Basically, it allows you to do something like this:

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