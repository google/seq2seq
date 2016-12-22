## Development Setup

- Install Python3. If you're on a Mac the easiest way to do this is probably using [Homebrew](http://brew.sh/): `brew install python3`
- Clone this repository: `git clone https://github.com/dennybritz/seq2seq.git`. Change into it: `cd seq2seq`
- Create a new virtual environment and activate it: `python3 -m venv ~/path/to/your/venv`. Then, `source ~/path/to/your/venv/bin/activate`
- Install package dependencies: `pip install -e .`
- Install testing utilities: `pip install nose pylint tox yapf`
- Run tests and make sure they pass: `nosetests`
- Code :)

## Github Workflow

Pushing directly to the master branch is blocked. In order to make changes you must:

- Make a new branch for your feature. For example, `git checkout -b feature/my-new-feature`
- Make changes and commits
- Run:
   - `nosetests` to make sure tests are passing
   - `pylint ./seq2seq` for linting and catching obvious errors
   - `yapf -ir ./seq2seq` to auto-format code
- Push your new branch to Github: `git push`
- Create a Pull Request on Github and make sure CircleCI tests are passing
- Have one person review before merging the change

To make things easier you can also use the [Github Desktop app](https://desktop.github.com/).

## General Style Guidelines

- Run [YAPF](https://github.com/google/yapf) to format your code, e.g. `yapf -ir ./seq2seq`.
- Run [pylint](https://www.pylint.org/).
- Code must be compatible with Python 2/3 using [futurize](http://python-future.org/futurize.html). That is, code should be written in Python 3 style and made backwards compatible with Python 2 by adding the appropriate imports.
- All public functions and classes must have docstring [following this style](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments).
- When in doubt, follow [this Python style guide](https://google.github.io/styleguide/pyguide.html). Running pylint should take care of most issues though.

## Tensorflow Style

### GraphModule

All classes that modify the Graph should inherit from `seq2seq.graph_module.GraphModule`, which is a wrapper around Tensorflow's [`tf.make_template`](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#make_template) function that allows for easy variable sharing. Basically, it allows you to do something like this:

```python
encode_fn = SomeEncoderModule(...)

# New variables are created in this call.
output1 = encode_fn(input1)

# No new variables are created here. The variables from the above call are re-used.
# Note how this is different from normal Tensorflow where you would need to use variable scopes.
output2 = encode_fn(input2)

# Because this is a new instance a second set of variables is created
encode_fn2 = SomeEncoderModule(...)
output3 = encode_fn2(input3)
```

### Functions vs. Classes

- Operations that **create new variables** must be implemented as classes and must inherit from `GraphModule`.
- Operations that **do not create new variables** can be implemented as standard python functions, or as classes that inherit from `GraphModule` if they have a lot of logic.
