from setuptools import setup

setup(
    name="seq2seq",
    version="0.1",
    install_requires=[
        "numpy",
        "https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.12.1-py3-none-any.whl",
        "scikit-learn"
    ]
)
