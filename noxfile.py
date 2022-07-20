import nox

PROJECT = 'emonet'


@nox.session(python=False)  # running in current environment to avoid rebuilding torch stuff
def tests(session):
    """Run unit tests in current Python environment."""
    # session.install('pytest', 'pytest-cov')
    # session.install('.')
    session.run('pytest')


@nox.session(reuse_venv=True)
def lint(session):
    """Lint source code using Pylint, Flake8 and Mypy."""
    session.install('pylint', 'flake8', 'mypy')
    # session.run('pylint', PROJECT, '--verbose')
    session.run('flake8', PROJECT, '--count', '--statistics', '--select=E9,F63,F7,F82', '--show-source')  # these fail
    session.run('flake8', PROJECT, '--count', '--statistics', '--exit-zero')  # these warn
    session.run('mypy', '--install-types')
    session.run('mypy', '-p', PROJECT)


@nox.session(python=False)  # running in current environment to avoid rebuilding torch stuff
def docs(session):
    """Build package documentation."""
    # session.install('.')
    # session.install('sphinx', 'furo', 'myst-parser')
    session.run('sphinx-apidoc', '--separate', PROJECT, '-o', 'docs/source/')
    session.run('sphinx-build', '-b', 'html', 'docs/source/', 'docs/build/html')


@nox.session(reuse_venv=True)
def qa(session):
    """Run QA code checks."""
    session.install('check-manifest', 'isort', 'pre-commit', 'black')
    session.run('check-manifest')
    session.run('isort', '.')
    session.run('black', PROJECT)
    session.run('pre-commit', 'run', 'trailing-whitespace', '--files', '*.py')
    session.run('pre-commit', 'run', 'end-of-file-fixer', '--files', '*.py')
