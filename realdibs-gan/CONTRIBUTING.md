Contributing
============

This project is a community effort within *Disney Research Studios(DRS) - Zurich*
where decisions are made based on technical merit and consensus. Everyone is
welcome to contribute. This document describes how to do so, and provides
some guidelines, good practices and conventions.

Remember that code is not the only way to contribute to this project. Using
it, reporting issues, reviewing pull requests, adopting some of the ideas
here present for your own projects, or simply spreading the word within our
organization is absolutely helpful.

Overview of contribution process
--------------------------------

Changes to **noice-toolbox** are made through [pull-requests(PR)](https://www.atlassian.com/git/tutorials/making-a-pull-request)
either from a [branch](https://www.atlassian.com/git/tutorials/using-branches)
of the noice-toolbox repository, or from a [fork](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).
Those changes usually target the `master` branch which is the current status of
our code base, but they can be done to any other branch of the repository.

Since our team is rather small we all work against the same remote (no forks)
we try to follow the [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
workflow (see [git feature-branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
as well). But with some differences like the fact that we don't have a
`develop` branch. We follow the continuous delivery strategy and therefore
our common develop branch is `master`. Contributors are encouraged to have
their own sandbox as `dev_MyNAME` but thats what they are: sandboxes, and we
take no responsibility with divergences with `master`.

The encouraged branch naming is as follows:

- `master` this is the production branch. The current state of the codebase.
- `release/vX.Y.Z` the release branch for a particular version.
- `feature/awesome_feature` branch where awesome feature is being cooked.
- `mnt/awesome_refactor` branch where awesome refactor is being cooked.
- `hotfix/fix` branch where we amend some our shortcomings.
- `dev_MyNAME` my sandbox.

As a reminder, we try to develop our features as close as possible to
`master`. We develop minimum working options (a completed subset at a time)
of the features and try to integrate them to master as soon as possible
(always with a PR), then iterate and we avoid branches diverging from
`master`.

Commits, Issues and Pull-Request Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rewriting history of branches nobody else is working on is completely fine,
even encouraged in order to keep clean commit messages and content. History
is part of the repo. Commit messages and history bring valuable inside for
code forensics and it should be treated as such with care. On the other hand,
rewriting history in `master` **is not OK**. If ever in need to modify the
history of `master`, everyone should be notified.

Here is an example of a perfectly valid way of working:

```sh
git checkout -b foo origin/master  # create a new local branch called foo from origin master
# hack some code
git commit -am "wip"  # commit my work in progress
# hack some code
git commit -am "xx"  # some more work (maybe xx is descriptive enough)
# hack some code (maybe a nice refactoring https://youtu.be/59YClXmkCVM)
git commit -am ".."  # just a commit
git rebase --interactive HEAD~3  # use git rebase to clean up the history
git push --set-upstream origin foo
# make a pull request
```

We strongly recommend contributors to get familiar with interactive rebasing,
the notion of `HEAD`, as well as `git reset --hard`, `git reset --soft`,
`git clean -f`. They are all supper handy and we are going to make extensive
use of all these commands.

Considerations when filling issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**This section needs lots of love**

We file issues for many reasons (see PRs and Issues tag conventions to get an
idea). But the most common reason is because we found something that needs
fixing. The question is once found a problem what do we do? Here are some ideas:

* **It is easy to solve (I can do it in 30min, and I'm going to do it _nowish_)**, then:
  - Write yourself a note, so that you don't forget.
  - Write a test that reveals the problem. (That might be a good time to make a WIP PR)
  - Fix the problem.
  - Make a PR.

* **It will take me more than couple hours**, then:
  - Write yourself a note, so that you don't forget.
  - Write an issue describing the problem. The issue should contain:
    - A full description of what was expected and what actually happen (both).
    - The environment that produced the unexpected results.
    - A [Mininum Working Example](https://stackoverflow.com/help/minimal-reproducible-example) to reproduce the errors.
    - (and/or) a test revealing the problem.
  - Add it to the backlog with the priority required.

Here are some other tips in [how to write useful issues](https://medium.com/nyc-planning-digital/writing-a-proper-github-issue-97427d62a20f).


Commit Messages and Issues/PR titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Commit messages and titles of issues or PRs are part of the codebase and
therefore they are important. There is no need to overdo them, but they
require some thought. No more or less care than a function or variable name
deserves. In a mature codebase, reading the commit headers is a routine task;
Lists of issues, and pull-requests are constantly filtered, and read in a
list form. The future versions of ourselves will thank us for spending those
extra minutes. Or commits and titles try to follow this (in 50 characters or
less):
```
TAGA, TAGB: some descriptive name
```

Here is an incomplete of some of tags we use, and that are commonly accepted
withing the scientific python community:

```
   API: an (incompatible) API change
   DEP: deprecate something, or remove a deprecated object
   DOC: documentation
   ENH: enhancement
   FIX: bug fix
   HOTFIX: bug fix (those are usually meant to be backported)
   MAINT: maintenance commit (refactoring, typos, etc.)
   REL: related to releasing
   REV: revert an earlier commit (the commit should be in the message)
   STY: style fix (whitespace, PEP8)
   TST: addition or modification of tests
   WIP: work in progress
```

Apart from the tags mentioned above, commit messages can contain special
tags enclosed in square brakcets meant to trigger certain actions by the
project Continuous Integration(CIs) and bots
(i.e: `git commit -am "WIP: some work [skip ci]"`).
Here follows a list of those tags; use them judiciously.

- `[skip ci]`: skip running CI tests. **This is not meant to make CIs green**
  **to justify merging with a broken testsuit. This is meant to reduce**
  **computation time and not get billed by running code that we known in**
  **advance that it will fail.**
- `[skip pep]`: skip [PEP8](https://www.python.org/dev/peps/pep-0008/) linter.
- `[run benchmark]`: recompute all our benchmarks based on this commit.


Pull-requests reviews
^^^^^^^^^^^^^^^^^^^^^

[Here](https://github.com/mne-tools/mne-python/pull/6230) is workout example
from the [mne-python's community](https://mne.tools/) on what to expect in a
PR review: A user submits a PR with a clear description of what changes is
s/he bringing to the codebase and users and developers make constructive
comments about the work. Since the code merged to master will become
everybody's asset/liability.

See some of the python's core-dev **Jack Diederich** taks in reviewing PRs from
his pycon2018 talk: [How to write a function](https://www.youtube.com/watch?v=rrBJVMyD-Gs).

Setting up the development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note: This is similar to the installation instructions in the README.rst, and*
*you can always double check with our Continuous Integration(CI)s config files*
*for discrepancies.*

The recommended way to install and contribute to this project is using
**Conda**. We recommend using `miniconda` and `conda-forge` over `anaconda`
and the official channels. If you have `$PYTHONPATH` or `$PYTHONHOME`
environment variables set you might run into trouble with `Conda` (Check
`Anaconda's trubleshooting guide`_ for more).

Remember that in **conda**-managed environments it is preferible to keep
`$PYTHONPATH` and `$PYTHONHOME` permanently unset and use `conda activate`
and `conda deactivate` to switch between python environments.

Also if you want to keep your development environment separated to your
installation environment, you can substitute `noice` for `noice_dev` in the
following instructions:

```
$ conda env update --name noice --file environment.yml
$ conda env update --name noice --file environment_dev.yml
$ conda activate noice
```

*Remember that you might also need to set up your `~/.noice_toolbox/config.json`.
see README for more.*


noice-toolbox coding conventions
--------------------------------

All new functionality must have test coverage
---------------------------------------------

If a new module, class, function, or virtually any code is added to the
codebase (i.e: `noice/datasets/my_dataset.py`) it should have its
corresponding tests file (i.e: `tests/datasets/test_my_dataset.py`).
Otherwise it wont be added to `master` and your functionalities will keep
drifting without the love and care of the other contributors.

All new functionality must be documented
----------------------------------------

Good coding does not require much documentation. But the required
documentation is extremely necessary and it needs to be in the right place,
with the right amount, and with the right style and conventions. See the
following snippets:

```py
foo = Object_int()

# manipulate foo
a = foo  # get reference
a = a + 1 # foo gets incremented by 1
```

```py
def manipulate(element, _increment=1):
"""Increment an element in place."""
  element += _increment

...

foo = Object_int()
manipulate(foo)
```

The previous two snippets might seem artificial, but they are meant just to
discuss what are we referring to when we are talking about documentation. All
code should be self explanatory. When we have random comments next to lines
of code **that is not documentation**; it is usually a code smell telling us
that there is a better way to write our code. In the same manner having a
title on some lines blob **is not documentation** either. It is a code smell
telling us that such lines are most likely a missing function.

Maybe the naming of the function and its parameters is descriptive enough and
there is no need for further documentation. But if that is not the case, the
[docstring](https://en.wikipedia.org/wiki/Docstring) is the right place where
to document our code. Editors and interpreters know how to use those docstrings
so that we can see the accompanying documentation of the code when needed.

*Side note: How many times do you need to see a snippet of code to*
*factor it out into a function?*

*Only one. There is no need to have code duplication to factor some code into*
*a function. If the benefit of encapsulating them and having some associated*
*documentation is bigger than the cost of calling the function (which it*
*always is) then even a single line that you see once is worth to become a*
*function*

All new functionality should include proper docstring descriptions for all
public API changes, as well as narrative documentation for larger
contributions. Those could be in the form of tutorials, howtos, or worked out
examples. Regarding private functions, docstrings could be more sparse or
less complete. But they should never be neglected.

**noice-toolbox** uses Sphinx follows [NumPy docstring style guidelines](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)

**noice-toolbox** uses
[Numpy style](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
to render the docstrings in sphinx, and this is enforced by our CIs.
Therefore your code would not be accepted unless its properly documented and
follows the guidelines. More over our CIs also have a spellchecker, so mind
your typos. If you need to add some special word add it to `ignore_words.txt`
at the root of the project.

Here follows an example adapted from the
[Shpinx documentation](https://hplgit.github.io/teamods/sphinx_api/html/sphinx_api.html#how-to-format-doc-strings)
illustrating how to write a numpy styled docstring using sphinx's internal
and external code corss-referening, external links, highlighting, code formatting, math rendering using latex; and other Sphinx caveats.

```py
from numpy.lib.scimath import sqrt

def roots(a, b, c, verbose=False):
    """Compute the roots of a quadratic equation (short one-liner title REQUIRED).

    A long description of what this function does (when needed). Here just to
    illustrate some of the features shpinx offers us.

    Return the two roots in the quadratic equation (that will be rendered in
    latex):

    .. math::
        ax^2 + bx + c = 0

    The returned roots are real or complex numbers, depending on the values
    of the arguments `a`, `b`, and `c`.

    .. note::
      	Will add a inline highlight of something that needs consideration.

    .. warning::
        Can be used to get an inline highlight of something that needs
        consideration and its dangerous.

    Some more text explaining how cool this function is. To do so we can use
    all the power of rst to do _italic_, `more italic`, *bold*, etc..
    It is that cool that if we need to use a table to show off we can do so:

    ======   =========   ================================
    Name     is cool     Description
    ======   =========   ================================
    foo      yes         some description of the foo.
    bar      no          some longer description of bar
                         that require a second line.
    ======   =========   ================================

    We can also use the `double colon` (``::``) to render some python code
    (and under some circumstances the code in here will be executed when
    rendering the documentation to make sure that the documentation holds,
    and if its not true it will produce an error. More on this latter)

    ::

      from foo import foo as bar
      bar()

    Enough nonsense narrative, let's see other sections.

    Parameters
    ----------
    a: int | real | complex
       coefficient of the quadratic term
    b: int | real | complex
       coefficient of the linear term
    c: int | real | complex
       coefficient of the constant term
    verbose: bool, (optional)
       prints the quantity ``b**2 - 4*a*c`` and if the roots are real or
       complex.
       Defaults to False.

    Returns
    -------
    root1, root2: real, complex
        the roots of the quadratic polynomial.

    Raises
    ------
    ValueError:
        when `a` is zero.

    See Also
    --------
    :class:`Quadratic`: which is a class for quadratic polynomials
        that also has a :func:`Quadratic.roots` method for computing
        the roots of a quadratic polynomial. There is also a class
        :class:`~linear.Linear` in the module :mod:`linear`
        (i.e., :class:`linear.Linear`).

    .. warning::
        Here we are assuming that ``Quadratic``, ``linear``, ``roots``, etc.
        are part of the **API** of this package. Thats why we can
        cross-refernce them like this.

    .. note::
        We can also cross reference some packages like **numpy** or
        **tensorflow** as if they were part of this package
        (i.e: :func:`tf.nn.sigmoid_cross_entropy_with_logits` to link
        directly to the **tensorflow**'s documentation).
        Note that some sphinx configuration might be required if the package
        is not there yet.

    Notes
    -----
    If the parameters, or returns are super clear from the signature of the
    function, there are no related elements to put in ``See Also``, there
    are no notes, references, nor examples; Then there is no need to add
    any of those sections (add always the bare minimum necessary to have
    complete information).

    The algorithm is a straightforward implementation of a very well known
    formula [1]_, but if some extra explanation from the long description was
    required then ``Notes`` is the right place to explain them.

    References
    ----------
    .. [1] Any textbook on mathematics or
           `Wikipedia <http://en.wikipedia.org/wiki/Quadratic_equation>`_.

    Examples
    --------
    As above stated, code snippets correspond to an indented block after
    ``::``. If the code starts with ``>>>`` the following statement is sent
    to the python interpreter during the documentation rendering, and error
    if the output does not match whatever is written in the subsequent line.
    This is known as a **doctest**.

    ::
      >>> roots(-1, 2, 10)
      (-5.3166247903553998, 1.3166247903553998)
      >>> roots(-1, 2, -10)
      ((-2-3j), (-2+3j))
    """
    if abs(a) < 1E-14:
        raise ValueError(f"a={a} is too close to zero")

    q = b**2 - 4*a*c
    if verbose:
        print("q={q}: {root_type} roots".format(
            q=q,
            root_type='real' if q>0 else 'complex',
        ))

    root1 = (-b + sqrt(q))/float(2*a)
    root2 = (-b - sqrt(q))/float(2*a)
    return root1, root2
```



Code style and notes
--------------------
Here is a none exhaustive list of things we take into account.
- We use [PEP8](https://www.python.org/dev/peps/pep-0008/) but we do not
  enforce any specific linter. We do use `flake8` in our CIs driven by
  `setup.cfg` in the root of the project. To replicate CIs behavior just run:

  ```sh
  flake8
  ```

- [We are all consenting adults](https://python-guide-chinese.readthedocs.io/zh_CN/latest/writing/style.html) (more or less).
- We don't like trailing spaces.
- We use classes only when they are the right choice, we prefer free
  functions and build-in types (we also do an extensive use of `Bunch` and
  its derivatives).
- We like short functions.
- Functions check inputs, do something and return in that order. Unless we
  are throwing exceptions or early returning.
- We like throwing errors (standard errors) rather than changing the return type.
- We type-annotate public API functions.
- We like helper functions; we look into our `utils` before creating a helper.
- We use `base.py` to expose things in the `__init__.py`. (Explain how modules grow)
- We use prepending `_` to denote something local. Most probably should not
  be used out of the current scope.
- Some naming conventions:
  * `*_config_set` is an iterator over multiple configs.
  * `*_keys` is a list of the keys changing across that config iterator.
  * `iter_configs` is an iterator over all combinations of configs (Bunch iterator)

*_cfg is a final use-able single-valued config to use (single Bunch)


**TODO**

Code organization
-----------------
**TODO**

Running the test suite
----------------------
**TODO**

Building the documentation
--------------------------

The documentation is created using
[Sphinx](http://www.sphinx-doc.org/en/stable/).
In addition, the examples are created using `sphinx-gallery`. Therefore, to
generate locally the documentation, you need to make sure that the packages
in `environment-dev.yml` are present.

```sh
$ conda env update -n $MY_ENVIRONMENT --file  environment-dev.yml
```

The documentation is made of:

* a home page, `doc/index.rst`;
* an API documentation, `doc/api.rst` in which you should add all public
  objects for which the docstring should be exposed publicly.
* a User Guide documentation, `doc/user_guide.rst`, containing the narrative
  documentation of your package, to give as much intuition as possible to your
  users.
* examples which are created in the `examples/` folder. Each example
  illustrates some usage of the package. the example file name should start by
  `plot_*.py`.

The documentation is built with the following commands:

```sh
    $ cd doc
    $ make html
```

File header
-----------
File headers are required. They can contain full fledge docstring documentation,
just a one liner or no docstring at all. But they do require the encoding and a
list of authors. This list is just to know who to contact in case someone needs
some clarification. So if you modify a file enough please add yourself to be
contacted.

```py
# -*- coding: utf-8 -*-
"""Descriptive short one-liner title (if required).

longer module description if the one liner is not enough.
"""
# Author: your name <name@disneyresearch.com>
```

If the python file is supposed to be a runnable, add the dynamic python
on the top like this:

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptive short one-liner title (if required).

longer module description if the one liner is not enough.
"""
# Author: your name <name@disneyresearch.com>
```