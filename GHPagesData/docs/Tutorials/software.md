# Python

## The command line

You'll want to have access to a computer with a command line interface - on Mac or Linux, this is just going to be via the app Terminal. On Windows this is called the Command Prompt. (If you have trouble installing you can also use the Google Colab I will mention later!)

## Installing Python

Most software development will be done in Python, and I recommend using Anaconda to install Python 3 and pip to manage packages. What it does is create its own version of Python that doesn't interfere with your default install, and has 'environments' into which you can install software safely that doesn't interact with software in other environments.

There is a great [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) with lots of tools to help you use Conda.

Anaconda is available on the UQ Digital Workspaces as the package `Anaconda3-2020.02`. 

To install Python on a Mac or Linux machine of your own, I recommend you install Conda from [here](https://www.anaconda.com/products/individual).

When working in Python, it is best to create a new environment for each project. For this project, you will want these important packages pre-installed:
- `pip`, which installs other Python packages,
- `numpy`, which is a general-purpose maths library,
- `matplotlib`, the general-purpose Python plotting library,
- `astropy`, which has lots of functions for astronomy, including how we will load data, and the Lomb-Scargle Periodogram for period determination,
- `scipy`, with miscellaneous scientific Python features, and
- `ipykernel`, which runs Jupyter notebooks.

You can create a new environment called `ladder` with all of this, using the following terminal commands:

```shell
conda init
conda create --name ladder python=3.9 pip numpy matplotlib astropy scipy ipykernel
```

Then you want to *activate* this ennvironment, and install the Jupyter notebook packages.

```shell
conda activate ladder
conda install -c conda-forge notebook
conda install -c conda-forge jupyterlab
conda install -c conda-forge nb_conda_kernels
``` 

Then you can work to your heart's content in this conda environment. 

### Python in Windows

Most software development is done in Unix-based operating systems, the main examples being the Linux distributions, and Apple's Mac OSX. For a long time it has been a bit of a hassle to use the latest tools in Windows, but things have got a lot better recently!

One recent development is the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install), which runs a Linux distribution (by default Ubuntu, which is my favourite too) inside of Windows. You should be able to do basically everything I suggested on the command line through this. By default, on Windows 10 or later, you should be able to open a PowerShell command line and type

```shell
wsl --install
```

and it should install; if this doesn't work, there are more detailed instructions [here](https://docs.microsoft.com/en-us/windows/wsl/install).

## Using Python

There is a great intro textbook for Python for astronomers, freely available online [here](https://prappleizer.github.io/) by Pasha & Agostino.


## Jupyter Notebooks

The main way that professional data scientists, physicists, and astronomers interactively use Python is through the Jupyter Notebook environment. This is an interactive, browser-based interpreter for Python.

Here is a [great tutorial]((https://www.dataquest.io/blog/jupyter-notebook-tutorial/)) on using Jupyter notebooks - if you haven't before, I recommend you try this! 

If you install Conda like above, you'll be good to go with the terminal command `jupyter notebook`. 

### Google Colab

While some Windows users have Conda working very nicely, some users have been having difficulty.

If this is the case for you, I recommend you use __Google Colab__, a free web-hosted Jupyter notebook environment that works like Google Docs. (There are paid versions, but you will not need to use these right now.)

Try it [here](https://colab.research.google.com/)! You can upload your project data to your Google drive, and instead of using `conda` to manage your environment and installed packages, in the first cell just execute

```shell
! pip install numpy matplotlib astropy scipy ipykernel
```

I have created Google Colab versions of the tutorial pages from this site: 

- [HR Diagram](https://colab.research.google.com/drive/1dKY_ERciOdq0aSoDaUmNJLO5rkh6Ho1t?usp=sharing)
- [Lomb-Scargle Periodogram](https://colab.research.google.com/drive/11EYzk_5cyNpCWWItp0NusuBRHh__TY-l?usp=sharing)

The important thing to remember is you will have to host your project data on your Google Drive or GitHub and download it when you start your Colab session.

## Git Repositories

How do you share code with your colleagues? This is an important skill not just for this course - but anything you do down the track in science, software engineering, or the commercial sector. This doesn't just apply to Python scripts you'll write here, but to any sort of code - including this website.

You want to have everything in a *repository*, which is a remote server in the cloud that has all your software backed up, with saved checkpoints you can go back to if something is wrong.

The most popular software for interacting with repositories is `git`, and there are two big companies that offer comparable cloud storage for your repos: 
- Atlassian offers [BitBucket](https://bitbucket.org/product)
- Microsoft runs [GitHub](https://github.com/)

You can share these codes with your teammates and jointly collaborate on a project. The user interface is a little harder than Google Docs but I recommend you master it - it is a huge transferable skill.  

All my code is version-controlled [github.com/benjaminpope](https://github.com/benjaminpope/). On my local machine I made a directory `/Users/benjaminpope/code/`, using terminal commands

```shell
cd .
mkdir code
```
and you can download a repo (like this website as an example!) to your machine like this:

```shell
cd code
git clone https://github.com/benjaminpope/ladder/
cd ladder
```

A really great way of organizing this project, from past experience, is if you create a repository of your own, shared with your group, and develop there. 

## Making Open-Source Software

Christina Hedges (NASA Ames) has a fantastic introduction to open-source software practices for astronomy, in which the above tools and many others are explained, for an audience of ~ PhD students:

- [christinahedges.github.io/astronomy_workflow](https://christinahedges.github.io/astronomy_workflow/)