# Open Source Software

When we say software is open-source, we mean that you can access the code, and are allowed to use it and modify it yourself under various light conditions. One of the great things about astronomy is that more than any other natural science we have a culture of open data, open-source software, and open access publishing - if you have a laptop and an internet connection, the barrier between you and looking at real cutting-edge astronomical data is very low. Part of the motivation for this course is to get you familiar with these tools and ideas, that you can use yourself in physics or any other subject or industry.

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
git clone https://github.com/astrouq/ladder/
cd ladder
```

A really great way of organizing this project, from past experience, is if you create a repository of your own, shared with your group, and develop there. 

## Making Open-Source Software

Christina Hedges (NASA Ames) has a fantastic introduction to open-source software practices for astronomy, in which the above tools and many others are explained, for an audience of ~ PhD students:

- [christinahedges.github.io/astronomy_workflow](https://christinahedges.github.io/astronomy_workflow/)