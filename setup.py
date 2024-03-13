import setuptools
##############################################

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read()


script_list=[]

setuptools.setup(
     name='wstats',  
     version='0.0.99',
     author="Scott Tyler",
     author_email="scottyler89@gmail.com",
     description="An implementation of a few weighted statistics",
     long_description_content_type="text/markdown",
     long_description=long_description,
     install_requires = install_requires,
     url="https://github.com/scottyler89/weighted_stats/",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

