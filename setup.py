import os
from setuptools import find_namespace_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name="gcpds-optimization",
    version='0.1',
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["RPINN*", "CLRPINN*"]),
    author="Diego Armando Perez Rosero",
    author_email="dieaperezros@unal.edu.co",
    maintainer="Diego Armando Perez Rosero",
    maintainer_email="dieaperezros@unal.edu.co",
    download_url='',
    install_requires=[
        "tensorflow==2.15.0",
        "cvxpy[CLARABEL,CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "ipython",
    ],
    scripts=[
    ],
    include_package_data=True,
    license='Simplified BSD License',
    description="",
    zip_safe=False,
    long_description=README,
    long_description_content_type='text/markdown',
    python_requires='>=3.9,<3.12',

    #https://pypi.org/classifiers/
    classifiers=[
    ],
)
  
