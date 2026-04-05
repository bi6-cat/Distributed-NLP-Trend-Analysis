from setuptools import find_packages, setup

setup(
    name="crawl_project",
    packages=find_packages(exclude=["crawl_project_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
