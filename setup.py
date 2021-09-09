from setuptools import setup, find_packages

setup(
        name='bilayer_letb',
        version='0.0.2',
        author="Shivesh Pathak, Tawfiq Rakib, Run Hou, Andriy Nevidomskyy, Elif Ertekin, Harley T. Johnson, Lucas K. Wagner",
        author_email="shiveshapathak@gmail.com, lkwagner@illinois.edu",
        description="LETB model for twisted bilayer graphene",
        long_description="LETB model for twisted bilayer graphene",
        url="https://github.com/qmc-hamm/bilayer_tight_binding",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        python_requires=">=3.6, <4",
        install_requires = ["numpy", "scipy", "pandas", "h5py", "ase", "pythtb"],
)
