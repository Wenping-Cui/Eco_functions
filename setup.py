from setuptools import setup

setup(name='Eco_function',
    version='0.1',
    description='Simulations about consumer resource models',
    url='https://github.com/Wenping-Cui/Eco_functions',
    author='Wenping Cui',
    author_email='wenpingcui@gmail.com',
    license='MIT',
    packages=['Eco_function'],
    install_requires=['numpy','scipy','cvxopt','pandas','matplotlib'],
    zip_safe=False)
