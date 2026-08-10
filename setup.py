try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

Description = (
    "Polynomial optimization via SOS/SONC/SDP hierarchies - modernized rewrite "
    "with SymEngine, CVXPY, and structural reductions."
)

setup(
    name='Irene',
    version='2.0.0.dev0',
    author='Mehdi Ghasemi',
    author_email='mehdi.ghasemi@gmail.com',
    packages=['Irene', 'pyProximation'],
    url='https://github.com/mghasemi/Irene.git',
    license='MIT License',
    python_requires='>=3.10',
    description=Description,
    long_description=open('README.rst', encoding='utf-8').read(),
    long_description_content_type='text/x-rst',
    keywords=[
        'polynomial-optimization',
        'semidefinite-programming',
        'SOS',
        'SONC',
        'moment-problem',
        'differential-algebra',
    ],
    install_requires=[
        'sympy>=1.12',
        'numpy>=1.24',
        'scipy>=1.10',
        'cvxpy>=1.3',
        'cvxopt>=1.3',
        'gpkit>=1.0',
        'multiprocess>=0.70',
    ],
    extras_require={
        'symengine': ['symengine>=0.9'],
        'dev': [
            'pytest>=7.0',
            'pytest-timeout>=2.0',
            'pytest-cov>=4.0',
            'coverage>=7.0',
            'pyyaml>=6.0',
        ],
        'solvers': [
            'clarabel>=0.6',
            'scs>=3.0',
            'osqp>=1.0',
        ],
    },
)
