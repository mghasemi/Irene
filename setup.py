try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

Description = "Solve a generic optimization problem based on truncated moment\
				problem by constructing a series of semidefinite relaxations."

setup(
    name='Irene',
    version='1.1.0',
    author='Mehdi Ghasemi',
    author_email='mehdi.ghasemi@gmail.com',
    packages=['Irene'],
    url='https://github.com/mghasemi/Irene.git',
    license='MIT License',
    description=Description,
    long_description=open('README.rst').read(),
    keywords=["Optimization", "Semidefinite Programming", "Convex Optimization",
              "Polynomial Optimization", "Non-Convex Optimization"],
    install_requires=['sympy', 'numpy', 'joblib']
)
