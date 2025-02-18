import setuptools


def package_readme():
  with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()
  return content


def package_version(version_file: str):
  with open(version_file, 'r') as f:
    exec(compile(f.read(), version_file, 'exec'))
  return locals()['__version__']



if __name__ == '__main__':
  setuptools.setup(
    name='open-gaze-estimation',
    version=package_version('opengaze/version.py'),
    description='Open Implementations for Gaze Estimation',
    long_description=package_readme(),
    long_description_content_type='text/markdown',
    author='Elorfiniel',
    author_email='markgenthusiastic@gmail.com',
    keywords='computer vision, gaze estimation, eye tracking',
    url='https://gitee.com/elorfiniel/open-implementations-for-gaze-estimation',
    packages=setuptools.find_packages(include=['opengaze', 'opengaze.*']),
    classifiers=[
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11',
      'Programming Language :: Python :: 3.12',
    ],
    license='MIT License',
    python_requires=">=3.9",
  )
