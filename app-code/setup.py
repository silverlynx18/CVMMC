#!/usr/bin/env python3
"""
Setup script for Pedestrian Counting System for Oasys MassMotion

Installation:
    pip install -e .

Or for development:
    python setup.py develop
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="pedestrian-counting-system",
    version="1.0.0",
    description="Computer vision system for pedestrian traffic analysis and MassMotion data generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/pedestrian-counting-system",
    
    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'archive', 'archive.*']),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Entry points (console scripts)
    entry_points={
        'console_scripts': [
            'analyze-workflow=analyze_workflow:main',
            'analyze-peak-window=analyze_peak_window:main',
            'compare-peak-detection=compare_peak_detection:main',
            'zone-editor=zone_editor:main',
            'line-editor=line_editor:main',
            'launch-gui=launch_gui',
            'generate-analytic-video=generate_analytic_video_from_results:main',
        ],
    },
    
    # Additional metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords
    keywords="pedestrian counting, computer vision, traffic analysis, massmotion, yolo, tracking",
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/yourusername/pedestrian-counting-system",
        "Source": "https://github.com/yourusername/pedestrian-counting-system",
        "Issues": "https://github.com/yourusername/pedestrian-counting-system/issues",
    },
)
