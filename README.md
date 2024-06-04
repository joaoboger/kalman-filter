# Kalman-Filter Charged Particle Tracker

This Git repository serves as the culmination of my MSc project, focusing on the implementation of the Kalman-Filter technique from scratch. The project centers around a 2D toy model charged particle tracker, specifically designed to work with a transverse magnetic field.

## Repository Structure

- **data/**
  - This folder contains the datasets used for the project. These datasets are used as input for the Kalman-Filter implementations.

- **makeTracks/**
  - This directory includes scripts and utilities for generating and manipulating track data for charged particles.

- **utils/**
  - Utility scripts and helper functions that support the main Kalman-Filter implementations.

- **CKFwithChargedParticles.py**
  - Implementation of the Combinatorial Kalman-Filter algorithm with charged particles.

- **KFwithChargedParticles.py**
  - Basic Kalman-Filter implementation specifically for tracking charged particles in a transverse magnetic field.

- **KFwithNeutralParticles.py**
  - Implementation of the Kalman-Filter algorithm for tracking neutral particles, included for comparison purposes.

## Project Description

The primary goal of this project is to implement the Kalman-Filter technique from scratch for tracking charged particles in a 2D environment with a transverse magnetic field. The project includes multiple implementations of the Kalman-Filter algorithm to handle different types of particles and scenarios.

### Key Features

- **Kalman-Filter Implementations:** 
  - The repository includes three main Python scripts that implement the Kalman-Filter technique for different particle scenarios: CKF with charged particles (`CKFwithChargedParticles.py`), basic charged particles (`KFwithChargedParticles.py`), and neutral particles (`KFwithNeutralParticles.py`).

- **Utility Functions:**
  - The `utils` directory provides additional support functions that aid in data processing, visualization, and algorithm implementation.
