                                           #  AI-Based Sign Language Translator (PT442)

An AI-powered, real-time, bidirectional sign language translation project developed as a Major Project at IILM University, Greater Noida.

The proposed system combines Computer Vision, Deep Learning, Natural Language Processing (NLP), Speech Processing, and 3D Computer Graphics to support two-way communication between sign-language users and non-signers.

# Project Overview

The system has two main communication directions:

1. Sign Language → Text / Speech

Camera / Video
      ↓
OpenCV + MediaPipe
      ↓
Hand + Face + Pose Landmarks
      ↓
ST-GCN / Transformer
      ↓
Sign / Gloss Sequence
      ↓
NLP
      ↓
English Text
      ↓
Text-to-Speech

2. Text / Speech → Sign Language

Text / Voice Input
      ↓
Speech-to-Text (if required)
      ↓
NLP / Gloss Processing
      ↓
Sign Sequence / Motion
      ↓
3D Avatar
      ↓
Animated Sign Output

The proposal focuses on a software-only, camera-based approach that avoids specialized wearable gloves and  integrated mobile camera hardware.

# Problem

Existing sign-language assistance systems can face several limitations:

Some solutions depend on intrusive wearable hardware such as smart gloves.

Static image classifiers are limited when handling continuous signing.

Continuous sign recognition requires modelling temporal movement and contextual relationships.

Many systems focus mainly on one-way Sign-to-Text translation instead of bidirectional communication.

3D avatar systems may struggle with natural, continuous sign transitions.

# Objectives

The Phase I project defines the following objectives:

Implement real-time landmark tracking using OpenCV and MediaPipe.

Develop and evaluate a deep-learning gesture recognition model using Transformer / ST-GCN approaches.

Build an NLP module for converting spoken/text input into a standardized sign-language gloss representation.

Integrate Speech-to-Text and Text-to-Speech functionality.

Develop a procedural 3D avatar system capable of displaying continuous sign movements.

# Proposed Architecture

                    ┌─────────────────────────┐
                    │        USER INPUT        │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
       Sign / Video Input                      Text / Voice
              │                                     │
              ▼                                     ▼
       OpenCV + MediaPipe                     STT / Text Input
              │                                     │
              ▼                                     ▼
       Landmark Extraction                         NLP
              │                                     │
              ▼                                     ▼
       ST-GCN + Transformer                 Gloss / Sign Mapping
              │                                     │
              ▼                                     ▼
        Sign / Gloss Output                    Sign Motion
              │                                     │
              ▼                                     ▼
             NLP                              3D Avatar
              │
              ▼
       English Text / Speech

# Core Technologies

Area

Technologies

Programming

Python, JavaScript / Dart

Computer Vision

OpenCV, MediaPipe Holistic

Deep Learning

PyTorch / TensorFlow

Sequence Modelling

Transformers, ST-GCN

NLP

Hugging Face / NLP processing

Speech

Speech-to-Text, Text-to-Speech

3D Graphics

Three.js

Mobile Target

Flutter

API / Backend

Decoupled API architecture

Development

Git / GitHub

# Role of Each Technology

OpenCV: Used for handling camera/video input and basic image/video processing.

MediaPipe: Used as the perception layer to extract body landmarks from video, including hand, face, and pose information.

ST-GCN: Used to model the spatial relationships between body joints and their changes over time.

Transformer: Used for sequence modelling and contextual understanding of continuous sign-language movements.

NLP: Used to process language and convert between sign-language gloss representations and natural-language text.

Speech-to-Text / Text-to-Speech: Used to support voice input and spoken output.

Three.js: Used to render and control the proposed 3D animated avatar in the browser.

Dataset: The continuous sign-language datasets as the data source for the project and specifically mentions ISL-CSLTR as an example for the  data preparation.

The final dataset selection, preprocessing strategy, and exact train/validation/test split are to be established during implementation.

# Expected data pipeline:

Raw Sign-Language Video
        ↓
Video Validation
        ↓
Frame Extraction
        ↓
MediaPipe Landmark Extraction
        ↓
Normalization / Preprocessing
        ↓
Sequence Formation
        ↓
Training / Validation / Testing

# Model Development Plan

The project will be developed incrementally:

Dataset exploration and preprocessing
            
Landmark extraction and visualization

Baseline recognition model

ST-GCN model

Transformer-based model

Hybrid ST-GCN + Transformer experiments

Sign/gloss recognition and translation

Speech integration

3D avatar prototype

API integration

Web/mobile user interface

End-to-end testing and evaluation

Evaluation

Planned evaluation includes:

Recognition accuracy

Precision

Recall

F1-score

Confusion matrix

Translation performance

Real-time inference performance

Landmark tracking frame rate

The project will also evaluate practical limitations such as lighting, camera visibility, body occlusion, and latency.

Project Scope

In Scope

Real-time sign-language recognition

Camera-based landmark extraction

Continuous gesture recognition

Sign-to-text translation

Sign-to-speech output

Text/speech-to-sign processing

3D avatar-based sign output

Web prototype

Future mobile deployment

# Limitations

Performance depends on camera visibility and lighting.

Heavy body/hand occlusion can reduce landmark quality.

Continuous sign translation requires substantial training data and experimentation.

3D sign generation is more complex than simple sign recognition.

# Product Modules

Module 1 — Vision & Gesture Engine

Video
  ↓
Landmark Extraction
  ↓
Sign Recognition
  ↓
Speech / Text

Module 2 — Speech & Avatar Engine

Voice / Text
  ↓
Speech-to-Text
  ↓
NLP
  ↓
Gloss / Sign Representation
  ↓
3D Avatar

Module 3 — User Client

A clean dual-view interface for:

camera-based sign input

translated text/speech

text/voice input

3D avatar output

# Hardware Requirements

The proposal identifies the following preliminary requirements:

Development workstation / GPU-enabled system

Standard 720p/1080p RGB webcam

Existing computer resources where available

# Software Requirements

Python

OpenCV

MediaPipe

PyTorch / TensorFlow

Transformer libraries

NLP libraries

Web Speech API / speech tools

Three.js

Flutter for the future mobile target

# Planned Repository Structure

AI-Based-Sign-Language-Translator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── landmarks/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── api/
├── frontend/
├── models/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore

# Future Development

The next phases will focus on:

Finalizing the system architecture and API design

Preparing and curating continuous sign-language datasets

Extracting spatial coordinates with MediaPipe

Building the initial WebGL / Three.js avatar prototype

Identifying latency bottlenecks

Investigating model optimization and quantization

# Research Direction

The project is designed as a research-oriented prototype rather than a clinical or universal translation system. The intended contribution is the integration of:

Computer Vision + Landmark-Based Representation + ST-GCN + Transformer Sequence Modelling + NLP + 3D Avatar Synthesis

into a single bidirectional sign-language communication framework.

Team

Vanshraj Singh Tomar - BTECH CSE (IILM UNVERSITY)                                                                                   2024-2028

Arpit Dubey — BTECH CSE (IILM UNVERSITY)                                                                                            2024-2028

Aditya Kishor — BTECH CSE (IILM UNVERSITY)                                                                                          2024-2028

Khushi Shekhawat — BTECH CSE (IILM UNVERSITY)                                                                                       2024-2028

Project Code: PT442

Programme: B.Tech. CSE Core

Academic Session: 2026–27

University: IILM University, Greater Noida

Project Supervisor: Mr. Nishant Upadhyay

Project Is In Develpoment Phases Readme File Update Time To Time Till Production..