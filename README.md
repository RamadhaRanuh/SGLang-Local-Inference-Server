# SGLang Exploration Repository

This repository serves as a comprehensive exploration and experimentation platform for SGLang, a powerful framework for efficiently serving Large Language Models (LLMs). The project aims to provide practical examples, benchmarks, and tools for understanding and leveraging SGLang's capabilities.

## Key Features and Explorations

**V0.1 - Server Launch and Inference Testing**

* Provides scripts and code examples for launching the SGLang server and performing basic inference tests.
* Includes tools for analyzing token generation information, enabling in-depth performance analysis.
* Implements robust server termination code for clean and efficient testing.
* Establishes a dedicated SGLang code workspace for organized experimentation.

**V0.2 - Benchmarking and Deployment**

* Introduces Docker Compose for streamlined deployment and environment management.
* Implements basic and concurrent benchmarking scripts to evaluate SGLang's performance under varying workloads.
* Integrates OpenAI API compatibility testing to demonstrate SGLang's versatility.
* Adds LoRA (Low-Rank Adaptation) finetuning code, enabling efficient model customization.

**V0.3 - Frontend Integration and Token Streaming**

* Develops a user-friendly frontend interface for interacting with the SGLang server.
* Implements token streaming functionality, providing a real-time, interactive user experience.

**V0.4 - Batching Feature Integration**

* Enhanced the frontend to include batching capabilities, demonstrating SGLang's efficient handling of multiple concurrent requests.

**V0.5 - Monitoring and Visualization**

* Added Docker Compose configuration for Prometheus and Grafana, enabling comprehensive monitoring of SGLang server performance.
* Provides a pre-configured Grafana dashboard for visualizing key metrics such as request latency, throughput, and resource utilization.

## Repository Structure

* **`benchmarks/`:** Contains scripts and data for benchmarking SGLang's performance.
* **`docker/`:** Includes Docker Compose files for server deployment and monitoring.
* **`finetuning/`:** Holds code and scripts for LoRA finetuning.
* **`frontend/`:** Contains the frontend application with token streaming and batching functionalities.
* **`inference_tests/`:** Provides scripts for basic and OpenAI API compatibility inference tests.
* **`server_launch/`:** Includes scripts for launching and terminating the SGLang server.
* **`workspace/`:** A dedicated space for SGLang code exploration and development.
* **`monitoring/`:** Holds the docker compose file for prometheus and grafana.

## Purpose

This repository aims to:

* Provide a practical and accessible learning resource for SGLang.
* Demonstrate SGLang's capabilities through real-world examples and benchmarks.
* Facilitate experimentation and exploration of SGLang's features.
* Offer tools and resources for deploying and monitoring SGLang-based applications.

