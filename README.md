# Dynamic Pricing Model Optimization with Gradient Descent

##  Project Overview
This project implements a **dynamic pricing regression model from scratch using Gradient Descent**.  
The goal is to **predict the optimal price of a service** based on multiple numerical factors, and to **expose this prediction through a REST API**.

The project focuses on:
- Understanding Gradient Descent mechanics
- Comparing **Batch**, **Stochastic**, and **Mini-Batch Gradient Descent**
- Analyzing convergence behavior and learning rate impact
- Deploying a trained model behind an API endpoint

---

##  Problem Statement
Predict the optimal price of a service based on multiple input variables such as demand-related features, time-based factors, or other numerical indicators.

---

##  Technical Objectives
- Implement **Linear Regression with Gradient Descent from scratch**
- Compare:
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-Batch Gradient Descent
- Train and evaluate the model
- Save the trained model
- Build an API to serve predictions

---

##  Deployment

The application is deployed on **Render** and publicly accessible.
- Backend: Flask REST API
- Model: Trained Gradient Descent regression model
- Frontend: Web interface for user input and price prediction

🔗 Live URL: https://dynamic-pricing-app-tkhs.onrender.com/

---

##  Project Overview

![Frontend Interface](images/frontend.png)

This project implements a **dynamic pricing regression model from scratch using Gradient Descent**.  
The frontend serves as the user-facing layer, enabling interaction with the pricing model
through a REST API to retrieve real-time price predictions.
