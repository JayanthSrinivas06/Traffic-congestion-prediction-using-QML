# Quantum-Enhanced Traffic Congestion Prediction 🚦⚛️

A cutting-edge intelligent traffic analysis pipeline that integrates classical Computer Vision (YOLOv8 object tracking) for feature extraction with a Quantum Machine Learning (QML) Neural Network to accurately predict intersection congestion states.

## 🌟 The Architecture

This system leverages a hybrid Classical-Quantum pipeline to extract dense physical mechanics from traffic intersections and map them into multidimensional Hilbert space for robust state classification.

### System Flow
```mermaid
graph TD;
    %% Data Collection & Vision
    A[Traffic Camera Video Data] --> B[YOLOv8 + ByteTrack]
    B -->|Object Det. & Tracking| C[Frame-by-Frame Vehicle Check]
    
    %% Feature Extraction Pipeline
    C --> D[Feature Engineering Engine]
    D -->|Count Distinct IDs| E(Traffic Density)
    D -->|Compute Track Deltas| F(Avg Velocity)
    D -->|Compute Dispersion| G(Velocity Variance)
    
    %% Quantum Processing
    E --> H[MinMaxScaler Processing]
    F --> H
    G --> H
    
    H -->|Feature Encoding| I[Quantum Circuit]
    
    subgraph Quantum Neural Network [QML Backend]
        I --> J[ZZFeatureMap <br/> 3 Qubits, 2 Reps]
        J --> K[RealAmplitudes Ansatz <br/> 3 Qubits, 4 Reps]
        K --> L[Aer Sampler Primitive]
    end
    
    %% Output
    L --> M{State Probabilities}
    M -->|Class 0| N[Low Congestion]
    M -->|Class 1| O[Medium Congestion]
    M -->|Class 2| P[High Congestion]

    %% Styling
    classDef classical fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#0f172a
    classDef quantum fill:#1e1e2f,stroke:#8a2be2,stroke-width:2px,color:#fff
    classDef result fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    
    class A,B,C,D,E,F,G,H classical
    class I,J,K,L,M quantum
    class N,O,P result
```

### 1. Classical Vision Node (Offloaded)
We process real-world intersection videos utilizing `ultralytics` **YOLOv8n** to robustly detect and track active instances of vehicles (Cars, Motorcycles, Buses, and Trucks). Using bounding box coordinate differentials over strict frame-rate timings, we compute the following core metrics per video-window:
- **Active Vehicles (Density)**: The total unique objects localized.
- **Average Velocity**: The combined pixel/second trajectory speed vectors.
- **Velocity Variance**: Evaluates how consistently traffic is moving or stopping.

### 2. The Quantum Neural Network
Once features are compiled and scaled, they are funneled into a **Qiskit `SamplerQNN`**. 
- The data is embedded using a robust non-linear **`ZZFeatureMap`** to translate the classical metrics into a higher-dimensional quantum state. 
- A **`RealAmplitudes`** parameterized ansatz serves as the tunable weights of our neural network, consisting of entangling gates.
- We then execute the parameterized circuit and sample the probable measurements (`AerSampler`) to map to our 3 final congestion classes.

## 🚀 Live Web Interface

We ship a complete **FastAPI** + **Vanilla HTML/CSS/JS** production app that encapsulates our trained QML model.

1. Navigate to the `backend/` folder.
2. Install dependencies: `pip install -r requirements.txt` (or if none, install `fastapi uvicorn qiskit qiskit_machine_learning pandas`)
3. Run the backend: `python main.py`
4. Open **http://localhost:8000** to utilize the manual data-entry interface and evaluate custom intersection densities directly off the Quantum Sampler!

## 📂 Repository Contents
- **`QML/qml.ipynb`**: Complete training pipeline and Jupyter implementation of the Quantum Network.
- **`YOLO/tracking.ipynb`**: Original vision extraction notebooks used to formulate our datasets.
- **`backend/`**: Contains the decoupled production environment logic.
- **`qml_trained_weights.npy`**: The frozen target parameters iteratively learned by our `COBYLA` optimizer.
- **`scaler.pkl`**: Standard MinMax scales to constrain numerical extremes prior to phase encoding.
