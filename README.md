# Retrieval-Based Inertial Localization

The project addresses the domain of “Retrieval-based inertial localization” based on IMU inputs.
Hence, it will address the indoor localization problem using inertial sensor IMU inputs only.
This will not use external infrastructure such as Wi-Fi, BLE, and Floorplan etc. The project
estimate the absolute indoor location,

1. Through a retrieval-based approach.
2. Given only a sequence of IMU data.
3. Without using external infrastructure such as Wi-Fi, BLE, Floorplan, etc.

![alt text](https://github.com/FYPUoM18/fyp-retrival-based-localization/blob/develop/architecture/diag.drawio.png)

uvicorn server:app --reload --host 0.0.0.0 --port 8000
